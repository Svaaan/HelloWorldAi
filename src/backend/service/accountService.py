"""Signing in with GitHub, without taking custody of anybody's key.

Why this wraps rather than replaces
-----------------------------------
The identity here has always been a secret the browser holds, and the
coordinator stores only `sha256(key)` -- see submitterService. That is a good
property: reading the database gives an attacker nothing they can act with.

It also means the obvious version of "sign in and get your work back" is
impossible. The coordinator cannot hand your key to a new laptop because it
never had your key. Storing it so that it could would throw away the one thing
that makes the current model worth defending.

So an account does not own a key. It links to the *digest*, and a signed-in
session authorises directly against that digest:

    GitHub identity  ->  account  ->  [submitter_id, ...]     (digests only)

Signed in, the browser needs no key at all: the coordinator already knows which
jobs are yours. Signed out, the key file works exactly as before. Neither path
takes anything away from the other, and the database still holds no credential.

What is stored
--------------
A GitHub numeric id, the login name for display, and the digests linked to it.

Not the access token -- it is used once to ask GitHub who just signed in and is
then discarded, because nothing here needs to act on anyone's behalf on GitHub.
Not the avatar, the email, or anything else in the profile either: the page
shows a name, so a name is what is kept.
"""

from __future__ import annotations

import logging
import os
import secrets
from typing import Optional

import httpx
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from backend.service.submitterService import submitter_id_from_key

logger = logging.getLogger(__name__)

GITHUB_AUTHORIZE = "https://github.com/login/oauth/authorize"
GITHUB_TOKEN = "https://github.com/login/oauth/access_token"
GITHUB_USER = "https://api.github.com/user"

CLIENT_ID = os.getenv("GITHUB_CLIENT_ID", "").strip()
CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "").strip()

# A month. Long enough that signing in is rare; short enough that a session left
# on a shared machine does not last forever.
SESSION_TTL = int(os.getenv("SESSION_TTL_SECONDS", 30 * 24 * 3600))
SESSION_COOKIE = "hw_session"

_SECRET = os.getenv("NODE_TOKEN_SECRET") or secrets.token_urlsafe(32)
_sessions = URLSafeTimedSerializer(secret_key=_SECRET, salt="account-session")
_states = URLSafeTimedSerializer(secret_key=_SECRET, salt="oauth-state")

# The round trip to GitHub and back should take seconds, not hours.
STATE_TTL = 600


class AccountError(Exception):
    """Something went wrong signing in, phrased for the person it happened to."""


def configured() -> bool:
    """Whether this deployment can offer GitHub sign-in at all.

    Checked rather than assumed so the page can hide the button instead of
    offering one that leads to an error, and so a contributor running the stack
    at home is not told to go and register an OAuth application.
    """
    return bool(CLIENT_ID and CLIENT_SECRET)


# --- the round trip --------------------------------------------------------

def start_url(redirect_uri: str) -> tuple[str, str]:
    """Where to send the browser, and the state to check when it comes back.

    The state is signed rather than stored. There is nothing to clean up, it
    survives a restart mid-sign-in, and a forged callback cannot produce one.
    """
    nonce = secrets.token_urlsafe(16)
    state = _states.dumps(nonce)

    query = (
        f"client_id={CLIENT_ID}"
        f"&redirect_uri={redirect_uri}"
        f"&state={state}"
        # No scopes. The only question being asked is "who are you", and
        # asking for more would be asking for access nothing here uses.
        f"&scope="
    )
    return f"{GITHUB_AUTHORIZE}?{query}", state


def check_state(state: str) -> None:
    try:
        _states.loads(state, max_age=STATE_TTL)
    except SignatureExpired:
        raise AccountError("That sign-in took too long. Try again.")
    except BadSignature:
        raise AccountError("That sign-in did not start here. Try again.")


async def identify(code: str, redirect_uri: str) -> dict:
    """Swap the callback code for who GitHub says this is.

    The access token is used once, here, and never stored. Nothing in this
    service acts on GitHub on anybody's behalf, so keeping it would be holding a
    credential for no reason.
    """
    async with httpx.AsyncClient(timeout=15) as client:
        token_response = await client.post(
            GITHUB_TOKEN,
            data={"client_id": CLIENT_ID, "client_secret": CLIENT_SECRET,
                  "code": code, "redirect_uri": redirect_uri},
            headers={"Accept": "application/json"},
        )

        if token_response.status_code != 200:
            raise AccountError("GitHub would not complete the sign-in.")

        access_token = (token_response.json() or {}).get("access_token")
        if not access_token:
            raise AccountError("GitHub did not return a sign-in token.")

        user_response = await client.get(
            GITHUB_USER,
            headers={"Authorization": f"Bearer {access_token}",
                     "Accept": "application/vnd.github+json"},
        )

    if user_response.status_code != 200:
        raise AccountError("GitHub would not say who signed in.")

    user = user_response.json() or {}
    if not user.get("id"):
        raise AccountError("GitHub returned an account with no id.")

    # The id and the name it goes by. Not the avatar URL, not the email, not
    # anything else GitHub sends back: none of it is used, and a field that is
    # stored and never read is a field somebody has to justify in a breach.
    return {
        "github_id": int(user["id"]),
        "login": str(user.get("login") or ""),
    }


# --- the session -----------------------------------------------------------

def issue_session(github_id: int) -> str:
    return _sessions.dumps({"github_id": int(github_id)})


def read_session(token: Optional[str]) -> Optional[int]:
    """The GitHub id a session token proves, or None."""
    if not token:
        return None
    try:
        payload = _sessions.loads(token, max_age=SESSION_TTL)
    except (BadSignature, SignatureExpired):
        return None
    github_id = (payload or {}).get("github_id")
    return int(github_id) if github_id else None


# --- the account -----------------------------------------------------------

async def upsert(db, profile: dict, submitter_key: Optional[str] = None) -> dict:
    """Record who signed in, and link the key their browser is already using.

    Linking on sign-in is what makes this a wrap rather than a replacement: the
    jobs somebody has already sent stay theirs, under the same digest, and the
    account simply learns about them. Nothing migrates and nothing is rewritten.
    """
    account = {
        "_id": int(profile["github_id"]),
        "login": profile.get("login") or "",
    }

    await db.accounts_collection.update_one(
        {"_id": account["_id"]},
        {"$set": account, "$setOnInsert": {"submitter_ids": []}},
        upsert=True,
    )

    if submitter_key:
        try:
            digest = submitter_id_from_key(submitter_key)
        except Exception:                                   # noqa: BLE001
            digest = None                                   # a malformed key is not fatal
        if digest:
            await link(db, account["_id"], digest)

    return await get(db, account["_id"]) or account


async def link(db, github_id: int, submitter_id: str) -> None:
    """Attach a digest to an account, once.

    $addToSet rather than $push: signing in twice from the same browser should
    not produce two of the same digest.
    """
    await db.accounts_collection.update_one(
        {"_id": int(github_id)},
        {"$addToSet": {"submitter_ids": submitter_id}},
    )


async def get(db, github_id: int) -> Optional[dict]:
    return await db.accounts_collection.find_one({"_id": int(github_id)})


async def owns(db, submitter_id: Optional[str]) -> bool:
    """Whether any account has linked this digest.

    Asked by the retention sweep, which keeps a signed-in person's data long
    enough to come back to and an anonymous submitter's only as long as the
    immediate work needs it. Answered by lookup rather than carried on the task,
    because somebody can sign in and link a key *after* the job was sent -- and
    the point of doing that is precisely to keep what they already have.
    """
    if not submitter_id:
        return False

    account = await db.accounts_collection.find_one(
        {"submitter_ids": submitter_id}, {"_id": 1})
    return account is not None


async def submitter_ids(db, github_id: int) -> list:
    account = await get(db, github_id)
    return list((account or {}).get("submitter_ids") or [])


async def primary_submitter_id(db, github_id: int) -> Optional[str]:
    """The digest a signed-in submission is recorded against.

    The first one linked, so that jobs sent from a new device join the work
    already under that identity rather than starting a second pile beside it.
    """
    ids = await submitter_ids(db, github_id)
    return ids[0] if ids else None

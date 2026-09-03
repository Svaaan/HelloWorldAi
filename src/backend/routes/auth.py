"""Signing in with GitHub.

Four routes and no passwords. The account wraps the existing key rather than
replacing it -- see accountService for why that is the only shape that fits a
service whose database deliberately holds no credential.

Nothing here is required. A browser holding a key works exactly as it did, and
a deployment with no OAuth application configured hides the button rather than
offering one that leads to an error.
"""

import logging
import os
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse

from backend.routes.deps import Database, get_db
from backend.service import accountService

logger = logging.getLogger("coordinator")

router = APIRouter()


def _public_base(request: Request) -> str:
    """The address the browser is actually using.

    Not the one this process sees. The coordinator sits behind the dashboard
    proxy, which sits behind Caddy, so `request.url` here is
    http://coordinator:8100 -- a name that exists only inside the compose
    network. Sending GitHub back to it would produce a callback the browser
    cannot follow, and marking the session cookie insecure because the internal
    hop is plain HTTP.

    PUBLIC_URL is set explicitly rather than guessed from forwarded headers,
    because the proxy does not pass those on and a guess that is wrong here
    breaks sign-in in a way that is tedious to diagnose.
    """
    configured = os.getenv("PUBLIC_URL", "").strip().rstrip("/")
    if configured:
        return configured
    # Local development, talking to the coordinator directly.
    return str(request.base_url).rstrip("/")


def _redirect_uri(request: Request) -> str:
    """Where GitHub sends the browser back to.

    Must match the callback URL registered on the OAuth application exactly.
    """
    return f"{_public_base(request)}/auth/github/callback"


@router.get("/auth/config")
async def auth_config():
    """Whether this deployment offers sign-in, for the header to decide with."""
    return {"github": accountService.configured()}


@router.get("/auth/github/start")
async def github_start(request: Request):
    if not accountService.configured():
        raise HTTPException(
            status_code=503,
            detail=("GitHub sign-in is not set up on this deployment. It needs "
                    "GITHUB_CLIENT_ID and GITHUB_CLIENT_SECRET."))

    url, _state = accountService.start_url(_redirect_uri(request))
    return RedirectResponse(url, status_code=307)


@router.get("/auth/github/callback", name="github_callback")
async def github_callback(
    request: Request,
    code: Optional[str] = None,
    state: Optional[str] = None,
    db: Database = Depends(get_db),
    x_submitter_key: Optional[str] = Header(default=None),
):
    """Back from GitHub: identify, link, and set a session.

    The key currently in the browser cannot be read from here -- a redirect
    carries no custom headers -- so linking it is done by the page afterwards,
    through /auth/link. This route establishes who signed in and nothing more.
    """
    if not accountService.configured():
        raise HTTPException(status_code=503, detail="GitHub sign-in is not set up.")

    if not code or not state:
        raise HTTPException(status_code=400, detail="That sign-in was incomplete.")

    try:
        accountService.check_state(state)
        profile = await accountService.identify(code, _redirect_uri(request))
        account = await accountService.upsert(db, profile)
    except accountService.AccountError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:                                  # noqa: BLE001
        logger.exception("GitHub sign-in failed")
        raise HTTPException(status_code=502, detail=f"Sign-in failed: {e}")

    response = RedirectResponse("/workspace", status_code=303)
    response.set_cookie(
        accountService.SESSION_COOKIE,
        accountService.issue_session(profile["github_id"]),
        max_age=accountService.SESSION_TTL,
        httponly=True,          # no script needs to read it, so none may
        samesite="lax",         # survives the redirect back from GitHub
        # From the public address, not this hop: the browser may be on HTTPS
        # while the request reaching the coordinator is plain HTTP.
        secure=_public_base(request).startswith("https://"),
        path="/",
    )

    logger.info("Signed in %s (%s), %d key(s) linked",
                account.get("login"), profile["github_id"],
                len(account.get("submitter_ids") or []))
    return response


@router.post("/auth/link")
async def link_key(
    db: Database = Depends(get_db),
    hw_session: Optional[str] = Cookie(default=None),
    x_submitter_key: Optional[str] = Header(default=None),
):
    """Attach the key this browser holds to the signed-in account.

    This is what makes signing in a wrap rather than a fresh start: the jobs
    already sent under this key stay exactly where they are, under the same
    digest, and the account simply learns that they are yours. Nothing is
    migrated and no task is rewritten.

    Only the digest is stored. The key itself is not sent anywhere it is not
    already sent on every other request, and is not kept.
    """
    github_id = accountService.read_session(hw_session)
    if not github_id:
        raise HTTPException(status_code=401, detail="Not signed in.")

    from backend.service.submitterService import read_submitter_key
    digest = read_submitter_key(x_submitter_key)
    if not digest:
        return {"linked": False, "reason": "no key in this browser"}

    await accountService.link(db, github_id, digest)
    account = await accountService.get(db, github_id)
    return {"linked": True,
            "keys": len(account.get("submitter_ids") or [])}


@router.get("/auth/me")
async def me(db: Database = Depends(get_db),
             hw_session: Optional[str] = Cookie(default=None)):
    """Who is signed in, if anybody."""
    github_id = accountService.read_session(hw_session)
    if not github_id:
        return {"signed_in": False, "github": accountService.configured()}

    account = await accountService.get(db, github_id)
    if not account:
        # Signed in against an account that no longer exists.
        return {"signed_in": False, "github": accountService.configured()}

    return {
        "signed_in": True,
        "github": True,
        "login": account.get("login"),
        "keys_linked": len(account.get("submitter_ids") or []),
    }


@router.post("/auth/sign-out")
async def sign_out(request: Request):
    """Forget the session. The key in the browser is untouched.

    Deliberately separate from the existing sign-out, which clears the keys
    themselves. Ending a session should not throw away an identity that cannot
    be reissued.
    """
    response = JSONResponse({"signed_out": True})
    response.delete_cookie(accountService.SESSION_COOKIE, path="/")
    return response

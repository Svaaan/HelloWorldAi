"""Signing in wraps the key model; it does not replace it.

The identity here has always been a secret the browser holds, with only
`sha256(key)` reaching the database. That is worth keeping: reading the database
gives an attacker nothing they can act with.

It also rules out the obvious version of an account. The coordinator cannot hand
somebody their key on a new laptop, because it never had their key. Storing it so
that it could would throw away the only thing that makes the current model worth
defending.

So an account links to the *digest*, and a signed-in session authorises against
that digest directly. These check that the wrap holds: that no key is stored,
that a key still works without an account, that an account still works without a
key, and that signing in does not quietly move anybody's existing work.
"""

import asyncio
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service import accountService                      # noqa: E402
from backend.service.submitterService import submitter_id_from_key  # noqa: E402


A_KEY = "k" * 40
ANOTHER_KEY = "j" * 40


class FakeAccounts:
    """Just enough of a collection for the two update shapes used here."""

    def __init__(self):
        self.docs = {}

    async def update_one(self, query, update, upsert=False):
        key = query["_id"]
        doc = self.docs.get(key)

        if doc is None:
            if not upsert:
                return
            doc = dict(update.get("$setOnInsert") or {})
            doc["_id"] = key
            self.docs[key] = doc

        doc.update(update.get("$set") or {})

        for field, value in (update.get("$addToSet") or {}).items():
            existing = doc.setdefault(field, [])
            if value not in existing:
                existing.append(value)

    async def find_one(self, query):
        return self.docs.get(query["_id"])


class FakeDb:
    def __init__(self):
        self.accounts_collection = FakeAccounts()


PROFILE = {"github_id": 4242, "login": "svaaan"}


def run(coro):
    """Drive one coroutine to completion.

    The suite has no async plugin and does not need one here: these are plain
    calls against a fake collection, not a running server. Same helper as
    test_quota.py, for the same reason.
    """
    return asyncio.new_event_loop().run_until_complete(coro)


# --- what is and is not stored ---------------------------------------------

def test_the_key_itself_is_never_stored():
    """The whole point. An account holds the digest and nothing more."""
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))

    stored = repr(db.accounts_collection.docs)
    assert A_KEY not in stored, (
        "the submitter key reached the database. The coordinator is supposed to "
        "be unable to act as anybody, and storing this would end that.")

    account = run(accountService.get(db, 4242))
    assert submitter_id_from_key(A_KEY) in account["submitter_ids"]


def test_no_github_token_is_kept():
    """It answers one question -- who is this -- and is then finished with."""
    import inspect
    source = inspect.getsource(accountService.upsert)
    assert "access_token" not in source, (
        "the account record should not carry a GitHub token; nothing here acts "
        "on anybody's behalf on GitHub")


# --- the wrap ---------------------------------------------------------------

def test_signing_in_links_the_key_already_in_the_browser():
    """Existing jobs stay yours: same digest, nothing migrated."""
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))

    assert (run(accountService.primary_submitter_id(db, 4242))
            == submitter_id_from_key(A_KEY))


def test_a_second_browser_adds_a_key_rather_than_replacing_one():
    """Somebody with two machines owns both piles of work, not the newer one."""
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))
    run(accountService.upsert(db, PROFILE, submitter_key=ANOTHER_KEY))

    ids = run(accountService.submitter_ids(db, 4242))
    assert len(ids) == 2
    assert submitter_id_from_key(A_KEY) in ids
    assert submitter_id_from_key(ANOTHER_KEY) in ids


def test_the_first_key_stays_primary():
    """New work joins the pile already under this identity.

    If the newest key won, signing in on a fresh laptop would start a second
    collection of jobs beside the first, which is the confusion accounts are
    meant to remove.
    """
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))
    run(accountService.upsert(db, PROFILE, submitter_key=ANOTHER_KEY))

    assert (run(accountService.primary_submitter_id(db, 4242))
            == submitter_id_from_key(A_KEY))


def test_signing_in_twice_does_not_duplicate_a_key():
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))

    assert len(run(accountService.submitter_ids(db, 4242))) == 1


def test_an_account_with_no_key_yet_is_not_broken():
    """Signed in on a browser that has never sent anything."""
    db = FakeDb()
    run(accountService.upsert(db, PROFILE))

    assert run(accountService.submitter_ids(db, 4242)) == []
    assert run(accountService.primary_submitter_id(db, 4242)) is None


def test_a_malformed_key_does_not_break_the_sign_in():
    """A short or corrupted key is worth ignoring, not worth failing on."""
    db = FakeDb()
    account = run(accountService.upsert(db, PROFILE, submitter_key="tooshort"))

    assert account["_id"] == 4242
    assert run(accountService.submitter_ids(db, 4242)) == []


# --- sessions ---------------------------------------------------------------

def test_a_session_round_trips():
    token = accountService.issue_session(4242)
    assert accountService.read_session(token) == 4242


def test_a_forged_session_is_refused():
    assert accountService.read_session("not-a-token") is None
    assert accountService.read_session("") is None
    assert accountService.read_session(None) is None


def test_a_tampered_session_is_refused():
    token = accountService.issue_session(4242)
    # Change the payload and keep the signature.
    broken = ("A" + token[1:]) if token[0] != "A" else ("B" + token[1:])
    assert accountService.read_session(broken) is None


def test_the_oauth_state_is_signed_not_stored():
    """Nothing to clean up, and it survives a restart mid-sign-in."""
    _url, state = accountService.start_url("https://example.test/cb")
    accountService.check_state(state)          # does not raise

    with pytest.raises(accountService.AccountError):
        accountService.check_state("forged")


def test_sign_in_asks_for_no_scopes():
    """The only question is who you are.

    Any scope would be asking for access to somebody's repositories that
    nothing in this service uses.
    """
    url, _state = accountService.start_url("https://example.test/cb")
    assert "scope=" in url
    assert "repo" not in url and "user:email" not in url


# --- degrading ---------------------------------------------------------------

def test_a_deployment_without_an_oauth_app_says_so(monkeypatch):
    """The page hides the button rather than offering a broken one."""
    monkeypatch.setattr(accountService, "CLIENT_ID", "")
    monkeypatch.setattr(accountService, "CLIENT_SECRET", "")
    assert accountService.configured() is False

    monkeypatch.setattr(accountService, "CLIENT_ID", "abc")
    monkeypatch.setattr(accountService, "CLIENT_SECRET", "def")
    assert accountService.configured() is True


# --- the key still wins ------------------------------------------------------

def test_a_held_key_takes_precedence_over_a_session():
    """Loading a key file means acting as that key.

    Somebody who has just loaded a key means to use it, and a session left from
    an earlier sign-in must not quietly redirect their work elsewhere.
    """
    import inspect
    from backend.routes import deps

    source = inspect.getsource(deps.submitter_or_session)
    key_line = source.index("by_key = read_submitter_key")
    session_line = source.index("read_session")

    assert key_line < session_line, (
        "the session is consulted before the key, so a stale sign-in would "
        "override a key the person just loaded")
    assert "if by_key:" in source


# --- what signing in actually reaches ----------------------------------------
#
# An account that does not change what you can see is decoration. These are
# about the payoff: the desktop's jobs showing up on the laptop.

def _scope(monkeypatch, db, key=None, session=None):
    """Run the real dependency against a fake database."""
    from backend.routes import deps
    monkeypatch.setattr(deps, "Database", db)
    return run(deps.submitter_scope(x_submitter_key=key, hw_session=session))


def test_a_key_on_its_own_is_its_own_scope(monkeypatch):
    """Nothing about this changed for somebody who never signs in."""
    assert _scope(monkeypatch, FakeDb(), key=A_KEY) == [submitter_id_from_key(A_KEY)]


def test_nobody_at_all_has_no_scope(monkeypatch):
    assert _scope(monkeypatch, FakeDb()) == []


def test_signing_in_reaches_work_sent_from_another_machine(monkeypatch):
    """The laptop, holding no key, sees what the desktop sent.

    This is the whole reason for the account. Before it, the two machines were
    two strangers and there was no way for them not to be.
    """
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))       # desktop

    laptop = _scope(monkeypatch, db, session=accountService.issue_session(4242))
    assert laptop == [submitter_id_from_key(A_KEY)]


def test_a_second_machine_sees_both_piles(monkeypatch):
    """Not one or the other. Both keys are the same person's work."""
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))
    run(accountService.upsert(db, PROFILE, submitter_key=ANOTHER_KEY))

    scope = _scope(monkeypatch, db,
                   key=ANOTHER_KEY,
                   session=accountService.issue_session(4242))

    assert set(scope) == {submitter_id_from_key(A_KEY),
                          submitter_id_from_key(ANOTHER_KEY)}
    assert scope[0] == submitter_id_from_key(ANOTHER_KEY), (
        "the key in the browser should come first, so a caller that wants one "
        "answer gets the one it would have had before accounts existed")


def test_a_stranger_signing_in_does_not_reach_your_jobs(monkeypatch):
    """An account only ever reaches digests it has been linked to."""
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))

    stranger = {"github_id": 9999, "login": "someone-else"}
    run(accountService.upsert(db, stranger))

    assert _scope(monkeypatch, db,
                  session=accountService.issue_session(9999)) == []


def test_an_expired_or_forged_session_reaches_nothing(monkeypatch):
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))

    assert _scope(monkeypatch, db, session="forged") == []


def test_the_scope_holds_no_duplicates(monkeypatch):
    """Holding the very key the account is linked to is the common case."""
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))

    scope = _scope(monkeypatch, db,
                   key=A_KEY, session=accountService.issue_session(4242))
    assert scope == [submitter_id_from_key(A_KEY)]


def test_reads_span_the_account_and_writes_do_not():
    """Two questions, two dependencies, on purpose.

    A job is filed under one digest -- it has one submitter_id column and there
    is no sensible way to write a set into it. Reading is the opposite: the
    point of signing in is that it spans everything you own.
    """
    import inspect
    from backend.routes import tasks

    listing = inspect.signature(tasks.list_my_tasks).parameters
    assert "scope" in listing, "listing jobs should span the account"

    submitting = inspect.signature(tasks.submit_task_anywhere).parameters
    assert "submitter" in submitting, "a submitted job needs one owner, not a set"


def test_nothing_from_the_profile_is_kept_beyond_a_name():
    """A field stored and never read is one to justify in a breach.

    GitHub hands back an avatar URL, a company, a location and more. The page
    shows a name, so a name is what is kept.
    """
    db = FakeDb()
    run(accountService.upsert(db, PROFILE, submitter_key=A_KEY))

    doc = db.accounts_collection.docs[4242]
    assert set(doc) == {"_id", "login", "submitter_ids"}, (
        f"unexpected fields stored on an account: {sorted(doc)}")

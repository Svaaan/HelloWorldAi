"""How long a finished job's data is kept, and whose it is.

It was a flat hour for everybody. That is the right number for the two things
that happen immediately -- verification reads the holdout, a retry reuses the
same split -- and the wrong one for the feature those exist to serve.

"Adjust and run" trains the same data again with different settings, scored
against the same withheld rows so the runs compare. Come back the next morning
and the data is gone, so the file goes up again. The loop was built; an
hour-long timer was undoing it.

These pin the policy, because it decides when somebody's data is destroyed and
that is not a thing to get subtly wrong.
"""

import os
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service import retention                            # noqa: E402


NOW = datetime(2026, 9, 4, 12, 0, 0)


def finished(minutes_ago):
    return NOW - timedelta(minutes=minutes_ago)


# --- the two windows --------------------------------------------------------

def test_an_anonymous_job_keeps_the_old_hour():
    """Nothing changed for somebody who never signed in.

    Not meanness: there is nobody to keep it for. The key that owns the job may
    already be gone from the browser that made it, and nothing can reach them.
    """
    assert retention.should_forget(finished(61), NOW,
                                   has_account=False, rank=0) is True
    assert retention.should_forget(finished(59), NOW,
                                   has_account=False, rank=0) is False


def test_a_signed_in_job_survives_the_night():
    """The case the whole change exists for."""
    overnight = finished(16 * 60)
    assert retention.should_forget(overnight, NOW,
                                   has_account=True, rank=0) is False, (
        "an idea the next morning should not need the file uploading again")


def test_a_signed_in_job_still_expires_eventually():
    assert retention.should_forget(finished(8 * 24 * 60), NOW,
                                   has_account=True, rank=0) is True


def test_the_windows_are_the_right_way_round():
    assert retention.window(True) > retention.window(False)


# --- the cap ----------------------------------------------------------------

def test_an_owner_cannot_keep_an_unbounded_pile():
    """Storage is real. Past the cap the oldest go early, clock or no clock.

    Without this, signing in turns a bounded cost into an unbounded one, and
    the feature becomes a bill.
    """
    just_finished = finished(61)

    assert retention.should_forget(just_finished, NOW, has_account=True,
                                   rank=retention.KEEP_PER_OWNER - 1) is False
    assert retention.should_forget(just_finished, NOW, has_account=True,
                                   rank=retention.KEEP_PER_OWNER) is True


def test_the_cap_drops_the_oldest_rather_than_the_newest():
    """rank 0 is the most recent, and it is the one that survives.

    The other way round would delete what somebody is working on and keep what
    they have finished with.
    """
    old = finished(61)
    assert retention.should_forget(old, NOW, has_account=True, rank=0) is False
    assert retention.should_forget(old, NOW, has_account=True,
                                   rank=retention.KEEP_PER_OWNER + 5) is True


# --- edges ------------------------------------------------------------------

def test_a_row_with_no_finish_time_is_not_immortal():
    """A task holding a dataset with no finished_at would never be swept."""
    assert retention.should_forget(None, NOW, has_account=True, rank=0) is True


def test_the_expiry_is_offered_to_the_page():
    """So a countdown can be shown rather than the deletion being discovered
    by pressing a button and being told the data is gone."""
    at = retention.expires_at(finished(0), has_account=False)
    assert at.replace(tzinfo=None) == NOW + timedelta(
        minutes=retention.SHORT_MINUTES)

    assert retention.expires_at(None, has_account=True) is None


def test_the_expiry_says_which_zone_it_is_in():
    """A naive datetime serialises with no offset, and JavaScript reads that as
    local time. Two hours east of UTC, "40 minutes left" became "two hours ago"
    and the countdown rendered nothing."""
    from datetime import timezone

    at = retention.expires_at(finished(0), has_account=True)
    assert at.tzinfo is not None
    assert at.utcoffset().total_seconds() == 0

    # ISO form is what actually reaches the browser.
    assert at.isoformat().endswith("+00:00")


def test_it_describes_itself_in_units_people_use():
    assert retention.describe(False) == "60 minutes"
    assert retention.describe(True) == "7 days"


def test_the_short_window_still_reads_the_old_setting():
    """Deployments already set DATASET_RETENTION_MINUTES; it must keep working."""
    import inspect
    source = inspect.getsource(retention)
    assert "DATASET_RETENTION_MINUTES" in source


# --- who the sweep thinks owns a job ---------------------------------------
#
# The window depends on whether any account has linked the job's digest, and
# that is a lookup rather than something carried on the task -- because
# somebody can sign in and link a key *after* sending the job, and keeping what
# they already have is the whole point of doing that.

import asyncio                                                   # noqa: E402

from backend.service import accountService                       # noqa: E402


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class FakeAccounts:
    def __init__(self, docs):
        self.docs = docs
        self.queries = []

    async def find_one(self, query, projection=None):
        self.queries.append(query)
        wanted = query.get("submitter_ids")
        for doc in self.docs:
            if wanted in (doc.get("submitter_ids") or []):
                return {"_id": doc["_id"]}
        return None


class FakeDb:
    def __init__(self, docs):
        self.accounts_collection = FakeAccounts(docs)


def test_a_linked_digest_is_owned():
    db = FakeDb([{"_id": 4242, "submitter_ids": ["digest_a", "digest_b"]}])
    assert run(accountService.owns(db, "digest_b")) is True


def test_an_unlinked_digest_is_not():
    db = FakeDb([{"_id": 4242, "submitter_ids": ["digest_a"]}])
    assert run(accountService.owns(db, "digest_zzz")) is False


def test_no_digest_at_all_is_not_owned():
    """An anonymous API submission has no submitter_id, and must not be
    treated as somebody's to keep."""
    db = FakeDb([{"_id": 4242, "submitter_ids": ["digest_a"]}])
    assert run(accountService.owns(db, None)) is False
    assert db.accounts_collection.queries == [], "it should not even ask"


def test_it_asks_by_digest_and_takes_only_the_id():
    """Indexed on submitter_ids, and it does not drag whole accounts back."""
    db = FakeDb([{"_id": 4242, "submitter_ids": ["digest_a"]}])
    run(accountService.owns(db, "digest_a"))
    assert db.accounts_collection.queries == [{"submitter_ids": "digest_a"}]

"""How much one submitter may ask of the network, and what happens to work
addressed to a machine that has gone away.

Two problems, both of which only appear once the coordinator has a public
address.

The first: a submitter key is minted by the browser, costs nothing, and can be
regenerated forever. Requiring one to upload stopped anonymous abuse; it did
nothing about cheap abuse. One person with a script could queue unlimited jobs,
each carrying up to 512 MB, spending strangers' electricity and filling the
disk. There was no counter anywhere, and uploads did not even record who stored
them, so there was nothing to count.

The second: a job queued to a node that never comes back was never touched
again. requeue_stale_tasks looked only at 'running', and _redispatch fired only
on an explicit decline, so a task sat 'pending' forever on a machine that had
been switched off. The submitter saw a job that never started and no reason
why.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta

import pytest

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "..", "src")
sys.path.insert(0, SRC)
os.environ.setdefault("ENV", "test")

from backend.service import quota                     # noqa: E402


def run(coro):
    """Drive one coroutine to completion.

    The suite has no async plugin and does not need one for this: these are
    plain calls with a fake database, not a running server.
    """
    return asyncio.new_event_loop().run_until_complete(coro)


def read(*parts):
    with open(os.path.join(SRC, *parts), encoding="utf-8") as handle:
        return handle.read()


# --- fakes: enough of motor to count with -------------------------------

class FakeCursor:
    def __init__(self, rows):
        self._rows = list(rows)

    def __aiter__(self):
        async def gen():
            for row in self._rows:
                yield row
        return gen()


class FakeCollection:
    def __init__(self, rows=None):
        self.rows = list(rows or [])

    async def count_documents(self, query):
        return sum(1 for r in self.rows if _matches(r, query))

    def find(self, query, _projection=None):
        return FakeCursor([r for r in self.rows if _matches(r, query)])

    def aggregate(self, pipeline):
        match = pipeline[0]["$match"]
        total = sum(int(r.get("metadata", {}).get("bytes", 0))
                    for r in self.rows if _matches(r, match))
        return FakeCursor([{"_id": None, "total": total}])


def _get(row, dotted):
    value = row
    for part in dotted.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def _matches(row, query):
    for key, want in query.items():
        got = _get(row, key)
        if isinstance(want, dict):
            if "$nin" in want and got in want["$nin"]:
                return False
            if "$gte" in want and (got is None or got < want["$gte"]):
                return False
        elif got != want:
            return False
    return True


class FakeDb:
    def __init__(self, artifacts=None, tasks=None):
        self.db = {"artifacts.files": FakeCollection(artifacts)}
        self.tasks_collection = FakeCollection(tasks)


def artifact(uploader, size_mb, hours_ago=0):
    return {"metadata": {"uploader": uploader,
                         "bytes": size_mb * 1024 * 1024,
                         "uploaded_at": datetime.utcnow() - timedelta(hours=hours_ago)}}


def task(submitter, status="pending", hours_ago=0):
    return {"submitter_id": submitter, "status": status,
            "submitted_at": datetime.utcnow() - timedelta(hours=hours_ago)}


# --- who is counted ------------------------------------------------------

def test_a_node_is_never_throttled():
    """A node uploads weights only for work it was handed.

    Its usage is already bounded by what the coordinator gave it, and refusing
    the upload would throw away a job that has already finished training on
    somebody's GPU.
    """
    db = FakeDb(artifacts=[artifact("node:n1", 5000)])

    run(quota.check_upload(db, "node:n1", 5000 * 1024 * 1024))   # must not raise


def test_a_submitter_under_the_limit_is_left_alone():
    db = FakeDb(artifacts=[artifact("submitter:s1", 10)])

    run(quota.check_upload(db, "submitter:s1", 1024))            # must not raise


def test_a_submitter_over_the_daily_limit_is_refused():
    used = quota.UPLOAD_BYTES_PER_DAY // (1024 * 1024)
    db = FakeDb(artifacts=[artifact("submitter:s1", used)])

    with pytest.raises(quota.QuotaExceeded) as excinfo:
        run(quota.check_upload(db, "submitter:s1", 1024 * 1024))

    # The message has to say what to do next, not merely "no".
    assert "daily upload limit" in str(excinfo.value)


def test_yesterdays_uploads_stop_counting():
    """A rolling day, not a total. Otherwise the first heavy week is the last."""
    db = FakeDb(artifacts=[
        artifact("submitter:s1", quota.UPLOAD_BYTES_PER_DAY // (1024 * 1024),
                 hours_ago=30),
    ])

    run(quota.check_upload(db, "submitter:s1", 1024 * 1024))     # must not raise


# --- how many jobs at once ----------------------------------------------

def test_one_submitter_cannot_occupy_the_whole_network():
    db = FakeDb(tasks=[task("s1") for _ in range(quota.ACTIVE_JOBS)])

    with pytest.raises(quota.QuotaExceeded) as excinfo:
        run(quota.check_new_job(db, "s1"))

    assert str(quota.ACTIVE_JOBS) in str(excinfo.value)


def test_finished_jobs_do_not_count_against_the_active_limit():
    db = FakeDb(tasks=[task("s1", status=s) for s in quota.FINISHED] * 5)

    run(quota.check_new_job(db, "s1"))                            # must not raise


def test_one_submitters_jobs_do_not_limit_another():
    db = FakeDb(tasks=[task("someone_else") for _ in range(quota.ACTIVE_JOBS * 2)])

    run(quota.check_new_job(db, "s1"))                            # must not raise


# --- the wiring ----------------------------------------------------------

def test_uploads_record_who_stored_them():
    """Without this there is nothing to count against.

    Requiring a caller to upload was only half of it: the caller was checked at
    the door and then not written on the box.
    """
    source = read("backend", "routes", "artifacts.py")

    assert 'metadata["uploader"] = uploader' in source
    assert source.count("uploader=uploader") >= 2, (
        "both upload paths must pass the caller through to storage"
    )


def test_both_upload_paths_check_the_quota():
    source = read("backend", "routes", "artifacts.py")

    assert source.count("quota.check_upload") == 2
    # Too soon, not too large: 413 would tell the person to shrink their file.
    assert "status_code=429" in source


def test_the_job_quota_sits_on_the_shared_path():
    """Both submit endpoints go through _queue_task, so one check covers them."""
    source = read("backend", "routes", "tasks.py")

    start = source.index("async def _queue_task")
    window = source[start:start + 1400]
    assert "quota.check_new_job" in window

    # and before the dataset is split and written, so a refusal costs nothing
    check_at = source.index("quota.check_new_job")
    split_at = source.index("prepare_dataset_split(")
    assert check_at < split_at


# --- work addressed to a machine that vanished --------------------------

def test_pending_jobs_on_a_dead_node_are_not_ignored():
    source = read("backend", "routes", "tasks.py")

    start = source.index("async def requeue_stale_tasks")
    window = source[start:start + 6000]

    assert '{"status": "pending"}' in window, (
        "the loop only ever looked at 'running', so a job queued to a node "
        "that never came back sat pending forever"
    )
    assert "NODE_GONE_MINUTES" in window


def test_a_chosen_node_is_not_silently_overridden():
    """If the submitter named a machine, their work does not go elsewhere.

    It fails with a reason instead. Moving it would quietly undo a choice they
    made deliberately -- they may have picked that machine for a reason.
    """
    source = read("backend", "routes", "tasks.py")

    start = source.index("async def requeue_stale_tasks")
    window = source[start:start + 6000]

    assert 'task.get("placement") == "auto"' in window
    assert "stopped reporting in" in window

"""Offering a declined job to a different machine.

A decline used to end the job. The task went to "rejected", and the submitter
had to notice and resubmit by hand -- even though the network might have twenty
other GPUs happy to run it.

The rule that needs care is which jobs may move. If somebody picked a machine
deliberately, sending their work elsewhere overrides a choice they made; only
jobs the coordinator placed are re-offered.

These drive the real coordinator function with the database faked out, because
the interesting cases need several nodes and this machine has one GPU.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import backend.coordinator as coordinator  # noqa: E402


def node(node_id, tflops=10.0, connected=True, available=True):
    return {
        "_id": node_id,
        "total_gpu_tflops": tflops,
        "isConnected": connected,
        "isAvailable": available,
    }


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows

    def __aiter__(self):
        async def gen():
            for row in self.rows:
                yield dict(row)
        return gen()


class FakeNodes:
    def __init__(self, rows):
        self.rows = rows

    def find(self, query=None, projection=None):
        query = query or {}
        rows = self.rows
        if query.get("isConnected"):
            rows = [r for r in rows if r.get("isConnected")]
        return FakeCursor(rows)


class FakeTasks:
    def __init__(self):
        self.updates = []

    def aggregate(self, pipeline):
        return FakeCursor([])          # nothing queued anywhere

    async def update_one(self, where, change):
        self.updates.append((where, change))
        return None


class FakeDb:
    def __init__(self, nodes):
        self.nodes_collection = FakeNodes(nodes)
        self.tasks_collection = FakeTasks()


def redispatch(task, nodes, declined_by):
    db = FakeDb(nodes)
    moved = asyncio.get_event_loop().run_until_complete(
        coordinator._redispatch(db, task, declined_by)
    )
    return moved, db.tasks_collection.updates


def auto_task(**extra):
    task = {"_id": "task_1", "placement": "auto", "declined_by": [],
            "node_id": "node_a"}
    task.update(extra)
    return task


# --- which jobs may move ---------------------------------------------------

def test_a_job_the_coordinator_placed_is_offered_elsewhere():
    moved, _ = redispatch(auto_task(), [node("node_a"), node("node_b")], "node_a")
    assert moved == "node_b", moved


def test_a_job_sent_to_a_named_node_is_not_moved():
    """The submitter picked that machine; they may have had a reason."""
    task = auto_task(placement="chosen")
    moved, updates = redispatch(task, [node("node_a"), node("node_b")], "node_a")
    assert moved is None
    assert updates == [], "a chosen job must not be touched"


def test_a_task_with_no_placement_recorded_is_treated_as_chosen():
    # Tasks queued before placement was recorded must not start moving.
    task = auto_task()
    del task["placement"]
    moved, _ = redispatch(task, [node("node_a"), node("node_b")], "node_a")
    assert moved is None


# --- who it may move to ----------------------------------------------------

def test_it_is_never_offered_back_to_the_node_that_declined():
    moved, _ = redispatch(auto_task(), [node("node_a")], "node_a")
    assert moved is None, "offered the job back to the node that refused it"


def test_a_node_that_declined_earlier_is_not_asked_again():
    """Otherwise a job could bounce between two unwilling nodes for ever."""
    task = auto_task(declined_by=["node_b"])
    moved, _ = redispatch(task, [node("node_a"), node("node_b"), node("node_c")], "node_a")
    assert moved == "node_c", moved


def test_it_gives_up_once_every_node_has_refused():
    task = auto_task(declined_by=["node_b", "node_c"])
    moved, _ = redispatch(task, [node("node_a"), node("node_b"), node("node_c")], "node_a")
    assert moved is None


def test_offline_and_unavailable_nodes_are_not_offered_the_job():
    nodes = [node("node_a"), node("gone", connected=False),
             node("off", available=False)]
    moved, _ = redispatch(auto_task(), nodes, "node_a")
    assert moved is None


def test_the_replacement_is_chosen_by_the_usual_ranking():
    # Same policy as first placement: idle beats fast-but-busy, then compute.
    nodes = [node("node_a"), node("slow", 5.0), node("fast", 40.0)]
    moved, _ = redispatch(auto_task(), nodes, "node_a")
    assert moved == "fast", moved


# --- what it writes --------------------------------------------------------

def test_the_job_goes_back_to_pending_on_the_new_node():
    _, updates = redispatch(auto_task(), [node("node_a"), node("node_b")], "node_a")
    assert len(updates) == 1
    _where, change = updates[0]

    assert change["$set"]["status"] == "pending"
    assert change["$set"]["node_id"] == "node_b"


def test_the_refusal_is_recorded_so_it_is_not_repeated():
    _, updates = redispatch(auto_task(declined_by=["node_x"]),
                            [node("node_a"), node("node_b")], "node_a")
    _where, change = updates[0]
    assert set(change["$set"]["declined_by"]) == {"node_a", "node_x"}


def test_stale_run_state_is_cleared_when_the_job_moves():
    """A leftover started_at would make the new node's claim look stale and
    the requeue sweeper would fight with the dispatcher over it."""
    _, updates = redispatch(auto_task(), [node("node_a"), node("node_b")], "node_a")
    _where, change = updates[0]
    assert "started_at" in change["$unset"]
    assert "finished_at" in change["$unset"]


# --- standalone runner ---------------------------------------------------

def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as e:
            failed.append(name)
            print("  FAIL  %s: %s" % (name, e))
        except Exception as e:
            failed.append(name)
            print("  ERROR %s: %s: %s" % (name, type(e).__name__, e))
    print("")
    summary = "%d/%d passed" % (len(tests) - len(failed), len(tests))
    if failed:
        summary += " -- FAILED: %s" % ", ".join(failed)
    print(summary)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())

"""Who is allowed to read a stored artifact.

These exist because the rules were once absent entirely: /artifacts served any
blob to anyone, and the task listings published every dataset, holdout and
weights id. Together that meant a submitter's private training data was
readable by anyone who could reach the coordinator, and a node could fetch the
exact holdout its work was about to be scored against -- which makes
verification worthless while still reporting "accepted".

The rules under test:

  * a holdout is never served over HTTP, to anyone, whatever token they hold
  * every other blob goes only to a node that owns a task referencing it
  * no listing endpoint ever emits a holdout id

No database and no HTTP: the listing scrub is pure, and the download rules are
exercised against a small fake collection.
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))


# --- the listing scrub ----------------------------------------------------

def _public_task():
    from backend.coordinator import public_task
    return public_task


def test_holdout_id_never_appears_in_a_listing():
    public_task = _public_task()
    out = public_task({
        "task_id": "task_1",
        "dataset_id": "aaa",
        "holdout_artifact_id": "SECRET",
        "weights_id": "ccc",
    })
    assert "holdout_artifact_id" not in out
    assert "SECRET" not in repr(out), out


def test_the_fact_of_a_holdout_survives_for_the_dashboard():
    # The UI wants to say "this job is verifiable" without learning the id.
    public_task = _public_task()
    assert public_task({"holdout_artifact_id": "x"})["has_holdout"] is True
    assert public_task({"task_id": "t"})["has_holdout"] is False


def test_everything_else_is_left_alone():
    public_task = _public_task()
    task = {"task_id": "t", "status": "completed", "dataset_id": "d",
            "weights_id": "w", "metrics": {"final_loss": 0.5}}
    out = public_task(dict(task, holdout_artifact_id="h"))
    for key, value in task.items():
        assert out[key] == value, (key, out.get(key), value)


def test_the_scrub_does_not_mutate_the_document_it_was_given():
    # It is called in a loop over live cursor documents.
    public_task = _public_task()
    task = {"task_id": "t", "holdout_artifact_id": "h"}
    public_task(task)
    assert task["holdout_artifact_id"] == "h"


# --- the download rules ---------------------------------------------------

class FakeStream:
    def __init__(self, metadata, payload=b"data"):
        self.metadata = metadata
        self._payload = payload

    async def read(self):
        return self._payload


class FakeTasks:
    """Just enough of the tasks collection for the ownership check."""

    def __init__(self, tasks):
        self.tasks = tasks

    async def find_one(self, query, projection=None):
        for task in self.tasks:
            if "submitter_id" in query:
                if (task.get("submitter_id") == query["submitter_id"]
                        and task.get("weights_id") == query.get("weights_id")):
                    return {"_id": task["_id"]}
                continue

            if task.get("node_id") != query.get("node_id"):
                continue
            wanted = {c[k] for c in query.get("$or", []) for k in c}
            if task.get("dataset_id") in wanted or task.get("weights_id") in wanted:
                return {"_id": task["_id"]}
        return None


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def download(artifact_id, caller, metadata, tasks, submitter=None):
    """Call the real endpoint with the storage layer faked out."""
    from fastapi import HTTPException
    import backend.coordinator as coordinator
    from backend.routes import artifacts as artifacts_routes

    class FakeBucket:
        def __init__(self, *a, **k):
            pass

        async def open_download_stream(self, object_id):
            if metadata is None:
                raise FileNotFoundError("no such artifact")
            return FakeStream(metadata)

    class FakeDb:
        db = None
        tasks_collection = FakeTasks(tasks)

    # Patched on the module that defines download_artifact, not on the one
    # that re-exports it: a function looks its globals up where it was
    # written, so patching the re-export would change nothing.
    original_bucket = artifacts_routes.AsyncIOMotorGridFSBucket
    original_oid = artifacts_routes.ObjectId
    artifacts_routes.AsyncIOMotorGridFSBucket = FakeBucket
    artifacts_routes.ObjectId = lambda v: v     # ids here are plain strings
    try:
        return run(coordinator.download_artifact(
            artifact_id, FakeDb, caller, submitter)), None
    except HTTPException as e:
        return None, e
    finally:
        artifacts_routes.AsyncIOMotorGridFSBucket = original_bucket
        artifacts_routes.ObjectId = original_oid


OWN_TASK = [{"_id": "t1", "node_id": "node_a", "dataset_id": "train_1",
             "weights_id": "w_1", "submitter_id": "sub_a"}]


def test_a_node_cannot_download_the_holdout_it_is_scored_against():
    """The whole point. A node holding a perfectly valid token still cannot
    read the rows its work is graded on."""
    response, error = download("hold_1", "node_a", {"kind": "holdout"}, OWN_TASK)
    assert response is None
    assert error.status_code == 403, error.status_code


def test_a_holdout_is_refused_even_to_a_node_whose_task_references_it():
    tasks = [dict(OWN_TASK[0], dataset_id="hold_1")]
    response, error = download("hold_1", "node_a", {"kind": "holdout"}, tasks)
    assert response is None and error.status_code == 403


def test_a_node_can_download_the_training_data_for_its_own_task():
    response, error = download("train_1", "node_a", {"kind": "dataset"}, OWN_TASK)
    assert error is None, error
    assert response.body == b"data"


def test_a_node_can_download_weights_from_its_own_task():
    response, error = download("w_1", "node_a", {"kind": "weights"}, OWN_TASK)
    assert error is None, error


def test_a_node_cannot_download_another_nodes_dataset():
    response, error = download("train_1", "node_b", {"kind": "dataset"}, OWN_TASK)
    assert response is None
    assert error.status_code == 404, error.status_code


def test_an_artifact_no_task_references_is_not_downloadable():
    # e.g. the pre-split upload, which no task points at.
    response, error = download("orphan", "node_a", {"kind": "dataset"}, OWN_TASK)
    assert response is None and error.status_code == 404


def test_a_missing_artifact_and_a_forbidden_one_look_the_same():
    """Whether an id exists is itself worth not leaking."""
    _, missing = download("nope", "node_a", None, OWN_TASK)
    _, forbidden = download("train_1", "node_b", {"kind": "dataset"}, OWN_TASK)
    assert missing.status_code == forbidden.status_code == 404


def test_an_artifact_without_metadata_is_still_ownership_checked():
    # Older blobs may predate the kind metadata; they must not fall open.
    response, error = download("train_1", "node_b", {}, OWN_TASK)
    assert response is None and error.status_code == 404


# --- the submitter collecting their model ---------------------------------

def test_the_submitter_can_download_the_model_they_paid_for():
    """The point of the whole pipeline: the data owner gets the result."""
    response, error = download("w_1", None, {"kind": "weights"}, OWN_TASK, submitter="sub_a")
    assert error is None, error
    assert response.body == b"data"


def test_another_submitter_cannot_download_that_model():
    response, error = download("w_1", None, {"kind": "weights"}, OWN_TASK, submitter="sub_b")
    assert response is None and error.status_code == 404


def test_a_submitter_cannot_read_the_dataset_by_id():
    # They already hold their own data; serving it back only widens exposure.
    response, error = download("train_1", None, {"kind": "dataset"}, OWN_TASK, submitter="sub_a")
    assert response is None and error.status_code == 404


def test_a_submitter_cannot_read_the_holdout_either():
    response, error = download("hold_1", None, {"kind": "holdout"}, OWN_TASK, submitter="sub_a")
    assert response is None and error.status_code == 403


def test_no_credential_at_all_is_rejected_before_anything_is_read():
    response, error = download("w_1", None, {"kind": "weights"}, OWN_TASK, submitter=None)
    assert response is None and error.status_code == 401


# --- the endpoints that must demand a token -------------------------------

def test_download_and_legacy_result_sink_require_a_node_token():
    import inspect
    from backend.coordinator import download_artifact, receive_task_result

    params = inspect.signature(receive_task_result).parameters
    assert "authenticated_node" in str(params["caller"].default)

    # The download accepts either party, so it checks for both in its body
    # rather than through a raising dependency.
    params = inspect.signature(download_artifact).parameters
    assert "caller" in params and "submitter" in params


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

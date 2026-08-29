"""Stopping a job, and running it again.

A submitted job used to be unstoppable. If you sent 100,000 steps by mistake,
it ran to the end on someone else's graphics card and there was nothing either
of you could do about it. Nor was there any way to repeat a job that failed --
you rebuilt it by hand and hoped you matched the original.

The awkward part is that a running job lives inside the node, not here. The
coordinator cannot kill it, so cancelling records a request and the node stops
at its next step and reports back. These check the pieces of that handshake
that can be tested without a database or a GPU.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from backend.service.taskExecutor import (  # noqa: E402
    HANDLERS,
    JobCancelled,
    execute_task,
)


def run(task_data, handler=None):
    """Execute a task with a stub handler, returning (outcome, logs)."""
    logs = []
    if handler is not None:
        HANDLERS["stub"] = handler
        task_data = dict(task_data, task_type="stub")
    try:
        return execute_task(task_data, logs.append), logs
    finally:
        HANDLERS.pop("stub", None)


# --- a cancellation is not a failure --------------------------------------

def test_a_cancelled_job_is_reported_as_cancelled_not_failed():
    """The distinction matters: a failure suggests the node or the job was
    faulty, and a contributor should not wear that for someone else's change
    of mind."""
    def stub(task_data, log, dataset=None, on_progress=None):
        raise JobCancelled("Cancelled by the submitter.")

    outcome, _ = run({}, stub)
    assert outcome["status"] == "cancelled", outcome
    assert "failed" not in outcome["status"]


def test_the_reason_reaches_the_submitter():
    def stub(task_data, log, dataset=None, on_progress=None):
        raise JobCancelled("Cancelled by the submitter.")

    outcome, _ = run({}, stub)
    assert "submitter" in outcome["result"].lower(), outcome["result"]


def test_a_cancelled_job_is_marked_in_its_metrics():
    def stub(task_data, log, dataset=None, on_progress=None):
        raise JobCancelled("stopped")

    outcome, _ = run({}, stub)
    assert outcome["metrics"].get("cancelled") is True


def test_a_cancellation_with_no_message_still_explains_itself():
    def stub(task_data, log, dataset=None, on_progress=None):
        raise JobCancelled()

    outcome, _ = run({}, stub)
    assert outcome["result"], "an empty reason would tell the submitter nothing"


def test_a_real_failure_is_still_a_failure():
    # The new branch must not swallow genuine errors.
    def stub(task_data, log, dataset=None, on_progress=None):
        raise RuntimeError("the GPU fell over")

    outcome, _ = run({}, stub)
    assert outcome["status"] == "failed"
    assert "GPU fell over" in outcome["result"]


def test_cancelling_does_not_return_half_trained_weights():
    """A partial model would be indistinguishable from a finished one in the
    workspace, and could be downloaded as though it were complete."""
    def stub(task_data, log, dataset=None, on_progress=None):
        raise JobCancelled("stopped")

    outcome, _ = run({}, stub)
    assert "state_dict" not in outcome or not outcome.get("state_dict")


# --- the hook the node cancels through ------------------------------------

def test_a_raising_progress_callback_stops_the_run():
    """The node cancels by raising from on_progress, which is called once per
    step. If the trainer swallowed exceptions from it, cancellation would
    silently do nothing."""
    steps_run = []

    def stub(task_data, log, dataset=None, on_progress=None):
        for step in range(1, 100):
            steps_run.append(step)
            if on_progress:
                on_progress({"step": step, "steps": 100})
        return {"status": "completed", "result": "finished", "metrics": {}}

    def cancel_at_three(update):
        if update["step"] >= 3:
            raise JobCancelled("Cancelled by the submitter.")

    logs = []
    HANDLERS["stub"] = stub
    try:
        outcome = execute_task({"task_type": "stub"}, logs.append,
                               None, cancel_at_three)
    finally:
        HANDLERS.pop("stub", None)

    assert outcome["status"] == "cancelled"
    assert len(steps_run) == 3, f"ran {len(steps_run)} steps; should stop at 3"


def test_a_job_that_finishes_before_the_cancel_lands_is_not_cancelled():
    # The watcher polls on an interval, so a short job can finish first. That
    # must report normally rather than being retroactively cancelled.
    def stub(task_data, log, dataset=None, on_progress=None):
        if on_progress:
            on_progress({"step": 1, "steps": 1})
        return {"status": "completed", "result": "done", "metrics": {}}

    outcome, _ = run({}, stub)
    assert outcome["status"] == "completed"


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

"""Tests for the task execution seam.

No torch, no GPU, no database. The executor is deliberately importable on its
own so this can run anywhere.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from backend.service import taskExecutor  # noqa: E402
from backend.service.taskExecutor import describe_pool, execute_task  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


def requires_torch(fn):
    """Training runs for real, so these need torch present."""
    fn.requires_torch = True
    return fn


def run(task_data):
    """Execute a task, returning (outcome, logs)."""
    logs = []
    return execute_task(task_data, logs.append), logs


FAKE_POOL = [
    {"index": 0, "name": "RTX 3070", "measured_tflops": 38.2,
     "total_memory_mb": 8192, "free_memory_mb": 7800},
    {"index": 1, "name": "RTX 3050", "measured_tflops": 16.4,
     "total_memory_mb": 8192, "free_memory_mb": 7900},
    {"index": 2, "name": "GTX 1660", "measured_tflops": 9.1,
     "total_memory_mb": 6144, "free_memory_mb": 5900},
]


class fake_gpus:
    """Swap in a fixed pool for the duration of a block."""

    def __init__(self, devices):
        self.devices = devices

    def __enter__(self):
        self.original = taskExecutor.benchmark_all
        taskExecutor.benchmark_all = lambda *a, **k: self.devices

    def __exit__(self, *exc):
        taskExecutor.benchmark_all = self.original


# --- dispatch ------------------------------------------------------------

@requires_torch
def test_known_task_type_completes():
    outcome, logs = run({"task_type": "llm_training", "model_name": "demo"})
    assert outcome["status"] == "completed"
    assert "demo" in outcome["result"]
    assert logs


def test_unsupported_task_type_fails_without_raising():
    outcome, _ = run({"task_type": "mine_bitcoin"})
    assert outcome["status"] == "failed"
    assert "mine_bitcoin" in outcome["result"]


def test_missing_and_empty_task_data_fail_cleanly():
    for payload in [{}, None, {"task_type": None}]:
        outcome, _ = run(payload)
        assert outcome["status"] == "failed"


def test_handler_exception_is_reported_not_raised():
    def boom(task_data, log, dataset=None):
        raise RuntimeError("GPU fell over")

    taskExecutor.HANDLERS["explode"] = boom
    try:
        outcome, logs = run({"task_type": "explode"})
        assert outcome["status"] == "failed"
        assert "GPU fell over" in outcome["result"]
        assert any("GPU fell over" in line for line in logs)
    finally:
        del taskExecutor.HANDLERS["explode"]


def test_outcome_always_has_the_keys_the_node_reports():
    for payload in [{"task_type": "llm_training", "model_name": "m"}, {"task_type": "nope"}]:
        outcome, _ = run(payload)
        assert set(outcome) >= {"status", "result", "metrics"}


# --- batch size handling -------------------------------------------------

@requires_torch
def test_batch_size_is_taken_from_hyperparameters():
    outcome, _ = run({"task_type": "llm_training", "model_name": "m",
                      "hyperparameters": {"batch_size": 256}})
    assert outcome["metrics"]["batch_size"] == 256


@requires_torch
def test_invalid_batch_sizes_fall_back_to_the_default():
    for bad in ["abc", None, 0, -5, {}]:
        outcome, _ = run({"task_type": "llm_training", "model_name": "m",
                          "hyperparameters": {"batch_size": bad}})
        assert outcome["metrics"]["batch_size"] == taskExecutor.DEFAULT_BATCH_SIZE, bad


@requires_torch
def test_missing_hyperparameters_uses_the_default():
    outcome, _ = run({"task_type": "llm_training", "model_name": "m"})
    assert outcome["metrics"]["batch_size"] == taskExecutor.DEFAULT_BATCH_SIZE


# --- pool reporting ------------------------------------------------------

def test_pool_is_described_in_the_task_log():
    logs = []
    with fake_gpus(FAKE_POOL):
        summary, plan = describe_pool(256, logs.append)

    joined = "\n".join(logs)
    assert "63.7 TFLOPS across 3 GPU(s)" in joined, joined
    assert "cuda:0" in joined and "cuda:1" in joined and "cuda:2" in joined
    assert summary["device_count"] == 3
    # The plan handed to the trainer is the proportional split, not an even one.
    assert [p["batch_size"] for p in plan] == [153, 66, 37], plan


def test_log_calls_out_the_gain_over_an_even_split():
    logs = []
    with fake_gpus(FAKE_POOL):
        describe_pool(256, logs.append)
    joined = "\n".join(logs)
    assert "2.333x an even split" in joined, joined
    assert "GTX 1660" in joined  # named as the bottleneck


def test_identical_gpus_do_not_advertise_a_pointless_speedup():
    same = [dict(d, name="RTX 3060 #%d" % i, measured_tflops=12.0)
            for i, d in enumerate(FAKE_POOL)]
    logs = []
    with fake_gpus(same):
        describe_pool(96, logs.append)
    assert not any("even split" in line for line in logs)


@requires_torch
def test_machine_without_gpus_says_so_and_still_completes():
    logs = []
    with fake_gpus([]):
        summary, plan = describe_pool(64, logs.append)
        outcome, task_logs = run({"task_type": "llm_training", "model_name": "m",
                                  "hyperparameters": {"steps": 1, "batch_size": 8}})

    assert summary["device_count"] == 0
    assert plan == []
    assert any("No benchmarked GPU" in line for line in logs)
    # Falls back to CPU rather than failing, so a GPU-less node still works.
    assert outcome["status"] == "completed"


@requires_torch
def test_metrics_carry_the_measured_pool_total():
    with fake_gpus(FAKE_POOL):
        outcome, _ = run({"task_type": "llm_training", "model_name": "m"})
    assert abs(outcome["metrics"]["pooled_tflops"] - 63.7) < 0.01
    assert outcome["metrics"]["device_count"] == 3


@requires_torch
def test_training_reports_real_measured_work():
    outcome, _ = run({"task_type": "llm_training", "model_name": "m",
                      "hyperparameters": {"steps": 2, "batch_size": 8}})
    metrics = outcome["metrics"]

    # Nothing is simulated any more: these come from an actual training run.
    assert "simulated" not in metrics
    assert metrics["parameters"] > 0
    assert metrics["samples_per_second"] > 0
    assert metrics["initial_loss"] is not None
    assert metrics["final_loss"] is not None
    assert metrics["steps"] == 2


# --- standalone runner ---------------------------------------------------

def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failed = []
    skipped = 0
    for name, fn in tests:
        if getattr(fn, "requires_torch", False) and not HAVE_TORCH:
            skipped += 1
            print("  SKIP  %s (needs torch)" % name)
            continue
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
    summary = "%d/%d passed" % (len(tests) - len(failed) - skipped, len(tests) - skipped)
    if skipped:
        summary += " (%d skipped, no torch)" % skipped
    if failed:
        summary += " -- FAILED: %s" % ", ".join(failed)
    print(summary)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())

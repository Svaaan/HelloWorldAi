"""Run one task on this machine's own hardware.

This module dispatches a task to a handler and reports the outcome. It is
deliberately free of FastAPI, Mongo and HTTP imports so it can be tested without
the rest of the node.

Training runs for real on this machine's GPUs: the batch is split across them in
proportion to measured throughput (see poolPlanner) and executed by trainer.py.
"""

import inspect
import logging
from typing import Any, Callable, Dict, List, Tuple

from backend.service import trainer
from backend.service.thermalPolicy import ThermalAbort
from backend.service.gpuBenchmark import benchmark_all
from backend.service.poolPlanner import plan_batch_split, pool_summary

logger = logging.getLogger(__name__)


class JobCancelled(RuntimeError):
    """Raised to stop a running job because the submitter asked it to stop.

    Distinct from a failure: nothing went wrong, so it must not be reported as
    a fault of the node or of the job.
    """

DEFAULT_BATCH_SIZE = 64


def addressable_devices(devices: List[Dict[str, Any]],
                        log: Callable[[str], None]) -> List[Dict[str, Any]]:
    """Keep only the GPUs torch can actually address.

    The pool is enumerated with NVML, which can list cards torch will not
    accept. CUDA_VISIBLE_DEVICES narrows torch's view while NVML still sees
    every card, and a GPU can be taken away between one job and the next.

    Handing torch an index it does not have kills the job with a bare
    "CUDA error: invalid device ordinal" partway through, after the
    contributor's card has already done the work. Drop those devices up front,
    so the pool we advertise is the pool we can actually run on.
    """
    try:
        import torch
    except ImportError:
        return devices

    if not devices or not torch.cuda.is_available():
        return devices

    visible = torch.cuda.device_count()
    usable = [d for d in devices if d.get("index", 0) < visible]

    missing = len(devices) - len(usable)
    if missing:
        log(f"{missing} of {len(devices)} GPU(s) are not visible to torch "
            f"(it sees {visible}); planning around the rest.")
    return usable


def describe_pool(batch_size: int,
                  log: Callable[[str], None]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Benchmark local GPUs and log how a batch would be split across them.

    Returns (summary, plan). The plan is what the trainer shards the batch by.
    """
    devices = addressable_devices(benchmark_all(), log)
    summary = pool_summary(devices)
    plan = plan_batch_split(devices, batch_size)

    if not plan:
        log("No benchmarked GPU available on this machine.")
        return summary, []

    log(f"Pool: {summary['pooled_tflops']} TFLOPS across {summary['device_count']} GPU(s)")

    if summary["speedup_vs_even_split"] > 1.05:
        log(f"Proportional split is {summary['speedup_vs_even_split']}x an even split "
            f"(slowest device: {summary['bottleneck']})")

    for part in plan:
        log(f"  cuda:{part['device_index']} {part['name']} <- "
            f"{part['batch_size']} of {batch_size} samples")

    return summary, plan


def _run_llm_training(task_data: Dict[str, Any], log: Callable[[str], None],
                      dataset=None, on_progress=None) -> Dict[str, Any]:
    """Train for real on this machine's GPUs, sharded by measured throughput."""
    model_name = task_data.get("model_name")
    hyperparameters = task_data.get("hyperparameters", {}) or {}

    try:
        batch_size = int(hyperparameters.get("batch_size", DEFAULT_BATCH_SIZE))
    except (TypeError, ValueError):
        batch_size = DEFAULT_BATCH_SIZE
    if batch_size <= 0:
        batch_size = DEFAULT_BATCH_SIZE

    summary, plan = describe_pool(batch_size, log)

    outcome = trainer.train(task_data, log, plan,
                            batch_size=batch_size, dataset=dataset,
                            on_progress=on_progress)
    metrics = outcome["metrics"]
    metrics["pooled_tflops"] = summary["pooled_tflops"]
    metrics["device_count"] = max(summary["device_count"], len(metrics.get("devices", [])))

    data_note = (
        f" on {metrics['dataset_rows']:,} rows" if metrics.get("dataset_rows")
        else " on synthetic data"
    )

    return {
        "status": "completed",
        "result": (
            f"Trained {model_name or 'model'} "
            f"({metrics['parameters']:,} parameters) for {metrics['steps']} steps"
            f"{data_note}. "
            f"Loss {metrics['initial_loss']} -> {metrics['final_loss']}, "
            f"{metrics['achieved_tflops']} TFLOPS achieved."
        ),
        "metrics": metrics,
        "state_dict": outcome["state_dict"],
        # Travels with the weights so the submitter can rebuild the model.
        "manifest": outcome.get("manifest"),
    }


HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "llm_training": _run_llm_training,
}


def _accepts_progress(handler: Callable) -> bool:
    """Whether `handler` takes the on_progress argument."""
    try:
        params = inspect.signature(handler).parameters
    except (TypeError, ValueError):
        return True  # can't introspect it; assume the current signature
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
        return True
    return "on_progress" in params or len(params) >= 4


def execute_task(task_data: Dict[str, Any], log: Callable[[str], None],
                 dataset=None, on_progress=None) -> Dict[str, Any]:
    """Dispatch a task and return {status, result, metrics[, state_dict]}.

    `dataset` is the (x, y) pair the node downloaded for this task, or None.

    Never raises: a failing task reports `failed` so the node can hand the
    outcome back rather than looking like it went silent.
    """
    task_type = (task_data or {}).get("task_type")
    log(f"Handling task type: {task_type}")

    handler = HANDLERS.get(task_type)
    if handler is None:
        log(f"Unsupported task type: {task_type}")
        return {
            "status": "failed",
            "result": f"Unsupported task type: {task_type}",
            "metrics": {},
        }

    try:
        # on_progress is optional: HANDLERS is an extension point, and a
        # handler written against the older three-argument signature should
        # keep working rather than dying with a confusing TypeError.
        if _accepts_progress(handler):
            return handler(task_data, log, dataset, on_progress)
        return handler(task_data, log, dataset)
    except JobCancelled as e:
        logger.info(f"Task cancelled while running: {e}")
        return {
            "status": "cancelled",
            "result": str(e) or "Cancelled by the submitter.",
            "metrics": {"cancelled": True},
        }
    except ThermalAbort as e:
        # Not a fault in the job — the contributor's hardware got too hot.
        logger.warning(f"Task stopped on temperature: {e}")
        return {
            "status": "failed",
            "result": f"Stopped to protect the GPU: {e}",
            "metrics": {"thermal_abort": True},
        }
    except Exception as e:
        logger.exception("Task handler raised")
        log(f"Error processing task: {e}")
        return {"status": "failed", "result": f"Error: {e}", "metrics": {}}

"""Run one task on this machine's own hardware.

This module dispatches a task to a handler and reports the outcome. It is
deliberately free of FastAPI, Mongo and HTTP imports so it can be tested without
the rest of the node.

Training runs for real on this machine's GPUs: the batch is split across them in
proportion to measured throughput (see poolPlanner) and executed by trainer.py.
"""

import logging
from typing import Any, Callable, Dict, List, Tuple

from backend.service import trainer
from backend.service.thermalPolicy import ThermalAbort
from backend.service.gpuBenchmark import benchmark_all
from backend.service.poolPlanner import plan_batch_split, pool_summary

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64


def describe_pool(batch_size: int,
                  log: Callable[[str], None]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Benchmark local GPUs and log how a batch would be split across them.

    Returns (summary, plan). The plan is what the trainer shards the batch by.
    """
    devices = benchmark_all()
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
                      dataset=None) -> Dict[str, Any]:
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
                            batch_size=batch_size, dataset=dataset)
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
    }


HANDLERS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "llm_training": _run_llm_training,
}


def execute_task(task_data: Dict[str, Any], log: Callable[[str], None],
                 dataset=None) -> Dict[str, Any]:
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
        return handler(task_data, log, dataset)
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

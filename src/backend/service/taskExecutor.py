"""Run one task on this machine's own hardware.

This module is the seam where real compute lands. Everything around it -- the
node claiming work from the coordinator, logging, reporting results, retrying
abandoned tasks -- is finished. Only `_run_llm_training` is still simulated.

It is deliberately free of FastAPI, Mongo and HTTP imports so it can be tested
without the rest of the node, and it already benchmarks and plans across
whatever GPUs the machine has, so the pool a job would run on shows up in the
task log today.
"""

import logging
import time
from typing import Any, Callable, Dict

from backend.service.gpuBenchmark import benchmark_all
from backend.service.poolPlanner import plan_batch_split, pool_summary

logger = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 64


def describe_pool(batch_size: int, log: Callable[[str], None]) -> Dict[str, Any]:
    """Benchmark local GPUs and log how a batch would be split across them."""
    devices = benchmark_all()
    summary = pool_summary(devices)
    plan = plan_batch_split(devices, batch_size)

    if not plan:
        log("No benchmarked GPU available on this machine.")
        return summary

    log(f"Pool: {summary['pooled_tflops']} TFLOPS across {summary['device_count']} GPU(s)")

    if summary["speedup_vs_even_split"] > 1.05:
        log(f"Proportional split is {summary['speedup_vs_even_split']}x an even split "
            f"(slowest device: {summary['bottleneck']})")

    for part in plan:
        log(f"  cuda:{part['device_index']} {part['name']} <- "
            f"{part['batch_size']} of {batch_size} samples")

    return summary


def _run_llm_training(task_data: Dict[str, Any], log: Callable[[str], None]) -> Dict[str, Any]:
    """SIMULATED. Replace this body with a real training loop.

    The replacement consumes the same plan_batch_split() output that
    describe_pool() logs, giving each local GPU a share of the batch sized to
    its measured throughput.
    """
    model_name = task_data.get("model_name")
    hyperparameters = task_data.get("hyperparameters", {}) or {}

    try:
        batch_size = int(hyperparameters.get("batch_size", DEFAULT_BATCH_SIZE))
    except (TypeError, ValueError):
        batch_size = DEFAULT_BATCH_SIZE
    if batch_size <= 0:
        batch_size = DEFAULT_BATCH_SIZE

    summary = describe_pool(batch_size, log)

    log(f"Training {model_name} with hyperparameters {hyperparameters}")
    for i in range(1, 4):
        time.sleep(1)
        log(f"Processing batch {i}/3")
    log(f"Training {model_name} completed!")

    return {
        "status": "completed",
        "result": f"Training of {model_name} completed successfully.",
        "metrics": {
            "pooled_tflops": summary["pooled_tflops"],
            "device_count": summary["device_count"],
            "batch_size": batch_size,
            "simulated": True,
        },
    }


HANDLERS: Dict[str, Callable[[Dict[str, Any], Callable[[str], None]], Dict[str, Any]]] = {
    "llm_training": _run_llm_training,
}


def execute_task(task_data: Dict[str, Any], log: Callable[[str], None]) -> Dict[str, Any]:
    """Dispatch a task to its handler and return {status, result, metrics}.

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
        return handler(task_data, log)
    except Exception as e:
        logger.exception("Task handler raised")
        log(f"Error processing task: {e}")
        return {"status": "failed", "result": f"Error: {e}", "metrics": {}}

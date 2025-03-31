import uuid
import psutil
import GPUtil
import logging
from typing import Dict, Any
from fastapi import Request

logger = logging.getLogger(__name__)


def is_node_overloaded(cpu_threshold=90.0, gpu_threshold=90.0, memory_threshold=90.0) -> bool:
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        gpus = GPUtil.getGPUs()
        gpu = max((gpu.load * 100 for gpu in gpus), default=0.0)

        return cpu > cpu_threshold or memory > memory_threshold or gpu > gpu_threshold
    except Exception as e:
        logger.warning(f"⚠️ Failed to check node load: {e}")
        return False  # Fail-safe: allow task if we can't check


def process_task(task: Dict[str, Any], node_info: Dict[str, Any], request: Request) -> Dict[str, Any]:
    try:
        client_ip = request.client.host
        client_id = task.get("client_id")
        task_type = task.get("type")

        # Optional owner-defined policies
        accept_tasks = node_info.get("accept_tasks", True)
        allowed_clients = node_info.get("allowed_clients", [])
        accepted_task_types = node_info.get("accepted_task_types", [])
        max_tasks = node_info.get("max_tasks", 100)

        if not accept_tasks:
            return {"error": "This node is not currently accepting tasks."}

        if allowed_clients and client_id not in allowed_clients:
            return {"error": f"Client '{client_id}' is not allowed to run tasks on this node."}

        if accepted_task_types and task_type not in accepted_task_types:
            return {"error": f"Task type '{task_type}' is not accepted by this node."}

        if node_info["total_tasks_processed"] >= max_tasks:
            return {"error": "Node has reached its maximum task limit."}

        if is_node_overloaded():
            return {"error": "Node is currently overloaded and cannot accept new tasks."}

        logger.info(f"🧠 Task received from {client_id or client_ip} — Type: {task_type}")

        gpu_list = node_info["capabilities"].get("gpu", [])
        gpu_available = next((gpu for gpu in gpu_list if gpu.get("name") != "No GPU Detected"), None)

        # Increment processed task count
        node_info["total_tasks_processed"] += 1

        # Prefer GPU if available
        if gpu_available:
            result = _process_gpu_task(task)
            return {
                "task_id": task.get("task_id", str(uuid.uuid4())),
                "result": result,
                "gpu_used": gpu_available["name"],
                "gpu_load": gpu_available.get("load_percentage", 0),
                "processing_method": "GPU"
            }

        # Fallback to CPU
        result = _process_cpu_task(task)
        return {
            "task_id": task.get("task_id", str(uuid.uuid4())),
            "result": result,
            "processing_method": "CPU"
        }

    except Exception as e:
        logger.error(f"Computation error: {e}")
        return {
            "error": "Computation failed",
            "details": str(e)
        }


def _process_gpu_task(task: Dict[str, Any]) -> str:
    try:
        return str(task).upper()
    except Exception as e:
        logger.error(f"GPU task processing error: {e}")
        return f"GPU_ERROR: {e}"


def _process_cpu_task(task: Dict[str, Any]) -> str:
    try:
        return str(task).lower()
    except Exception as e:
        logger.error(f"CPU task processing error: {e}")
        return f"CPU_ERROR: {e}"

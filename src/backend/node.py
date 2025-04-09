# node.py

import os
import socket
import uuid
import logging
import requests
import time
import pynvml
from typing import Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.service.runningNodeService import process_task
from backend.service.systemInfoService import get_system_capabilities
from backend.service.connectionService import background_connection_handler
from backend.service.usageService import get_usage
from backend.executeTask import handle_task, validate_task_data
from psutil import cpu_percent, virtual_memory

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Compute Node", description="Distributed Computing Node")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Node info state
public_ip = socket.gethostbyname(socket.gethostname())
node_info = {
    "country": "Unknown",
    "capabilities": get_system_capabilities(),
    "connected": False,
    "accept_tasks": True,
    "allowed_clients": ["trusted-client-1", "trusted-client-2"],
    "total_tasks_processed": 0
}

# Task queues
task_queue: List[dict] = []
task_logs: Dict[str, list] = {}
completed_tasks: List[dict] = []

# === Utils ===

def get_country_from_ip(ip: str) -> str:
    try:
        res = requests.get(f"https://ipapi.co/{ip}/country_name/", timeout=5)
        if res.status_code == 200:
            return res.text.strip()
    except Exception as e:
        logger.warning(f"Failed to get country for IP {ip}: {e}")
    return "Unknown"

def get_gpu_info_list():
    try:
        pynvml.nvmlInit()
        gpu_info = []

        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_info.append({
                "name": pynvml.nvmlDeviceGetName(handle),
                "total_memory": round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**2),
                "free_memory": round(pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1024**2),
                "used_memory": round(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2),
                "load_percentage": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "temperature": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            })

        return gpu_info

    except pynvml.NVMLError as e:
        logger.warning(f"⚠️ NVML error: {e}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

# === Startup ===

# === Payload Builder ===

def build_node_payload() -> Dict[str, Any]:
    return {
        "ip": public_ip,
        "country": node_info["country"],
        "capabilities": node_info["capabilities"],
        "isConnected": node_info["connected"],
        "isAvailable": node_info.get("isAvailable", False),
        "cpu_verified": node_info.get("cpu_verified", False),
        "gpu_verified": node_info.get("gpu_verified", False),
        "cpu_benchmark": node_info.get("cpu_benchmark"),
        "gpu_benchmark": node_info.get("gpu_benchmark"),
    }

# === Routes ===

@app.post("/connect-node")
async def connect_node():
    if node_info.get("connected"):
        return {"status": "Node already connected", "connected": True, "node_id": node_info.get("node_id")}

    detected_gpus = get_gpu_info_list()

    if not detected_gpus or detected_gpus[0].get("name") in ["No GPU Detected", None, ""]:
        return {"status": "rejected", "reason": "No valid GPU detected. Node connection refused."}

    node_info["capabilities"]["gpu"] = detected_gpus
    node_info["connected"] = True

    payload = build_node_payload()

    # ✅ Await coordinator connection handler
    from backend.service.connectionService import background_connection_handler
    response = await background_connection_handler(payload, node_info)

    # ✅ Extract node_id from coordinator response and save locally
    if response and response.get("node_id"):
        node_info["node_id"] = response["node_id"]
    else:
        # Optional safety fallback
        node_info["node_id"] = "unknown-node-id"

    return {"status": "Node registered", "connected": True, "node_id": node_info["node_id"]}



@app.get("/usage")
async def get_usage_info():
    try:
        
        usage = await get_usage()

        return usage
    
    except Exception as e:
        logger.error(f"Error getting usage info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get usage info: {str(e)}")

@app.get("/node-capabilities")
def get_detailed_capabilities():
    return {
        "node_id": node_info["node_id"],
        "system_info": node_info["capabilities"],
        "country": node_info["country"],
        "status": {"connected": node_info["connected"], "total_tasks_processed": node_info["total_tasks_processed"]}
    }

@app.post("/queue-task/{node_id}")
async def queue_task(node_id: str, task_data: dict = Body(...), request: Request = None):
    logger.info(f"📥 Received task for node {node_id}: {task_data}")

    is_valid, error_message = validate_task_data(task_data)
    if not is_valid:
        return {"status": "error", "message": f"Invalid task data: {error_message}"}

    task_id = f"task_{len(task_queue) + 1}"

    task_queue.append({
        "task_id": task_id,
        "node_id": node_id,
        "task_data": task_data,
        "status": "pending",
        "origin_ip": request.client.host if request else None
    })

    logger.info(f"📝 Task queued: {task_id} from {request.client.host if request else 'unknown'}")
    return {"status": "success", "task_id": task_id, "message": "Task queued for approval"}

@app.get("/get-pending-tasks")
def get_pending_tasks():
    return [task for task in task_queue if task["status"] == "pending"]

@app.post("/process-task/{task_id}")
def process_task_endpoint(task_id: str, background_tasks: BackgroundTasks):
    task = next((t for t in task_queue if t["task_id"] == task_id), None)
    if not task:
        return {"status": "error", "message": "Task not found"}

    task["status"] = "processing"
    task_queue.remove(task)
    task_logs[task_id] = ["Task started"]

    background_tasks.add_task(task_with_logging, task)

    return {"status": "processing", "message": f"Task {task_id} is now being processed"}

@app.post("/reject-task/{task_id}")
def reject_task(task_id: str):
    task = next((t for t in task_queue if t["task_id"] == task_id), None)
    if not task:
        return {"status": "error", "message": "Task not found"}

    task_queue.remove(task)

    completed_tasks.append({
        "_id": str(uuid.uuid4()),
        "status": "rejected",
        "task_type": task.get("task_data", {}).get("task_type"),
        "completed_at": datetime.now().isoformat()
    })

    logger.info(f"Task {task_id} was rejected and removed from queue.")
    return {"status": "rejected", "message": f"Task {task_id} has been rejected."}


# === Background Task ===

def task_with_logging(task):
    import requests

    task_id = task["task_id"]
    task_data = task["task_data"]
    task_type = task_data.get("task_type")
    origin_ip = task.get("origin_ip")

    task_logs[task_id] = ["Task started"]

    def log(message):
        logger.info(message)
        task_logs[task_id].append(message)

    log(f"Handling task type: {task_type}")

    result_payload = {
        "_id": str(uuid.uuid4()),
        "status": "completed",
        "result": None,
        "original_task_type": task_type
    }

    try:
        if task_type == "llm_training":
            model_name = task_data.get("model_name")
            hyperparameters = task_data.get("hyperparameters", {})
            data = task_data.get("data", {})

            log(f"Training {model_name} with hyperparameters {hyperparameters}")
            log(f"Data: {data}")

            for i in range(1, 4):
                time.sleep(1)
                log(f"Processing batch {i}/3")

            log(f"Training {model_name} completed!")
            result_payload["result"] = f"Training of {model_name} completed successfully."

        else:
            log("Unsupported task type.")
            result_payload["status"] = "failed"
            result_payload["result"] = "Unsupported task type."

    except Exception as e:
        log(f"Error processing task: {str(e)}")
        result_payload["status"] = "failed"
        result_payload["result"] = f"Error: {str(e)}"

    if origin_ip:
        try:
            response = requests.post(
                f"http://{origin_ip}:3000/receive-task-result",
                json={**result_payload, "logs": task_logs.get(task_id, [])},
                timeout=5
            )
            log(f"✅ Result sent to {origin_ip}, Response: {response.status_code}")
        except Exception as e:
            log(f"❌ Failed to send result to {origin_ip}: {e}")
    else:
        log("⚠️ No origin IP found, result not sent back.")

from asyncio import Task, log
import datetime
import os
import socket
import time
import uuid
import logging
import requests
global node_info
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from backend.service.runningNodeService import process_task
from backend.service.systemInfoService import get_system_capabilities
from backend.service.connectionService import background_connection_handler
from psutil import cpu_percent, virtual_memory
from backend.executeTask import handle_task, validate_task_data
from fastapi import Body, Path
from typing import List
import pynvml
from fastapi import Body

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Compute Node", description="Distributed Computing Node")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

task_queue: List[dict] = []
task_logs: Dict[str, list] = {} 

def get_node_ip() -> str:
    try:
        return socket.gethostbyname(socket.gethostname())
    except Exception as e:
        logger.warning(f"IP retrieval error: {e}")
        return "127.0.0.1"

def get_country_from_ip(ip: str) -> str:
    try:
        res = requests.get(f"https://ipapi.co/{ip}/country_name/", timeout=5)
        if res.status_code == 200:
            return res.text.strip()
    except Exception as e:
        logger.warning(f"Failed to get country for IP {ip}: {e}")
    return "Unknown"

# Node configuration
public_ip = get_node_ip()
node_info = {
    "node_id": f"node_{uuid.uuid4()}",
    "country": get_country_from_ip(public_ip),
    "capabilities": get_system_capabilities(),
    "connected": False,
    "accept_tasks": True,
    "allowed_clients": ["trusted-client-1", "trusted-client-2"],
    "total_tasks_processed": 0
}

def connect_to_coordinator():
    gpu_info = get_gpu_info_list()
    if not gpu_info:
        print("No eligible GPU found. Skipping coordinator connection.")
        return  # Abort early

    try:
        coordinator_url = os.getenv("COORDINATOR_URL", "http://79.76.55.71:8100")
        payload = {
            "node_id": node_info["node_id"],
            "ip": public_ip,
            "country": node_info["country"],
            "capabilities": node_info["capabilities"],
            "isConnected": True,
            "isAvailable": True,
            "cpu_verified": node_info.get("cpu_verified", False),
            "gpu_verified": node_info.get("gpu_verified", False),
            "cpu_usage": node_info.get("cpu_usage", 0.0),
            "gpu_usage": node_info.get("gpu_usage", 0.0),
            "cpu_benchmark": node_info.get("cpu_benchmark"),
            "gpu_benchmark": node_info.get("gpu_benchmark")
        }
        response = requests.post(f"{coordinator_url}/connect-node", json=payload, timeout=5)
        print(f"Coordinator response: {response.json()}")
    except Exception as e:
        print(f"Failed to connect to coordinator: {e}")


def get_gpu_info_list():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

            gpu_info.append({
                "name": name,
                "total_memory": round(memory_info.total / 1024 ** 2),
                "free_memory": round(memory_info.free / 1024 ** 2),
                "used_memory": round(memory_info.used / 1024 ** 2),
                "load_percentage": utilization.gpu,
                "temperature": temperature
            })

        return gpu_info

    except pynvml.NVMLError as e:
        print("⚠️ NVML error:", e)
        return []

    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

    
@app.post("/connect-node")
async def connect_node(background_tasks: BackgroundTasks):
    if node_info["connected"]:
        return {
            "status": "Node already connected",
            "connected": True,
            "node_id": node_info["node_id"]
        }

    detected_gpus = get_gpu_info_list()

    # ✅ Check if GPU is present
    if not detected_gpus or detected_gpus[0].get("name") in ["No GPU Detected", None, ""]:
        return {
            "status": "rejected",
            "reason": "No valid GPU detected. Node connection refused."
        }

    node_info["capabilities"]["gpu"] = detected_gpus
    node_info["connected"] = True  

    # Prepare payload
    payload = {
        "node_id": node_info["node_id"],
        "ip": public_ip,
        "country": node_info["country"],
        "capabilities": node_info["capabilities"],
        "isConnected": node_info["connected"],
        "isAvailable": node_info.get("isAvailable", False),
        "total_compute_score": node_info.get("total_compute_score", 0),
        "cpu_verified": node_info.get("cpu_verified", False),
        "gpu_verified": node_info.get("gpu_verified", False),
        "cpu_usage": node_info.get("cpu_usage", 0.0),
        "gpu_usage": node_info.get("gpu_usage", 0.0),
        "cpu_benchmark": node_info.get("cpu_benchmark"),
        "gpu_benchmark": node_info.get("gpu_benchmark")
    }

    background_tasks.add_task(background_connection_handler, payload, node_info)

    return {
        "status": "Connection in progress",
        "connected": True,
        "node_id": node_info["node_id"]
    }

@app.get("/usage")
def get_usage():
    try:
        cpu_usage = cpu_percent(interval=1)
        gpu_usage = 0

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = utilization.gpu
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"GPU usage retrieval failed: {e}")
            gpu_usage = "N/A"

        return {
            "cpu_usage": cpu_usage,
            "gpu_usage": gpu_usage,
            "memory_usage": virtual_memory().percent
        }

    except Exception as e:
        logger.error(f"Usage retrieval error: {e}")
        return {"error": "Failed to retrieve usage information"}


@app.get("/node-capabilities")
def get_detailed_capabilities():
    return {
        "node_id": node_info["node_id"],
        "system_info": node_info["capabilities"],
        "country": node_info["country"],
        "status": {
            "connected": node_info["connected"],
            "total_tasks_processed": node_info["total_tasks_processed"]
        }
    }

@app.post("/queue-task/{node_id}")
async def queue_task(node_id: str, task_data: dict = Body(...), request: Request = None):
    print(f"📥 Received task for node {node_id}: {task_data}")

    # ✅ Validate task data
    is_valid, error_message = validate_task_data(task_data)
    if not is_valid:
        return {"status": "error", "message": f"Invalid task data: {error_message}"}

    task_id = f"task_{len(task_queue) + 1}"  

    task_queue.append({
        "task_id": task_id,
        "node_id": node_id,
        "task_data": task_data,
        "status": "pending",
        "origin_ip": request.client.host  
    })

    print(f"📝 Task queued: {task_id} from {request.client.host}")
    return {"status": "success", "task_id": task_id, "message": "Task queued for approval"}



@app.get("/get-pending-tasks")
def get_pending_tasks():
    return [task for task in task_queue if task["status"] == "pending"]

@app.post("/process-task/{task_id}")
def process_task_endpoint(task_id: str, background_tasks: BackgroundTasks):
    task = next((t for t in task_queue if t["task_id"] == task_id), None)
    if not task:
        return {"status": "error", "message": "Task not found"}

    # Update status before removing from queue
    task["status"] = "processing"
    task_queue.remove(task)

    # Init empty logs
    task_logs[task_id] = ["Task started"]

    # Add the background task to actually process the task
    background_tasks.add_task(task_with_logging, task)

    return {"status": "processing", "message": f"Task {task_id} is now being processed"}


    # Background task with logging
def task_with_logging(task):
    task_id = task["task_id"]
    task_data = task["task_data"]
    task_type = task_data.get("task_type")
    origin_ip = task.get("origin_ip")

    def log(message):
        print(message)
        task_logs[task_id].append(message)  # Add log message to task_logs

    log(f"Handling task type: {task_type}")

    result_payload = {
        "task_id": task_id,
        "status": "completed",
        "logs": task_logs.get(task_id, []),  # Make sure 'logs' are added
        "result": None
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

    # Store completed/failed task in history
    completed_tasks.append({
        **task,
        "status": result_payload["status"],
        "result": result_payload["result"],
        "logs": result_payload["logs"],  # Ensure logs are stored
        "completed_at": datetime.datetime.now().isoformat()
    })

    # Optional but good: Safe result delivery
    if not origin_ip:
        log("⚠️ No origin IP found, result not sent back.")
    else:
        try:
            response = requests.post(f"http://{origin_ip}:3000/receive-task-result", json=result_payload, timeout=5)
            log(f"Result sent to {origin_ip}, Response: {response.status_code}")
        except Exception as e:
            log(f"Failed to send result to {origin_ip}: {e}")




@app.post("/reject-task/{task_id}")
def reject_task(task_id: str):
    task = next((t for t in task_queue if t["task_id"] == task_id), None)
    if not task:
        return {"status": "error", "message": "Task not found"}

    task_queue.remove(task)
    
    completed_tasks.append({
        **task,
        "status": "rejected",
        "completed_at": datetime.datetime.now().isoformat()
    })
    
    print(f"Task {task_id} was rejected and removed from queue.")
    return {"status": "rejected", "message": f"Task {task_id} has been rejected."}

completed_tasks = []
@app.post("/compute")
def compute(task: Dict[str, Any], request: Request):
    return process_task(task, node_info, request)

@app.on_event("startup")
async def startup_event():
    connect_to_coordinator()

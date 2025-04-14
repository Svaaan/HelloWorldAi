
import uuid
import logging
from fastapi.responses import JSONResponse
import requests
import time
import json
from typing import Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.service.systemInfoService import get_system_capabilities  # Dynamically fetch system capabilities
from backend.service.usageService import get_usage
from backend.shared.nodeState import node_info
from backend.service.authNodeService import automatic_node_verification
from backend.executeTask import validate_task_data
from pydantic import BaseModel
# Import auth functions but NOT from authNodeService directly
from backend.service.authNodeService import check_existing_node,validate_gpu, trigger_background_connection

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

# Task queues
task_queue: List[dict] = []
task_logs: Dict[str, list] = {}
completed_tasks: List[dict] = []

# === Utils ===

class NodeRegistrationRequest(BaseModel):
    node_name: str
    public_key: str

def get_country_from_ip(ip: str) -> str:
    try:
        res = requests.get(f"https://ipapi.co/{ip}/country_name/", timeout=5)
        if res.status_code == 200:
            return res.text.strip()
    except Exception as e:
        logger.warning(f"Failed to get country for IP {ip}: {e}")
    return "Unknown"

# === Routes ===

@app.post("/connect-node")
async def connect_node(payload: NodeRegistrationRequest):
    node_name = payload.node_name
    public_key = payload.public_key

    # ✅ Update local node_info but skip generating node_id here!
    node_info["connected"] = False
    node_info["node_name"] = node_name
    node_info["public_key"] = public_key

    # ✅ Validate GPU
    gpu_valid, error_response = validate_gpu()
    if not gpu_valid:
        return error_response

    # ✅ Call background connection and WAIT for response
    coordinator_response = await trigger_background_connection()

    if not coordinator_response or "node_id" not in coordinator_response:
        return JSONResponse(content={"error": "Failed to connect to coordinator, no node_id received."}, status_code=500)

    # ✅ Update local node_info with real node_id from coordinator
    node_info["node_id"] = coordinator_response["node_id"]
    node_info["connected"] = True

    # ✅ Now perform automatic verification with correct node_id
    await automatic_node_verification(node_info["node_id"])

    return {
        "status": coordinator_response.get("status", "success"),
        "connected": True,
        "node_id": node_info["node_id"]
    }

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
    capabilities = get_system_capabilities()  # Fetch capabilities dynamically each time
    return {
        "node_id": node_info.get("node_id", "unknown-node-id"),
        "system_info": capabilities,  # Use dynamically fetched capabilities
        "country": node_info["country"],
        "status": {"connected": node_info["connected"]}
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
        "node_id": node_id,  # node_id is included here
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
    node_id = task["node_id"]  # Ensure node_id is included in task

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
        "original_task_type": task_type,
        "node_id": node_id  # Ensure node_id is included in the result
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

import os
import uuid
import asyncio
import logging
from fastapi.responses import JSONResponse
import httpx
import psutil
import requests
import time
from typing import Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, Request, BackgroundTasks, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.service.systemInfoService import get_system_capabilities  # Dynamically fetch system capabilities
from backend.service.usageService import get_usage
from backend.shared.nodeState import node_info
from backend.executeTask import validate_task_data
from pydantic import BaseModel
from backend.database.nodedb import db  
from backend.service.authNodeService import validate_gpu, trigger_background_connection
from backend.utils.config import COORDINATOR_URL
from backend.service.taskExecutor import execute_task

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

# How often this node reports in to the coordinator. The coordinator marks a
# node disconnected after 5 minutes without a heartbeat.
HEARTBEAT_INTERVAL_SECONDS = int(os.getenv("HEARTBEAT_INTERVAL", 60))

# How often an idle node asks the coordinator for work.
TASK_POLL_INTERVAL_SECONDS = int(os.getenv("TASK_POLL_INTERVAL", 10))

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

    node_info["connected"] = False
    node_info["node_name"] = node_name
    node_info["public_key"] = public_key

    gpu_valid, error_response = validate_gpu()
    if not gpu_valid:
        return error_response

    # Skip calling trigger_background_connection() here!

    return {
        "status": "pending",
        "message": "Node information saved. Please proceed with verification."
    }


@app.post("/finalize-connection")
async def finalize_connection(request: Request):
    try:
        body = await request.json()
        req_public_key = body.get("public_key")
        if not req_public_key:
            raise ValueError("Missing public_key in request.")

        # 🧠 Restore or validate public key in memory
        if not node_info.get("public_key"):
            node_info["public_key"] = req_public_key
            logger.info("🔐 Public key restored from request body.")

        if node_info["public_key"] != req_public_key:
            raise HTTPException(status_code=400, detail="Public key mismatch. Aborting.")

        # 🔎 Recover node_id from the DB, or register with the coordinator if this
        # public key has never been seen before (first-time registration).
        existing_node = await db.nodes.find_one({"public_key": req_public_key})

        if existing_node:
            node_id = existing_node["_id"]
            node_info["node_id"] = node_id
        else:
            logger.info("🆕 Public key not registered yet — registering with the coordinator.")
            registration = await trigger_background_connection()

            if not registration or not registration.get("node_id"):
                raise HTTPException(
                    status_code=502,
                    detail="Coordinator did not return a node_id. Registration failed."
                )

            if registration.get("status") == "rejected":
                raise HTTPException(
                    status_code=400,
                    detail=registration.get("reason", "Coordinator rejected this node.")
                )

            node_id = registration["node_id"]
            node_info["node_id"] = node_id

        node_info["connected"] = True

        # ✅ Fresh system capabilities
        capabilities = get_system_capabilities()
        node_info["system_info"] = capabilities
        node_info["total_gpu_tflops"] = capabilities.get("total_gpu_tflops", 0)

        # Optional: log re-connection in DB
        await db.nodes.update_one(
            {"_id": node_id},
            {
                "$set": {
                    "last_connected": datetime.utcnow(),
                    "isConnected": True
                }
            }
        )

        logger.info(f"✅ Finalize connection complete for node {node_id}")
        return {
            "status": "success",
            "connected": True,
            "node_id": node_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"⚠️ Could not finalize connection: {e}")
        raise HTTPException(status_code=400, detail=str(e))


class NodeSessionPayload(BaseModel):
    node_id: str
    token: str


@app.post("/node-session")
async def store_node_session(payload: NodeSessionPayload):
    """Receive the session token the browser obtained from /verify-challenge.

    The private key never leaves the browser, so this handoff is the only way the
    node process itself gets credentials to authenticate its heartbeats with.
    """
    node_info["node_id"] = payload.node_id
    node_info["session_token"] = payload.token
    node_info["connected"] = True

    logger.info(f"🎟️ Stored session token for node {payload.node_id}.")

    # Report in right away so the node shows as connected without waiting a cycle.
    try:
        await send_heartbeat_once()
    except Exception as e:
        logger.warning(f"⚠️ Initial heartbeat failed: {e}")

    return {"status": "success", "node_id": payload.node_id}


async def send_heartbeat_once() -> bool:
    """Post one heartbeat to the coordinator. Returns False if we have no session."""
    node_id = node_info.get("node_id")
    token = node_info.get("session_token")

    if not node_id or not token:
        return False

    capabilities = get_system_capabilities()
    gpus = capabilities.get("gpu", []) if isinstance(capabilities, dict) else []
    loads = [
        gpu.get("load_percentage") for gpu in gpus
        if isinstance(gpu, dict) and isinstance(gpu.get("load_percentage"), (int, float))
    ]

    payload = {
        # interval=None so we never block the event loop
        "cpu_usage": psutil.cpu_percent(interval=None),
        "gpu_usage": round(sum(loads) / len(loads), 2) if loads else 0.0,
        "capabilities": capabilities,
    }

    async with httpx.AsyncClient() as client:
        res = await client.post(
            f"{COORDINATOR_URL}/node-heartbeat/{node_id}",
            json=payload,
            headers={"Authorization": f"Bearer {token}"},
            timeout=10,
        )

    if res.status_code in (401, 403):
        logger.warning("🔒 Coordinator rejected the session token — clearing it. Re-verify this node.")
        node_info.pop("session_token", None)
        return False

    res.raise_for_status()
    return True


async def heartbeat_loop():
    while True:
        try:
            await send_heartbeat_once()
        except Exception as e:
            logger.warning(f"⚠️ Heartbeat failed: {e}")
        await asyncio.sleep(HEARTBEAT_INTERVAL_SECONDS)


@app.on_event("startup")
async def start_heartbeat_task():
    asyncio.create_task(heartbeat_loop())
    logger.info(f"💓 Heartbeat loop started (every {HEARTBEAT_INTERVAL_SECONDS}s).")


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


async def claim_and_run_one_task() -> bool:
    """Claim one task from the coordinator and run it. True if work was done.

    The node reaches out; the coordinator never connects inward. That is what
    lets a contributor behind a home router take part at all.
    """
    node_id = node_info.get("node_id")
    token = node_info.get("session_token")

    if not node_id or not token:
        return False
    if not node_info.get("accept_tasks", True):
        return False

    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{COORDINATOR_URL}/next-task/{node_id}", headers=headers, timeout=15
        )

    if res.status_code in (401, 403):
        logger.warning("Coordinator rejected the session token while polling - clearing it.")
        node_info.pop("session_token", None)
        return False

    res.raise_for_status()
    task = (res.json() or {}).get("task")
    if not task:
        return False

    task_id = task["task_id"]
    logs = []

    def log(message):
        logger.info(f"[{task_id}] {message}")
        logs.append(message)

    task_logs[task_id] = logs
    log("Task started")

    try:
        # execute_task blocks (and real training will block hard), so it runs off
        # the event loop or heartbeats would stall for the whole job.
        outcome = await asyncio.to_thread(execute_task, task.get("task_data", {}), log)
    except Exception as e:
        logger.error(f"Task {task_id} raised: {e}")
        log(f"Error processing task: {e}")
        outcome = {"status": "failed", "result": f"Error: {e}"}

    payload = {
        "status": outcome.get("status", "completed"),
        "result": outcome.get("result"),
        "metrics": outcome.get("metrics", {}),
        "logs": logs,
    }

    try:
        async with httpx.AsyncClient() as client:
            report = await client.post(
                f"{COORDINATOR_URL}/task-result/{task_id}",
                json=payload, headers=headers, timeout=30,
            )
            report.raise_for_status()
        logger.info(f"Reported {payload['status']} for task {task_id}")
    except Exception as e:
        # The coordinator requeues tasks whose node went quiet, so a lost report
        # means the task is retried rather than dropped.
        logger.error(f"Could not report result for {task_id}: {e}")

    completed_tasks.append({"task_id": task_id, "status": payload["status"]})
    return True


async def task_poll_loop():
    while True:
        did_work = False
        try:
            did_work = await claim_and_run_one_task()
        except Exception as e:
            logger.warning(f"Task poll failed: {e}")
        # Drain a backlog quickly, but idle politely when there is nothing to do.
        await asyncio.sleep(0.5 if did_work else TASK_POLL_INTERVAL_SECONDS)


@app.on_event("startup")
async def start_task_poll_task():
    asyncio.create_task(task_poll_loop())
    logger.info(f"Task poller started (every {TASK_POLL_INTERVAL_SECONDS}s when idle).")


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
            log(f" Result sent to {origin_ip}, Response: {response.status_code}")
        except Exception as e:
            log(f"❌ Failed to send result to {origin_ip}: {e}")
    else:
        log(" No origin IP found, result not sent back.")




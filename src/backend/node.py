
import json
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
from backend.service.authNodeService import (
    validate_gpu,
    trigger_background_connection,
    find_node_id_by_public_key,
)
from backend.utils.config import COORDINATOR_URL
from backend.service.taskExecutor import execute_task
from backend.service.gpuBenchmark import benchmark_all
from backend.service.poolPlanner import pool_summary
from backend.service.thermalPolicy import STATE_STOP, thermal_status
from backend.service.artifacts import ArtifactError, pack_state_dict, unpack_dataset

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

# The session token arrives from the browser and used to live only in memory,
# so restarting the node silently stopped every heartbeat until somebody
# reconnected by hand. Persist it so a restart -- or a reboot on a
# contributor's machine -- recovers on its own.
SESSION_FILE = os.getenv("NODE_SESSION_FILE", "/app/data/node-session.json")


def save_session():
    """Write the current node_id and token so a restart can resume."""
    try:
        directory = os.path.dirname(SESSION_FILE)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(SESSION_FILE, "w", encoding="utf-8") as handle:
            json.dump({
                "node_id": node_info.get("node_id"),
                "session_token": node_info.get("session_token"),
                "approval_mode": node_info.get("approval_mode", "auto"),
            }, handle)
        os.chmod(SESSION_FILE, 0o600)
    except Exception as e:
        logger.warning(f"Could not save the node session to {SESSION_FILE}: {e}")


def load_session():
    """Restore a previously handed-over session, if there is one."""
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as handle:
            stored = json.load(handle)
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.warning(f"Could not read the node session from {SESSION_FILE}: {e}")
        return False

    if not stored.get("node_id") or not stored.get("session_token"):
        return False

    node_info["node_id"] = stored["node_id"]
    node_info["session_token"] = stored["session_token"]
    node_info["approval_mode"] = stored.get("approval_mode", "auto")
    node_info["connected"] = True
    logger.info(f"🔓 Restored session for node {stored['node_id']} from disk.")
    return True

# How often an idle node asks the coordinator for work.
TASK_POLL_INTERVAL_SECONDS = int(os.getenv("TASK_POLL_INTERVAL", 10))

# What the node is doing right now, so the dashboard can show live progress.
current_task: Dict[str, Any] = {}

# A job waiting for its owner to say yes, when approval_mode is "ask".
# It is only *peeked*, never claimed, so the coordinator can hand it to
# somebody else if this owner never answers.
pending_approval: Dict[str, Any] = {}

# How long a peeked job waits for a human before it goes back to the queue.
# Without this a submitter would wait forever on an owner who is asleep.
APPROVAL_TIMEOUT_SECONDS = int(os.getenv("APPROVAL_TIMEOUT", 120))

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

        # 🔎 Ask the coordinator whether this key is already registered, or
        # register it if not. The node has no database of its own.
        existing_id = await find_node_id_by_public_key(req_public_key)

        if existing_id:
            node_id = existing_id
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

        # The coordinator marks the node connected from the heartbeat that
        # follows the session handover, so there is nothing to write here.
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
    save_session()

    logger.info(f"🎟️ Stored session token for node {payload.node_id}.")

    # Report in right away so the node shows as connected without waiting a cycle.
    try:
        await send_heartbeat_once()
    except Exception as e:
        logger.warning(f"⚠️ Initial heartbeat failed: {e}")

    return {"status": "success", "node_id": payload.node_id}


_warned_about_missing_session = False


class ApprovalModePayload(BaseModel):
    mode: str


@app.post("/approval-mode")
async def set_approval_mode(payload: ApprovalModePayload):
    """Choose whether jobs run automatically or wait for a click."""
    if payload.mode not in ("auto", "ask"):
        raise HTTPException(status_code=400, detail="mode must be 'auto' or 'ask'")

    node_info["approval_mode"] = payload.mode
    save_session()
    logger.info(f"Approval mode set to {payload.mode}.")
    return {"status": "success", "mode": payload.mode}


@app.post("/approve-task/{task_id}")
async def approve_task(task_id: str):
    """Owner said yes: claim the job properly and start it."""
    if pending_approval.get("task_id") != task_id:
        raise HTTPException(status_code=409, detail="That job is no longer waiting.")

    node_id = node_info.get("node_id")
    token = node_info.get("session_token")
    if not node_id or not token:
        raise HTTPException(status_code=409, detail="This node has no session.")

    pending_approval.clear()
    headers = {"Authorization": f"Bearer {token}"}

    # Claim for real now. The oldest pending task is the one just approved,
    # since nothing older can appear after it was peeked.
    task = await _fetch_next_task(node_id, headers, claim=True)
    if not task:
        raise HTTPException(status_code=409, detail="The job is no longer available.")

    asyncio.create_task(_run_task(task, node_id, headers))
    return {"status": "success", "task_id": task["task_id"]}


@app.post("/decline-task/{task_id}")
async def decline_task(task_id: str):
    """Owner said no: hand the job back to the coordinator."""
    if pending_approval.get("task_id") != task_id:
        raise HTTPException(status_code=409, detail="That job is no longer waiting.")

    token = node_info.get("session_token")
    if not token:
        raise HTTPException(status_code=409, detail="This node has no session.")

    pending_approval.clear()
    await _decline_task(task_id, {"Authorization": f"Bearer {token}"},
                        "Declined by the node owner.")
    return {"status": "success", "task_id": task_id}


@app.get("/current-task")
async def get_current_task():
    """What this node is doing right now, plus how hot it is.

    The dashboard polls this so a contributor can watch work go through and see
    the temperature that would stop it.
    """
    waiting = None
    if pending_approval:
        elapsed = time.time() - pending_approval.get("peeked_at_ts", 0)
        waiting = {
            **{k: v for k, v in pending_approval.items() if k != "peeked_at_ts"},
            "seconds_left": max(0, int(APPROVAL_TIMEOUT_SECONDS - elapsed)),
        }

    return {
        "task": current_task or None,
        "awaiting_approval": waiting,
        "approval_mode": node_info.get("approval_mode", "auto"),
        "thermal": thermal_status(),
        "accepting_work": bool(node_info.get("session_token"))
                          and node_info.get("accept_tasks", True),
    }


async def send_heartbeat_once() -> bool:
    """Post one heartbeat to the coordinator. Returns False if we have no session."""
    global _warned_about_missing_session

    node_id = node_info.get("node_id")
    token = node_info.get("session_token")

    if not node_id or not token:
        # This used to return silently, so a node that lost its session looked
        # healthy locally while the coordinator quietly marked it disconnected.
        if not _warned_about_missing_session:
            logger.warning(
                "No node session yet — heartbeats are paused. Open the dashboard "
                "and connect this node to hand over a session token."
            )
            _warned_about_missing_session = True
        return False

    _warned_about_missing_session = False

    capabilities = get_system_capabilities()

    # benchmark_all() is cached, so this is only expensive the first time; the
    # cache is warmed on startup. The spec-sheet figure stays alongside it for
    # comparison, but measured throughput is what the network should schedule on.
    try:
        summary = pool_summary(benchmark_all())
        if summary["device_count"]:
            capabilities["measured_tflops"] = summary["pooled_tflops"]
            capabilities["theoretical_tflops"] = capabilities.get("total_gpu_tflops")
            capabilities["total_gpu_tflops"] = summary["pooled_tflops"]
    except Exception as e:
        logger.warning(f"Could not attach measured throughput: {e}")

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
        save_session()
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


async def warm_benchmark():
    """Measure the GPUs once, off the event loop, so heartbeats stay cheap."""
    try:
        devices = await asyncio.to_thread(benchmark_all)
        summary = pool_summary(devices)
        if summary["device_count"]:
            logger.info(
                f"📊 Measured {summary['pooled_tflops']} TFLOPS across "
                f"{summary['device_count']} GPU(s)."
            )
    except Exception as e:
        logger.warning(f"GPU benchmark failed: {e}")


@app.on_event("startup")
async def start_heartbeat_task():
    load_session()
    asyncio.create_task(warm_benchmark())
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

    # Do not pick up new work on a card that is already too hot; let it cool.
    thermal = thermal_status()
    if thermal["state"] == STATE_STOP:
        logger.warning(f"Not claiming work — {thermal['reason']}")
        return False

    headers = {"Authorization": f"Bearer {token}"}

    # When the owner wants to approve each job, look without claiming.
    if node_info.get("approval_mode", "auto") == "ask":
        return await _handle_approval_mode(node_id, headers)

    task = await _fetch_next_task(node_id, headers, claim=True)
    if not task:
        return False

    return await _run_task(task, node_id, headers)


async def _fetch_next_task(node_id, headers, claim: bool):
    """Ask the coordinator for the next task, claiming it or only peeking."""
    async with httpx.AsyncClient() as client:
        res = await client.get(
            f"{COORDINATOR_URL}/next-task/{node_id}",
            params={"claim": str(claim).lower()},
            headers=headers, timeout=15,
        )

    if res.status_code in (401, 403):
        logger.warning("Coordinator rejected the session token while polling - clearing it.")
        node_info.pop("session_token", None)
        return None

    res.raise_for_status()
    return (res.json() or {}).get("task")


async def _handle_approval_mode(node_id, headers) -> bool:
    """Peek at the next job and wait for the owner, with a timeout."""
    if pending_approval:
        waited = time.time() - pending_approval.get("peeked_at_ts", 0)
        if waited > APPROVAL_TIMEOUT_SECONDS:
            task_id = pending_approval.get("task_id")
            logger.info(f"No answer on {task_id} after {int(waited)}s — returning it to the queue.")
            await _decline_task(task_id, headers,
                                "No answer from the node owner in time.")
            pending_approval.clear()
        return False        # one decision at a time

    task = await _fetch_next_task(node_id, headers, claim=False)
    if not task:
        return False

    pending_approval.clear()
    pending_approval.update({
        "task_id": task["task_id"],
        "model_name": (task.get("task_data") or {}).get("model_name"),
        "task_type": (task.get("task_data") or {}).get("task_type"),
        "has_dataset": bool(task.get("dataset_id")),
        "peeked_at_ts": time.time(),
        "expires_in": APPROVAL_TIMEOUT_SECONDS,
    })
    logger.info(f"Job {task['task_id']} is waiting for approval.")
    return False


async def _decline_task(task_id, headers, reason):
    """Hand a peeked job back rather than leaving a submitter hanging."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{COORDINATOR_URL}/task-result/{task_id}",
                json={"status": "rejected", "result": reason, "metrics": {}, "logs": []},
                headers=headers, timeout=15,
            )
            res.raise_for_status()
    except Exception as e:
        logger.warning(f"Could not decline {task_id}: {e}")


async def _run_task(task, node_id, headers) -> bool:

    task_id = task["task_id"]
    logs = []

    def log(message):
        logger.info(f"[{task_id}] {message}")
        logs.append(message)

    task_logs[task_id] = logs
    current_task.clear()
    current_task.update({
        "task_id": task_id,
        "model_name": (task.get("task_data") or {}).get("model_name"),
        "started_at": datetime.utcnow().isoformat(),
        "status": "running",
        "logs": logs,
    })
    log("Task started")

    # Fetch the submitter's dataset, if this job carries one. unpack_dataset
    # refuses anything that could execute code, so a hostile payload fails here
    # rather than on this contributor's machine.
    dataset = None
    dataset_id = task.get("dataset_id")
    if dataset_id:
        try:
            log(f"Downloading dataset {dataset_id}")
            async with httpx.AsyncClient() as client:
                res = await client.get(
                    f"{COORDINATOR_URL}/artifacts/{dataset_id}",
                    headers=headers, timeout=120,
                )
                res.raise_for_status()
            dataset = unpack_dataset(res.content)
            log(f"Dataset received: {dataset[0].shape[0]:,} rows")
        except ArtifactError as e:
            log(f"Rejected the dataset: {e}")
            await _report_result(task_id, headers,
                                 {"status": "failed",
                                  "result": f"Dataset rejected: {e}",
                                  "metrics": {}}, logs)
            return True
        except Exception as e:
            log(f"Could not download the dataset: {e}")
            await _report_result(task_id, headers,
                                 {"status": "failed",
                                  "result": f"Dataset download failed: {e}",
                                  "metrics": {}}, logs)
            return True

    try:
        # execute_task blocks (and real training will block hard), so it runs off
        # the event loop or heartbeats would stall for the whole job.
        outcome = await asyncio.to_thread(
            execute_task, task.get("task_data", {}), log, dataset
        )
    except Exception as e:
        logger.error(f"Task {task_id} raised: {e}")
        log(f"Error processing task: {e}")
        outcome = {"status": "failed", "result": f"Error: {e}"}

    # Hand the trained weights back so the submitter can collect them.
    weights_id = None
    state_dict = outcome.get("state_dict")
    if state_dict:
        try:
            blob = pack_state_dict(state_dict)
            async with httpx.AsyncClient() as client:
                res = await client.post(
                    f"{COORDINATOR_URL}/artifacts?kind=weights",
                    content=blob,
                    headers={**headers, "Content-Type": "application/octet-stream"},
                    timeout=120,
                )
                res.raise_for_status()
            weights_id = (res.json() or {}).get("artifact_id")
            log(f"Uploaded trained weights ({len(blob):,} bytes) as {weights_id}")
        except Exception as e:
            # The training itself succeeded, so report it rather than losing it.
            log(f"Could not upload trained weights: {e}")

    current_task["status"] = outcome.get("status", "completed")
    current_task["finished_at"] = datetime.utcnow().isoformat()

    await _report_result(task_id, headers, outcome, logs, weights_id)
    completed_tasks.append({"task_id": task_id, "status": outcome.get("status")})
    return True


async def _report_result(task_id, headers, outcome, logs, weights_id=None):
    """Send a task outcome to the coordinator."""
    payload = {
        "status": outcome.get("status", "completed"),
        "result": outcome.get("result"),
        "metrics": outcome.get("metrics", {}),
        "weights_id": weights_id,
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




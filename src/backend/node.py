
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
from backend.service.taskExecutor import JobCancelled, execute_task
from backend.service.gpuBenchmark import benchmark_all
from backend.service.poolPlanner import pool_summary
from backend.service.thermalPolicy import STATE_OK, STATE_STOP, thermal_status
from backend.service.artifacts import (
    ArtifactError, pack_state_dict, parse_csv_dataset, unpack_dataset,
)

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

    task_view = None
    if current_task:
        task_view = {k: v for k, v in current_task.items() if k != "started_ts"}
        if current_task.get("started_ts"):
            task_view["elapsed_s"] = round(time.time() - current_task["started_ts"], 1)

    return {
        "task": task_view,
        "awaiting_approval": waiting,
        "approval_mode": node_info.get("approval_mode", "auto"),
        "thermal": thermal_status(),
        "accepting_work": bool(node_info.get("session_token"))
                          and node_info.get("accept_tasks", True),
    }


def is_busy() -> bool:
    """Whether this node is occupied: a real job, or its own test."""
    return current_task.get("status") == "running" or bool(self_test.get("running"))


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
        # Whether the card is occupied. The coordinator can see a claimed job
        # by itself, but a self test never becomes a coordinator task, so
        # without this the node looks idle while it is flat out.
        "busy": is_busy(),
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


# --- self test -----------------------------------------------------------
#
# A contributor could register a node and have no idea whether it actually
# works until a stranger's job either ran or did not. This trains a small model
# locally, start to finish, through the same code a real job goes through.
#
# It also produces a better throughput figure than the synthetic benchmark. The
# benchmark times a burst of matrix multiplications, which flatters the card:
# on this machine it reports around 18 TFLOPS while a real transformer job
# sustains closer to 6. What a submitter cares about is how quickly their job
# finishes, so the sustained figure is the honest one to schedule on.
#
# Deliberately compute-bound. A tiny MLP would finish instantly and report a
# throughput near zero, which would say nothing about the card.
# Sized to saturate the card rather than to finish quickly. An earlier version
# ran 60 steps of a small model in 1.6 seconds and reported 1.25 TFLOPS against
# a 41 TFLOPS benchmark -- almost all of that gap was CUDA warm-up and model
# setup, not the card. This is the same shape of workload a real training job
# has, run for long enough that the start-up cost stops dominating.
SELF_TEST_TASK = {
    "task_type": "llm_training",
    "model_name": "self-test",
    "model_spec": {
        "architecture": "transformer",
        "vocab_size": 4096,
        "d_model": 384,
        "n_head": 6,
        "n_layer": 4,
        "seq_len": 256,
    },
    "hyperparameters": {"steps": 200, "batch_size": 24, "learning_rate": 0.0003},
}

# When the owner supplies a CSV the point changes: not "how fast is this card"
# but "does my own data train here". A feedforward net is the shape that fits
# rows of numbers, and its dimensions come from the file.
SELF_TEST_CSV_TASK = {
    "task_type": "llm_training",
    "model_name": "self-test",
    "model_spec": {"architecture": "mlp", "hidden_dim": 64, "depth": 2},
    "hyperparameters": {"steps": 200, "batch_size": 32, "learning_rate": 0.01},
}

# The long test. A quick run finishes before a card has warmed up, so it says
# nothing about whether a machine can hold a real job: fans spin up over
# minutes, and a case with poor airflow only shows itself once the heat has
# nowhere left to go. This runs until the owner stops it or the time is up,
# and reports how hot it got.
#
# Bigger than the quick test so the card is genuinely loaded rather than
# waiting on Python between steps. The step count is deliberately far more
# than can finish in the time: the clock ends the run, not the steps.
SELF_TEST_STRESS_TASK = {
    "task_type": "llm_training",
    "model_name": "self-test",
    "model_spec": {
        "architecture": "transformer",
        "vocab_size": 8192,
        "d_model": 512,
        "n_head": 8,
        "n_layer": 6,
        "seq_len": 256,
    },
    "hyperparameters": {"steps": 1_000_000, "batch_size": 24, "learning_rate": 0.0003},
}

STRESS_SECONDS = int(os.getenv("SELF_TEST_STRESS_SECONDS", 300))

# NVML is cheap but not free, and the progress hook fires many times a second.
THERMAL_SAMPLE_SECONDS = 1.0

self_test: Dict[str, Any] = {}


@app.get("/self-test")
async def get_self_test():
    """The last self test result, if one has been run."""
    return {"result": self_test or None, "running": bool(self_test.get("running"))}


@app.post("/self-test/stop")
async def stop_self_test():
    """Ask a running test to stop at its next step."""
    if not self_test.get("running"):
        raise HTTPException(status_code=409, detail="No test is running.")

    self_test["stop_requested"] = True
    self_test["ended_because"] = "stopped"
    logger.info("Self test stop requested by the owner.")
    return {"status": "success", "stopping": True}


@app.post("/self-test")
async def run_self_test(request: Request, mode: str = "quick"):
    """Train a small model here and now, and report what the card managed.

    An optional CSV body trains on the owner's own data instead of synthetic
    tokens. It is parsed and used here and never sent anywhere -- a self test
    involves no coordinator and no network.
    """
    if current_task.get("status") == "running":
        raise HTTPException(
            status_code=409,
            detail="This node is running a job. Try again when it has finished.",
        )
    if self_test.get("running"):
        raise HTTPException(status_code=409, detail="A self test is already running.")

    logs: List[str] = []

    def log(message):
        logs.append(str(message))

    # --- the owner's data, if they supplied any --------------------------
    stress = mode == "stress"

    dataset = None
    task_data = dict(SELF_TEST_STRESS_TASK if stress else SELF_TEST_TASK)

    body = await request.body()
    if body and body.strip():
        try:
            features, labels, class_names = parse_csv_dataset(
                body.decode("utf-8", errors="replace")
            )
        except (ArtifactError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"That CSV could not be read: {e}")

        dataset = (features, labels)
        task_data = dict(SELF_TEST_CSV_TASK)
        log(f"Training on your CSV: {features.shape[0]:,} rows, "
            f"{features.shape[1]} features")
        if class_names:
            log(f"Labels: {', '.join(class_names[:8])}")

    self_test.clear()
    self_test.update({
        "running": True,
        "mode": mode,
        "started_at": datetime.utcnow().isoformat(),
        "logs": logs,
        "used_dataset": dataset is not None,
        "stop_requested": False,
        "ended_because": None,
        "peak_temperature": None,
        "peak_power_w": None,
        "peak_utilisation": None,
        "ran_hot": False,
    })

    if stress:
        log(f"Working the card for up to {STRESS_SECONDS // 60} minutes. "
            f"Stop it whenever you like.")

    # Presented exactly like a real job, so the owner watches it in the same
    # live view -- progress, loss, temperature -- rather than staring at a
    # button. The dashboard reads all of this from /current-task.
    current_task.clear()
    current_task.update({
        "task_id": "self-test",
        "model_name": "Test job" + (" (your data)" if dataset else ""),
        "started_at": datetime.utcnow().isoformat(),
        "started_ts": time.time(),
        "status": "running",
        "logs": logs,
        "progress": None,
        "self_test": True,
    })

    started = time.time()
    last_sample = [0.0]

    def on_progress(update):
        elapsed = time.time() - started

        # A stress run is bounded by the clock, not the step count -- that is
        # set absurdly high on purpose. Reporting steps would leave the bar at
        # 0% for the whole five minutes, so it counts down the time instead.
        if stress:
            remaining = max(0, STRESS_SECONDS - int(elapsed))
            update = {
                **update,
                "step": min(int(elapsed), STRESS_SECONDS),
                "steps": STRESS_SECONDS,
                "label": f"{int(elapsed) // 60}:{int(elapsed) % 60:02d}"
                         f" of {STRESS_SECONDS // 60}:{STRESS_SECONDS % 60:02d}"
                         f" — {remaining // 60}:{remaining % 60:02d} left",
            }

        current_task["progress"] = update

        # Sampled rather than read every step: the hook fires many times a
        # second and NVML is not free.
        if elapsed - last_sample[0] >= THERMAL_SAMPLE_SECONDS:
            last_sample[0] = elapsed
            try:
                status = thermal_status()
            except Exception:
                status = None

            if status and status.get("gpus"):
                hottest = status["hottest"] or status["gpus"][0]
                for key, value in (
                    ("peak_temperature", hottest.get("temperature")),
                    ("peak_power_w", hottest.get("power_w")),
                    ("peak_utilisation", hottest.get("utilisation")),
                ):
                    if value is not None and (self_test.get(key) is None
                                              or value > self_test[key]):
                        self_test[key] = value

                if status.get("state") != STATE_OK:
                    self_test["ran_hot"] = True

                current_task["thermal_peak"] = {
                    "temperature": self_test.get("peak_temperature"),
                    "power_w": self_test.get("peak_power_w"),
                }

        if self_test.get("stop_requested"):
            raise JobCancelled("Stopped by the owner.")

        # The clock ends a stress run, not the step count.
        if stress and elapsed >= STRESS_SECONDS:
            self_test["ended_because"] = "finished"
            raise JobCancelled("Ran for the full time.")

    # Tell the coordinator now rather than at the next scheduled heartbeat: a
    # test is over in seconds, so a minute's delay would report it as idle for
    # its whole duration and then busy after it had finished.
    try:
        await send_heartbeat_once()
    except Exception as e:
        logger.debug(f"Could not announce the self test: {e}")

    try:
        outcome = await asyncio.to_thread(
            execute_task, task_data, log, dataset, on_progress
        )
    except Exception as e:
        logger.error(f"Self test raised: {e}")
        current_task.update({"status": "failed",
                             "finished_at": datetime.utcnow().isoformat()})
        self_test.update({"running": False, "status": "failed", "result": str(e)})
        return {"status": "failed", "result": str(e), "logs": logs}

    metrics = outcome.get("metrics") or {}

    current_task.update({
        "status": "completed" if (stress and outcome.get("status") == "cancelled")
                  else outcome.get("status", "completed"),
        "result": outcome.get("result"),
        "metrics": metrics,
        "finished_at": datetime.utcnow().isoformat(),
    })

    # The two figures answer different questions, so both are reported rather
    # than one being quietly replaced by the other.
    # A stress run that hits its time limit, or that the owner stops, comes
    # back as "cancelled" from the executor -- which is accurate for a job but
    # wrong for a test that did exactly what was asked of it.
    status = outcome.get("status")
    ended = self_test.get("ended_because")
    if status == "cancelled" and stress:
        status = "completed"

    result = {
        "running": False,
        "mode": mode,
        "used_dataset": dataset is not None,
        "ended_because": ended or ("finished" if status == "completed" else None),
        "seconds_run": round(time.time() - started, 1),
        "peak_temperature": self_test.get("peak_temperature"),
        "peak_power_w": self_test.get("peak_power_w"),
        "peak_utilisation": self_test.get("peak_utilisation"),
        "ran_hot": bool(self_test.get("ran_hot")),
        "status": status,
        "result": outcome.get("result"),
        "finished_at": datetime.utcnow().isoformat(),
        "sustained_tflops": metrics.get("achieved_tflops"),
        "peak_tflops": metrics.get("pooled_tflops"),
        "steps": metrics.get("steps"),
        "seconds": metrics.get("seconds"),
        "initial_loss": metrics.get("initial_loss"),
        "final_loss": metrics.get("final_loss"),
        "devices": metrics.get("devices"),
        "logs": logs,
    }

    # Did it actually learn? A card that runs without reducing the loss is
    # producing numbers, not training.
    first, last = result["initial_loss"], result["final_loss"]
    result["learned"] = bool(
        first is not None and last is not None and last < first
    )
    if stress:
        # Nothing is returned by a run that was interrupted, and a heat test
        # was never about the loss.
        result["learned"] = True

    self_test.clear()
    self_test.update(result)

    logger.info(
        f"Self test finished: {result['status']}, "
        f"{result['sustained_tflops']} TFLOPS sustained, learned={result['learned']}"
    )

    try:
        await send_heartbeat_once()      # free again
    except Exception as e:
        logger.debug(f"Could not announce the end of the self test: {e}")

    return result


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
    if self_test.get("running"):
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


CANCEL_POLL_SECONDS = int(os.getenv("CANCEL_POLL_INTERVAL", 3))


async def _watch_for_cancel(task_id, headers):
    """Set the cancel flag if the submitter asks the job to stop.

    Runs only while a job is running. The heartbeat is once a minute and the
    task poller is blocked for the duration of a job, so neither could carry
    this without making cancellation take a minute or never arrive.
    """
    url = f"{COORDINATOR_URL}/task-cancelled/{task_id}"
    while True:
        await asyncio.sleep(CANCEL_POLL_SECONDS)
        try:
            async with httpx.AsyncClient() as client:
                res = await client.get(url, headers=headers, timeout=10)
            if res.status_code == 200 and (res.json() or {}).get("cancel_requested"):
                current_task["cancel_requested"] = True
                logger.info(f"Task {task_id} was cancelled by the submitter.")
                return
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A watcher that cannot reach the coordinator must not take the
            # job down with it; the job simply stays uncancellable for now.
            logger.debug(f"Cancellation check failed: {e}")


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
        "started_ts": time.time(),
        "status": "running",
        "logs": logs,
        "progress": None,
        "cancel_requested": False,
    })
    log("Task started")

    def on_progress(update):
        """Structured numbers for the dashboard, written as the job runs.

        Also the point where a cancellation takes effect. This runs in the
        worker thread once per step, so it only reads a flag -- the flag is set
        by _watch_for_cancel on the event loop, which is where the HTTP call
        belongs.
        """
        current_task["progress"] = update
        if current_task.get("cancel_requested"):
            raise JobCancelled("Cancelled by the submitter.")

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

    watcher = asyncio.create_task(_watch_for_cancel(task_id, headers))
    try:
        # execute_task blocks (and real training will block hard), so it runs off
        # the event loop or heartbeats would stall for the whole job.
        outcome = await asyncio.to_thread(
            execute_task, task.get("task_data", {}), log, dataset, on_progress
        )
    except Exception as e:
        logger.error(f"Task {task_id} raised: {e}")
        log(f"Error processing task: {e}")
        outcome = {"status": "failed", "result": f"Error: {e}"}
    finally:
        watcher.cancel()

    if outcome.get("status") == "cancelled":
        log("Stopped at the submitter's request.")

    # Hand the trained weights back so the submitter can collect them.
    weights_id = None
    state_dict = outcome.get("state_dict")
    if state_dict:
        try:
            blob = pack_state_dict(state_dict, outcome.get("manifest"))
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
            if outcome.get("manifest"):
                log("Model description packed with the weights.")
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




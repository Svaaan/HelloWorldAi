import os
import socket
import uuid
import logging
import requests
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from backend.service.runningNodeService import process_task
from backend.service.systemInfoService import get_system_capabilities
from backend.service.connectionService import background_connection_handler
from GPUtil import getGPUs

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Compute Node", description="Distributed Computing Node")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

def get_gpu_info_list():
    try:
        gpus = getGPUs()
        return [{
            "name": gpu.name,
            "total_memory": gpu.memoryTotal,
            "free_memory": gpu.memoryFree,
            "load_percentage": round(gpu.load * 100, 1),
            "temperature": gpu.temperature
        } for gpu in gpus]
    except Exception as e:
        print("⚠️ GPU detection failed:", e)
        return []


async def connect_node(background_tasks: BackgroundTasks):
    if node_info["connected"]:
        return {
            "status": "Node already connected",
            "connected": True,
            "node_id": node_info["node_id"]
        }

    # Auto-fetch GPU capabilities before sending (multi-GPU support)
    try:
        from GPUtil import getGPUs
        gpus = getGPUs()
        node_info["capabilities"]["gpus"] = []

        for gpu in gpus:
            node_info["capabilities"]["gpus"].append({
                "name": gpu.name,
                "total_memory": gpu.memoryTotal,
                "free_memory": gpu.memoryFree,
                "load_percentage": round(gpu.load * 100, 1),
                "temperature": gpu.temperature
            })
    except Exception as e:
        print("⚠️ Could not fetch GPU info:", e)
        node_info["capabilities"]["gpus"] = []

    payload = {
        "node_id": node_info["node_id"],
        "ip": public_ip,
        "country": node_info["country"],
        "capabilities": node_info["capabilities"]
    }

    background_tasks.add_task(background_connection_handler, payload, node_info)

    return {
        "status": "Connection in progress",
        "connected": False,
        "node_id": node_info["node_id"]
    }


@app.get("/usage")
def get_usage():
    from psutil import cpu_percent, virtual_memory
    import GPUtil

    try:
        cpu_usage = cpu_percent(interval=1)
        gpus = GPUtil.getGPUs()
        gpu_usage = round(gpus[0].load * 100, 2) if gpus else 0
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

@app.post("/compute")
def compute(task: Dict[str, Any], request: Request):
    return process_task(task, node_info, request)

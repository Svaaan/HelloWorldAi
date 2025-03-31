import os
import socket
import uuid
import logging
from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
from backend.service.runningNodeService import process_task
from backend.service.systemInfoService import get_system_capabilities
from backend.service.connectionService import background_connection_handler

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI app
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
        return os.getenv('NODE_HOSTNAME', socket.gethostname())
    except Exception as e:
        logger.warning(f"IP retrieval error: {e}")
        return "localhost"

# Node configuration
node_info = {
    "node_id": f"node_{uuid.uuid4()}",
    "ip": get_node_ip(),
    "port": os.getenv('PORT', '9100'),
    "capabilities": get_system_capabilities(),
    "connected": False,
    "accept_tasks": True,
    "allowed_clients": ["trusted-client-1", "trusted-client-2"],
    "last_heartbeat": None,
    "total_tasks_processed": 0
}

@app.post("/connect-node")
async def connect_node(background_tasks: BackgroundTasks):
    if node_info["connected"]:
        return {
            "status": "Node already connected",
            "connected": True,
            "node_id": node_info["node_id"]
        }

    payload = {
        "node_id": node_info["node_id"],
        "ip": node_info["ip"],
        "port": node_info["port"],
        "capabilities": node_info["capabilities"]
    }

    background_tasks.add_task(background_connection_handler, payload, node_info)

    return {
        "status": "Connection in progress",
        "connected": False,
        "node_id": node_info["node_id"]
    }

@app.get("/")
def get_node_status():
    return {
        "status": "online",
        "connected": node_info["connected"],
        "node": node_info
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
        "network": {
            "ip": node_info["ip"],
            "port": node_info["port"]
        },
        "status": {
            "connected": node_info["connected"],
            "total_tasks_processed": node_info["total_tasks_processed"]
        }
    }

@app.post("/compute")
def compute(task: Dict[str, Any], request: Request):
    return process_task(task, node_info, request)

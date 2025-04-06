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
from psutil import cpu_percent, virtual_memory
import pynvml  

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

def connect_to_coordinator():
    # ✅ Check for GPU first!
    gpu_info = get_gpu_info_list()
    if not gpu_info:
        print("❌ No eligible GPU found. Skipping coordinator connection.")
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
        print(f"✅ Coordinator response: {response.json()}")
    except Exception as e:
        print(f"❌ Failed to connect to coordinator: {e}")


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

    node_info["capabilities"]["gpu"] = get_gpu_info_list()

    payload = {
    "node_id": node_info["node_id"],
    "ip": public_ip,
    "country": node_info["country"],
    "capabilities": node_info["capabilities"],
    "isConnected": node_info.get("connected", False),
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
        "connected": False,
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

@app.post("/compute")
def compute(task: Dict[str, Any], request: Request):
    return process_task(task, node_info, request)

@app.on_event("startup")
async def startup_event():
    connect_to_coordinator()

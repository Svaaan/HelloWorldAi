import os
import socket
import psutil
import platform
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import GPUtil
import uuid
from typing import Dict, Optional

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_local_ip():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        print(f"Error getting IP address: {e}")
        return "127.0.0.1"

def get_advanced_gpu_capabilities() -> Dict:
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                "name": gpu.name,
                "total_memory": gpu.memoryTotal,  # Total GPU memory in MB
                "free_memory": gpu.memoryFree,
                "used_memory": gpu.memoryUsed,
                "load_percentage": gpu.load * 100,
                "temperature": gpu.temperature
            }
        return {"name": "No GPU"}
    except Exception as e:
        return {"name": "GPU detection failed", "error": str(e)}

def get_system_capabilities():
    # CPU info
    cpu_info = psutil.cpu_freq()
    cpu = {
        "brand": platform.processor(),
        "cores": psutil.cpu_count(logical=False),
        "threads": psutil.cpu_count(logical=True),
        "max_freq": cpu_info.max if cpu_info else None,
        "min_freq": cpu_info.min if cpu_info else None
    }

    # GPU info
    gpu = get_advanced_gpu_capabilities()

    return {"cpu": cpu, "gpu": gpu}

# Node configuration
node_info = {
    "node_id": f"node_{uuid.uuid4()}",
    "ip": get_local_ip(),
    "port": "9100",
    "capabilities": get_system_capabilities(),
    "connected": False
}

@app.get("/")
def get_node_status():
    return {
        "status": "online",
        "connected": node_info["connected"],
        "node": node_info
    }

@app.post("/connect-node")
async def connect_node():
    if node_info["connected"]:
        return {"status": "Node already connected", "connected": node_info["connected"]}

    payload = {
        "node_id": node_info["node_id"],
        "ip": node_info["ip"],
        "port": node_info["port"],
        "capabilities": node_info["capabilities"]
    }

    try:
        res = requests.post("http://127.0.0.1:8100/connect-node", json=payload)

        if res.status_code == 200:
            node_info["connected"] = True
            return {"status": "Node connected successfully!", "connected": node_info["connected"]}
        else:
            raise HTTPException(status_code=400, detail=f"Connection failed: {res.text}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to coordinator: {str(e)}")

@app.get("/usage")
def get_usage():
    cpu_usage = psutil.cpu_percent(interval=1)
    
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].load * 100 if gpus else 0

    return {
        "cpu_usage": cpu_usage,
        "gpu_usage": gpu_usage
    }

@app.get("/node-capabilities")
def get_detailed_capabilities():
    return {
        "node_id": node_info["node_id"],
        "system_info": node_info["capabilities"],
        "network": {
            "ip": node_info["ip"],
            "port": node_info["port"]
        }
    }

@app.post("/compute")
def compute(task):
    """
    Basic computation handler with GPU awareness
    """
    gpu_info = node_info["capabilities"]["gpu"]
    
    # Check GPU availability
    if gpu_info.get("name", "No GPU") != "No GPU":
        # Placeholder for GPU-accelerated computation
        result = _process_gpu_task(task)
        return {
            "task_id": task.get("task_id", str(uuid.uuid4())),
            "result": result,
            "gpu_used": gpu_info["name"],
            "gpu_load": gpu_info.get("load_percentage", 0)
        }
    
    # Fallback to CPU computation
    return _process_cpu_task(task)

def _process_gpu_task(task):
    # Simulated GPU task processing
    return str(task).upper()

def _process_cpu_task(task):
    # Simulated CPU task processing
    return str(task).lower()
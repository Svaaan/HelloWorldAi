import os
import socket
import psutil
import platform
import requests
import uuid
import logging
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import GPUtil
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Compute Node", description="Distributed Computing Node")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_node_ip() -> str:
    try:
        # For cloud environments, use environment variable or default to 0.0.0.0
        return os.getenv('NODE_IP', '0.0.0.0')
    except Exception as e:
        logger.warning(f"IP retrieval error: {e}")
        return "0.0.0.0"

def get_advanced_gpu_capabilities() -> Dict[str, Any]:
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            return {
                "name": gpu.name,
                "total_memory": gpu.memoryTotal,
                "free_memory": gpu.memoryFree,
                "used_memory": gpu.memoryUsed,
                "load_percentage": round(gpu.load * 100, 2),
                "temperature": gpu.temperature
            }
        return {"name": "No GPU Detected"}
    except Exception as e:
        logger.error(f"GPU detection error: {e}")
        return {"name": "GPU Detection Limited"}

def get_system_capabilities() -> Dict[str, Any]:
    try:
        cpu_info = psutil.cpu_freq()
        cpu = {
            "brand": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "max_freq": round(cpu_info.max, 2) if cpu_info else None,
            "min_freq": round(cpu_info.min, 2) if cpu_info else None,
            "current_freq": round(cpu_info.current, 2) if cpu_info else None
        }

        gpu = get_advanced_gpu_capabilities()
        return {"cpu": cpu, "gpu": gpu}
    except Exception as e:
        logger.error(f"System capabilities error: {e}")
        return {"error": "Limited system capabilities"}

# Node configuration with environment-aware initialization
node_info = {
    "node_id": f"node_{uuid.uuid4()}",
    "ip": get_node_ip(),
    "port": os.getenv('PORT', '9100'),
    "capabilities": get_system_capabilities(),
    "connected": False,
    "last_heartbeat": None,
    "total_tasks_processed": 0
}

def background_connection_handler(payload: Dict[str, Any]):
    """
    Background task for handling node connection
    """
    coordinator_url = os.getenv('COORDINATOR_URL', 'http://localhost:8100/connect-node')
    try:
        res = requests.post(coordinator_url, json=payload, timeout=10)
        
        if res.status_code == 200:
            node_info["connected"] = True
            logger.info("Node connected successfully to coordinator")
        else:
            logger.error(f"Connection failed: {res.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"Connection error: {e}")

@app.post("/connect-node")
async def connect_node(background_tasks: BackgroundTasks):
    """
    Endpoint to connect this node to the coordinator
    """
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

    # Use background task for connection
    background_tasks.add_task(background_connection_handler, payload)

    return {
        "status": "Connection in progress", 
        "connected": False,
        "node_id": node_info["node_id"]
    }

@app.get("/")
def get_node_status():
    """
    Basic node status endpoint
    """
    return {
        "status": "online",
        "connected": node_info["connected"],
        "node": node_info
    }

@app.get("/usage")
def get_usage():
    """
    Get current system usage
    """
    try:
        cpu_usage = psutil.cpu_percent(interval=1)
        
        gpus = GPUtil.getGPUs()
        gpu_usage = round(gpus[0].load * 100, 2) if gpus else 0

        return {
            "cpu_usage": cpu_usage,
            "gpu_usage": gpu_usage,
            "memory_usage": psutil.virtual_memory().percent
        }
    except Exception as e:
        logger.error(f"Usage retrieval error: {e}")
        return {"error": "Failed to retrieve usage information"}

@app.get("/node-capabilities")
def get_detailed_capabilities():
    """
    Detailed node capabilities endpoint
    """
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
def compute(task: Dict[str, Any]):
    """
    Computation handler with GPU and CPU task processing
    """
    try:
        gpu_info = node_info["capabilities"]["gpu"]
        
        # Increment total tasks processed
        node_info["total_tasks_processed"] += 1
        
        # GPU computation if available
        if gpu_info.get("name", "No GPU") != "No GPU":
            result = _process_gpu_task(task)
            return {
                "task_id": task.get("task_id", str(uuid.uuid4())),
                "result": result,
                "gpu_used": gpu_info["name"],
                "gpu_load": gpu_info.get("load_percentage", 0),
                "processing_method": "GPU"
            }
        
        # Fallback to CPU computation
        result = _process_cpu_task(task)
        return {
            "task_id": task.get("task_id", str(uuid.uuid4())),
            "result": result,
            "processing_method": "CPU"
        }
    
    except Exception as e:
        logger.error(f"Computation error: {e}")
        return {
            "error": "Computation failed",
            "details": str(e)
        }

def _process_gpu_task(task):
    # Simulated GPU task processing with error handling
    try:
        return str(task).upper()
    except Exception as e:
        logger.error(f"GPU task processing error: {e}")
        return f"GPU_ERROR: {e}"

def _process_cpu_task(task):
    # Simulated CPU task processing with error handling
    try:
        return str(task).lower()
    except Exception as e:
        logger.error(f"CPU task processing error: {e}")
        return f"CPU_ERROR: {e}"

if __name__ == "__main__":
    import uvicorn
    # Use environment variable for port, default to 9100
    port = int(os.getenv('PORT', 9100))
    uvicorn.run(app, host="0.0.0.0", port=port)
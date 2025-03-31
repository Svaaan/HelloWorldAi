import os
import socket
import psutil
import platform
import requests
import uuid
import logging
from fastapi import FastAPI, Request, BackgroundTasks
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
        # Use Docker container name if defined, else fallback to host IP
        return os.getenv('NODE_HOSTNAME', socket.gethostname())
    except Exception as e:
        logger.warning(f"IP retrieval error: {e}")
        return "localhost"

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

        gpus = []
        try:
            detected_gpus = GPUtil.getGPUs()
            for gpu in detected_gpus:
                gpus.append({
                    "name": gpu.name,
                    "total_memory": gpu.memoryTotal,
                    "free_memory": gpu.memoryFree,
                    "used_memory": gpu.memoryUsed,
                    "load_percentage": round(gpu.load * 100, 2),
                    "temperature": gpu.temperature
                })
        except Exception as gpu_error:
            logger.error(f"GPU detection error: {gpu_error}")
            gpus = [{"name": "GPU Detection Limited"}]

        return {
            "cpu": cpu,
            "gpu": gpus if gpus else [{"name": "No GPU Detected"}]
        }

    except Exception as e:
        logger.error(f"System capabilities error: {e}")
        return {"error": "Limited system capabilities"}
    
def get_advanced_gpu_capabilities() -> list:
    try:
        gpus = GPUtil.getGPUs()
        gpu_list = []
        for gpu in gpus:
            gpu_list.append({
                "name": gpu.name,
                "total_memory": gpu.memoryTotal,
                "free_memory": gpu.memoryFree,
                "used_memory": gpu.memoryUsed,
                "load_percentage": round(gpu.load * 100, 2),
                "temperature": gpu.temperature
            })
        return gpu_list if gpu_list else [{"name": "No GPU Detected"}]
    except Exception as e:
        logger.error(f"GPU detection error: {e}")
        return [{"name": "GPU Detection Limited"}]


# Node configuration with environment-aware initialization
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

def background_connection_handler(payload: Dict[str, Any]):
    """
    Background task for handling node connection
    """
    coordinator_url = os.getenv('COORDINATOR_URL', 'http://localhost:8100/connect-node')
    
    logger.info(f"📡 Attempting to connect to coordinator at {coordinator_url}")
    logger.debug(f"🔍 Payload being sent:\n{payload}")

    try:
        res = requests.post(coordinator_url, json=payload, timeout=10)

        logger.info(f"🔄 Coordinator responded with status {res.status_code}")
        if res.status_code == 200:
            node_info["connected"] = True
            logger.info(f"✅ Node '{node_info['node_id']}' connected successfully!")
            logger.debug(f"📥 Coordinator response: {res.json()}")
        else:
            logger.error(f"❌ Connection failed. Status: {res.status_code}, Response: {res.text}")

    except requests.exceptions.RequestException as e:
        logger.error(f"🚨 Connection error to coordinator: {e}")


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
def compute(task: Dict[str, Any], request: Request):
    """
    Computation handler with GPU and CPU task processing
    Includes task filtering, access control, and quota enforcement.
    """
    try:
        client_ip = request.client.host
        client_id = task.get("client_id")
        task_type = task.get("type")

        # Optional owner-defined policies
        accept_tasks = node_info.get("accept_tasks", True)
        allowed_clients = node_info.get("allowed_clients", [])
        accepted_task_types = node_info.get("accepted_task_types", [])
        max_tasks = node_info.get("max_tasks", 100)

        if not accept_tasks:
            return {"error": "This node is not currently accepting tasks."}

        if allowed_clients and client_id not in allowed_clients:
            return {"error": f"Client '{client_id}' is not allowed to run tasks on this node."}

        if accepted_task_types and task_type not in accepted_task_types:
            return {"error": f"Task type '{task_type}' is not accepted by this node."}

        if node_info["total_tasks_processed"] >= max_tasks:
            return {"error": "Node has reached its maximum task limit."}

        # Log the incoming request
        logger.info(f"🧠 Task received from {client_id or client_ip} — Type: {task_type}")

        gpu_list = node_info["capabilities"].get("gpu", [])
        gpu_available = next((gpu for gpu in gpu_list if gpu.get("name") != "No GPU Detected"), None)

        # Increment processed task count
        node_info["total_tasks_processed"] += 1

        # Prefer GPU if available
        if gpu_available:
            result = _process_gpu_task(task)
            return {
                "task_id": task.get("task_id", str(uuid.uuid4())),
                "result": result,
                "gpu_used": gpu_available["name"],
                "gpu_load": gpu_available.get("load_percentage", 0),
                "processing_method": "GPU"
            }

        # Fallback to CPU
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
    
def is_node_overloaded(cpu_threshold=90.0, gpu_threshold=90.0, memory_threshold=90.0) -> bool:
    try:
        cpu = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory().percent
        gpus = GPUtil.getGPUs()
        gpu = max((gpu.load * 100 for gpu in gpus), default=0.0)

        return cpu > cpu_threshold or memory > memory_threshold or gpu > gpu_threshold
    except Exception as e:
        logger.warning(f"⚠️ Failed to check node load: {e}")
        return False  # Fail-safe: allow task if we can't check


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
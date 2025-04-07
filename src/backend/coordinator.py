import uuid
import time
import threading
import numpy as np
from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import psutil
from pydantic import BaseModel, Field
from typing import Dict, Optional
from backend.mocked_task import verify_cpu_connected, verify_gpu_connected
import pynvml

app = FastAPI()
task_results = []

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GPUCapabilities(BaseModel):
    name: str = "No GPU"
    total_memory: Optional[int] = None
    free_memory: Optional[int] = None
    used_memory: Optional[int] = None
    load_percentage: Optional[float] = None
    temperature: Optional[float] = None

class CPUCapabilities(BaseModel):
    brand: str
    cores: int
    threads: int
    max_freq: Optional[float] = None
    min_freq: Optional[float] = None
    current_freq: Optional[float] = None

class NodeConnection(BaseModel):
    node_id: str = Field(default_factory=lambda: f"node_{uuid.uuid4()}")
    ip: str
    country: Optional[str] = "Unknown"
    capabilities: Dict = {
        "cpu": {},
        "gpu": {}
    }
    isConnected: bool = False
    isAvailable: bool = False
    cpu_verified: bool = False
    gpu_verified: bool = False
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    cpu_benchmark: Optional[int] = None
    gpu_benchmark: Optional[int] = None

connected_nodes: Dict[str, NodeConnection] = {}

system_usage = {
    "cpu_usage": 0.0,
    "gpu_usage": 0.0,
    "last_updated": time.time()
}

def get_gpu_info():
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

        pynvml.nvmlShutdown()
        return gpu_info

    except pynvml.NVMLError as e:
        print("⚠️ NVML error:", e)
        return []

@app.patch("/toggle-availability/{node_id}")
def toggle_availability(node_id: str):
    result = toggle_node_availability(node_id, connected_nodes)
    if result is None:
        return {"status": "error", "message": f"Node {node_id} not found"}, 404

    return {
        "status": "success",
        "node_id": node_id,
        "isAvailable": result,
        "message": f"Node availability toggled to {result}"
    }

@app.post("/connect-node")
def connect_node(node: NodeConnection, request: Request):
    gpu_capabilities = node.capabilities.get("gpu", [])
    if not gpu_capabilities or gpu_capabilities[0].get("name") in ["No GPU Detected", None, ""]:
        return {
            "status": "rejected",
            "reason": "No valid GPU detected. Node connection refused."
        }

    node.ip = request.client.host
    node.isConnected = True
    node.cpu_verified = False
    node.gpu_verified = False
    node.cpu_usage = 0.0
    node.gpu_usage = 0.0

    node.capabilities["gpu"] = gpu_capabilities

    connected_nodes[node.node_id] = node

    print(f"🔌 Node connected: {node.node_id}, Available: {node.isAvailable}")

    return {
        "status": "Node connected",
        "node_id": node.node_id,
        "ip": node.ip,
    }

@app.get("/get-task-results")
def get_task_results():
    return task_results

@app.post("/receive-task-result")
async def receive_task_result(result: dict):
    print(f"📥 Task result received: {result}")
    # Ensure logs are part of the result
    if 'logs' not in result:
        print("⚠️ No logs found in task result!")
    else:
        print(f"📝 Logs: {result['logs']}")
    task_results.append(result)  # Store the result including logs
    return {"status": "success", "message": "Result received"}

@app.get("/nodes")
def get_connected_nodes(request: Request, node_id: Optional[str] = None):
    # Remove IP filtering, show all nodes
    filtered_nodes = [
        {
            "node_id": node.node_id,
            "ip": node.ip,
            "country": node.country,
            "capabilities": node.capabilities,
            "cpu_verified": node.cpu_verified,
            "gpu_verified": node.gpu_verified,
            "cpu_usage": node.cpu_usage,
            "gpu_usage": node.gpu_usage,
            "isConnected": node.isConnected,
            "isAvailable": node.isAvailable
        }
        for node in connected_nodes.values()
    ]

    # ✅ If node_id query param is provided, filter further
    if node_id:
        filtered_nodes = [node for node in filtered_nodes if node["node_id"] == node_id]

    return filtered_nodes




@app.get("/available-nodes")
def get_available_nodes():
    return [
        {
            "node_id": node.node_id,
            "ip": node.ip,
            "country": node.country,
            "capabilities": node.capabilities,
            "cpu_verified": node.cpu_verified,
            "gpu_verified": node.gpu_verified,
            "cpu_usage": node.cpu_usage,
            "gpu_usage": node.gpu_usage,
            "isConnected": node.isConnected,
            "isAvailable": node.isAvailable
        }
        for node in connected_nodes.values()
        if node.isConnected and node.isAvailable
    ]


def toggle_node_availability(node_id: str, connected_nodes: dict):
    if node_id in connected_nodes:
        node = connected_nodes[node_id]
        node.isAvailable = not node.isAvailable
        print(f"🔁 Toggled availability for {node_id} to {node.isAvailable}")
        return node.isAvailable
    return None


@app.get("/get-connected-nodes-count")
def get_connected_nodes_count():
    return {"connected_nodes_count": len([n for n in connected_nodes.values() if n.isConnected])}

@app.get("/usage")
def get_usage_info():
    for node in connected_nodes.values():
        if node.isConnected:
            try:
                node.cpu_usage = psutil.cpu_percent(interval=0.1)
            except Exception:
                node.cpu_usage = 0.0

    connected = [n for n in connected_nodes.values() if n.isConnected]
    if connected:
        avg_cpu_usage = sum(n.cpu_usage for n in connected) / len(connected)
        avg_gpu_usage = sum(n.gpu_usage for n in connected) / len(connected)
    else:
        avg_cpu_usage = 0
        avg_gpu_usage = 0

    system_usage["cpu_usage"] = avg_cpu_usage
    system_usage["gpu_usage"] = avg_gpu_usage
    system_usage["last_updated"] = time.time()

    return {
        "cpu_usage": round(avg_cpu_usage, 1),
        "gpu_usage": round(avg_gpu_usage, 1),
        "last_updated": system_usage["last_updated"]
    }

@app.post("/verify-node/{node_id}/cpu")
def verify_node_cpu(node_id: str):
    if node_id not in connected_nodes:
        return {"status": "error", "message": f"Node {node_id} not found"}
    if not connected_nodes[node_id].isConnected:
        return {"status": "error", "message": f"Node {node_id} is not connected"}

    thread = threading.Thread(target=verify_cpu_connected, args=(node_id, connected_nodes))
    thread.daemon = True
    thread.start()

    return {"status": "success", "message": f"CPU verification started for node {node_id}"}

@app.post("/verify-node/{node_id}/gpu")
def verify_node_gpu(node_id: str):
    if node_id not in connected_nodes:
        return {"status": "error", "message": f"Node {node_id} not found"}
    if not connected_nodes[node_id].isConnected:
        return {"status": "error", "message": f"Node {node_id} is not connected"}

    thread = threading.Thread(target=verify_gpu_connected, args=(node_id, connected_nodes, pynvml))
    thread.daemon = True
    thread.start()

    return {"status": "success", "message": f"GPU verification started for node {node_id}"}

@app.get("/node-performance/{node_id}")
def get_node_performance(node_id: str):
    if node_id not in connected_nodes:
        return {"status": "error", "message": f"Node {node_id} not found"}

    node = connected_nodes[node_id]

    return {
        "status": "success",
        "node_id": node_id,
        "cpu_verified": node.cpu_verified,
        "gpu_verified": node.gpu_verified,
        "cpu_usage": node.cpu_usage,
        "gpu_usage": node.gpu_usage,
        "cpu_benchmark": node.cpu_benchmark,
        "gpu_benchmark": node.gpu_benchmark
    }

@app.post("/receive-task-result")
async def receive_task_result(result: dict):
    print(f"📥 Task result received: {result}")
    # Optionally store it, or notify frontend via WebSocket, etc.
    return {"status": "success", "message": "Result received"}


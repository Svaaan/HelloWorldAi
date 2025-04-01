import uuid
import time
import threading
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import psutil
from GPUtil import getGPUs
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from backend.mocked_task import (
    verify_cpu_connected,
    verify_gpu_connected
)
import torch

app = FastAPI()

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
    load_percentage: Optional[float] = None
    temperature: Optional[float] = None
    cuda_cores: Optional[int] = None
    compute_capability: Optional[str] = None

class CPUCapabilities(BaseModel):
    brand: str
    cores: int
    threads: int
    max_freq: Optional[float] = None
    min_freq: Optional[float] = None

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
    total_compute_score: float = 0
    cpu_verified: bool = False
    gpu_verified: bool = False
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    cpu_benchmark: Optional[int] = None
    gpu_benchmark: Optional[int] = None

# In-memory storage for connected nodes
connected_nodes: Dict[str, NodeConnection] = {}

# Track system-wide usage
system_usage = {
    "cpu_usage": 0.0,
    "gpu_usage": 0.0,
    "last_updated": time.time()
}


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
    node.ip = request.client.host
    node.isConnected = True  # Setting connection status to true
    node.cpu_verified = False
    node.gpu_verified = False
    node.cpu_usage = 0.0
    node.gpu_usage = 0.0

    connected_nodes[node.node_id] = node

    return {
        "status": "Node connected",
        "node_id": node.node_id,
        "ip": node.ip,
        "compute_score": node.total_compute_score
    }


@app.get("/nodes")
def get_connected_nodes():
    return [
        {
            "node_id": node.node_id,
            "ip": node.ip,
            "country": getattr(node, "country", "Unknown"),
            "capabilities": node.capabilities,
            "compute_score": node.total_compute_score,
            "cpu_verified": node.cpu_verified,
            "gpu_verified": node.gpu_verified,
            "cpu_usage": node.cpu_usage,
            "gpu_usage": node.gpu_usage,
            "isConnected": node.isConnected,
            "isAvailable": node.isAvailable

        }
        for node in connected_nodes.values()
    ]

def toggle_node_availability(node_id: str, connected_nodes: dict):
    if node_id in connected_nodes:
        node = connected_nodes[node_id]
        node.isAvailable = not getattr(node, 'isAvailable', False)
        print(f"🔁 Toggled availability for {node_id} to {node.isAvailable}")
        return node.isAvailable
    return None

@app.get("/get-connected-nodes-count")
def get_connected_nodes_count():
    return {"connected_nodes_count": len([n for n in connected_nodes.values() if n.isConnected])}


@app.get("/usage")
def get_usage_info():


    # Refresh live usage values for each connected node
    for node in connected_nodes.values():
        if node.isConnected:
            try:
                node.cpu_usage = psutil.cpu_percent(interval=0.1)
            except Exception:
                node.cpu_usage = 0.0

            try:
                gpus = getGPUs()
                if gpus:
                    avg_load = sum(gpu.load for gpu in gpus) / len(gpus)
                    node.gpu_usage = round(avg_load * 100, 2)
                else:
                    node.gpu_usage = 0.0
            except Exception:
                node.gpu_usage = 0.0

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

# Add endpoints for specific test types
@app.post("/verify-node/{node_id}/cpu")
def verify_node_cpu(node_id: str):
    if node_id not in connected_nodes:
        return {"status": "error", "message": f"Node {node_id} not found"}
    
    if not connected_nodes[node_id].isConnected:
        return {"status": "error", "message": f"Node {node_id} is not connected"}
    
    # Start CPU verification in background
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
    
    # Start GPU verification in background
    thread = threading.Thread(target=verify_gpu_connected, args=(node_id, connected_nodes, torch))
    thread.daemon = True
    thread.start()
    
    return {"status": "success", "message": f"GPU verification started for node {node_id}"}

# Add endpoint to get test results
@app.get("/node-performance/{node_id}")
def get_node_performance(node_id: str):
    if node_id not in connected_nodes:
        return {"status": "error", "message": f"Node {node_id} not found"}
    
    node = connected_nodes[node_id]
    
    return {
        "status": "success",
        "node_id": node_id,
        "cpu_verified": getattr(node, 'cpu_verified', False),
        "gpu_verified": getattr(node, 'gpu_verified', False),
        "cpu_usage": getattr(node, 'cpu_usage', 0),
        "gpu_usage": getattr(node, 'gpu_usage', 0),
        "cpu_benchmark": getattr(node, "cpu_benchmark", None),
        "gpu_benchmark": getattr(node, "gpu_benchmark", None),

    }

    
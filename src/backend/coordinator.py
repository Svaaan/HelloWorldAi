import os
import socket
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

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
    port: str
    capabilities: Dict = {
        "cpu": {},
        "gpu": {}
    }
    isConnected: bool = False
    last_heartbeat: Optional[float] = None
    total_compute_score: float = 0

# In-memory storage for connected nodes
connected_nodes: Dict[str, NodeConnection] = {}

def calculate_compute_score(node: NodeConnection) -> float:
    """
    Calculate a comprehensive compute score for a node
    """
    cpu_score = 0
    gpu_score = 0

    # CPU Score Calculation
    cpu = node.capabilities.get('cpu', {})
    cpu_score = (cpu.get('cores', 0) * 10) + (cpu.get('threads', 0) * 5)
    
    # GPU Score Calculation
    gpu = node.capabilities.get('gpu', {})
    if gpu and gpu.get('name', '') != "No GPU":
        gpu_score = (gpu.get('total_memory', 0) / 1024) * 2  # Memory in GB
        gpu_score += (gpu.get('cuda_cores', 0) / 1000) * 3
    
    return cpu_score + gpu_score

@app.post("/connect-node")
def connect_node(node: NodeConnection):
    # Calculate compute score
    node.total_compute_score = calculate_compute_score(node)
    node.isConnected = True

    # Store or update node
    connected_nodes[node.node_id] = node
    
    return {
        "status": "Node connected successfully", 
        "node_id": node.node_id,
        "compute_score": node.total_compute_score
    }

@app.get("/nodes")
def get_connected_nodes():
    return [
        {
            "node_id": node.node_id,
            "ip": node.ip,
            "port": node.port,
            "capabilities": node.capabilities,
            "compute_score": node.total_compute_score
        } 
        for node in connected_nodes.values() 
        if node.isConnected
    ]

@app.get("/get-connected-nodes-count")
def get_connected_nodes_count():
    return {"connected_nodes_count": len(connected_nodes)}

@app.get("/get-total-power")
def get_total_power():
    total_compute_score = sum(
        node.total_compute_score 
        for node in connected_nodes.values() 
        if node.isConnected
    )

    # Detailed breakdown
    node_details = [
        {
            "node_id": node.node_id,
            "cpu": node.capabilities.get('cpu', {}),
            "gpu": node.capabilities.get('gpu', {}),
            "compute_score": node.total_compute_score
        }
        for node in connected_nodes.values() 
        if node.isConnected
    ]

    return {
        "total_compute_score": total_compute_score,
        "node_details": node_details,
        "connected_nodes": len(connected_nodes)
    }

@app.get("/best-nodes-for-task")
def get_best_nodes_for_task(task_type: str = "general"):
    """
    Recommend best nodes for a specific task type
    """
    # Sort nodes by compute score in descending order
    sorted_nodes = sorted(
        [node for node in connected_nodes.values() if node.isConnected],
        key=lambda x: x.total_compute_score, 
        reverse=True
    )

    # Return top 3 nodes
    return [
        {
            "node_id": node.node_id,
            "ip": node.ip,
            "port": node.port,
            "compute_score": node.total_compute_score
        }
        for node in sorted_nodes[:3]
    ]
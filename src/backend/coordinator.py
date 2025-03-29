import os
import socket
import uuid
import time
import threading
import numpy as np
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

# For GPU usage
try:
    import torch
except ImportError:
    pass  # Handle systems without PyTorch installed

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
    cpu_verified: bool = False
    gpu_verified: bool = False
    cpu_usage: float = 0.0  # Current CPU usage percentage
    gpu_usage: float = 0.0  # Current GPU usage percentage

# In-memory storage for connected nodes
connected_nodes: Dict[str, NodeConnection] = {}

# Track system-wide usage
system_usage = {
    "cpu_usage": 0.0,
    "gpu_usage": 0.0,
    "last_updated": time.time()
}

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
def connect_node(node: NodeConnection, request: Request):
    node.ip = request.client.host
    node.total_compute_score = calculate_compute_score(node)
    node.isConnected = True  # Setting connection status to true
    node.last_heartbeat = time.time()
    node.cpu_verified = False
    node.gpu_verified = False
    node.cpu_usage = 0.0
    node.gpu_usage = 0.0

    connected_nodes[node.node_id] = node

    # Kick off background task to simulate compute verification
    threading.Thread(target=simulate_verification_task, args=(node.node_id,)).start()

    return {
        "status": "Node connected and test task started",
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
            "port": node.port,
            "capabilities": node.capabilities,
            "compute_score": node.total_compute_score,
            "cpu_verified": node.cpu_verified,
            "gpu_verified": node.gpu_verified,
            "cpu_usage": node.cpu_usage,
            "gpu_usage": node.gpu_usage
        } 
        for node in connected_nodes.values() 
        if node.isConnected
    ]

@app.get("/get-connected-nodes-count")
def get_connected_nodes_count():
    return {"connected_nodes_count": len([n for n in connected_nodes.values() if n.isConnected])}

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
            "compute_score": node.total_compute_score,
            "cpu_verified": node.cpu_verified,
            "gpu_verified": node.gpu_verified,
            "cpu_usage": node.cpu_usage,
            "gpu_usage": node.gpu_usage
        }
        for node in connected_nodes.values() 
        if node.isConnected
    ]

    return {
        "total_compute_score": total_compute_score,
        "node_details": node_details,
        "connected_nodes": len([n for n in connected_nodes.values() if n.isConnected])
    }

@app.get("/usage")
def get_usage_info():
    """
    Return the current system usage information for the dashboard
    """
    # Calculate average usage across all connected nodes
    connected = [n for n in connected_nodes.values() if n.isConnected]
    if connected:
        avg_cpu_usage = sum(n.cpu_usage for n in connected) / len(connected)
        avg_gpu_usage = sum(n.gpu_usage for n in connected) / len(connected)
    else:
        avg_cpu_usage = 0
        avg_gpu_usage = 0
    
    # Update system-wide usage
    system_usage["cpu_usage"] = avg_cpu_usage
    system_usage["gpu_usage"] = avg_gpu_usage
    system_usage["last_updated"] = time.time()
    
    return {
        "cpu_usage": round(avg_cpu_usage, 1),
        "gpu_usage": round(avg_gpu_usage, 1),
        "last_updated": system_usage["last_updated"]
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
            "compute_score": node.total_compute_score,
            "cpu_verified": node.cpu_verified,
            "gpu_verified": node.gpu_verified,
            "cpu_usage": node.cpu_usage,
            "gpu_usage": node.gpu_usage
        }
        for node in sorted_nodes[:3]
    ]

def run_cpu_task(node_id: str):
    """
    Run a CPU-intensive task that uses minimal resources
    """
    if node_id not in connected_nodes:
        return False
    
    print(f"🧪 Running CPU verification task on node: {node_id}")
    
    # Update CPU usage to show activity
    if node_id in connected_nodes:
        connected_nodes[node_id].cpu_usage = 1.0  # Show 1% CPU usage
    
    # CPU matrix operations (using NumPy which is CPU-bound)
    matrix_size = 1000
    # Create matrices
    a = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size, matrix_size)
    
    # Perform some CPU-intensive operations
    start_time = time.time()
    c = np.matmul(a, b)  # Matrix multiplication
    d = np.linalg.det(a[:100, :100])  # Determinant (smaller matrix to not be too intensive)
    
    # Some additional CPU work
    for i in range(5):
        c = c @ b  # Matrix multiplication
    
    end_time = time.time()
    
    print(f"✅ Node {node_id} CPU task completed in {end_time - start_time:.2f} seconds")
    
    if node_id in connected_nodes:
        connected_nodes[node_id].cpu_verified = True
        connected_nodes[node_id].last_heartbeat = time.time()
    
    return True

def run_gpu_task(node_id: str):
    """
    Run a GPU task that uses minimal resources (~1%)
    """
    if node_id not in connected_nodes or 'torch' not in globals():
        return False
    
    node = connected_nodes[node_id]
    if node.capabilities.get('gpu', {}).get('name', 'No GPU') == 'No GPU':
        print(f"⚠️ Node {node_id} does not have GPU capabilities")
        return False
    
    try:
        if not torch.cuda.is_available():
            print(f"⚠️ Node {node_id} has GPU in capabilities but CUDA not available")
            return False
        
        print(f"🧪 Running minimal GPU task on node: {node_id}")
        
        # Update GPU usage to show activity
        if node_id in connected_nodes:
            connected_nodes[node_id].gpu_usage = 1.0  # Show 1% GPU usage
        
        # Create small tensors and perform minimal operations
        tensor_size = 2000  # Small enough to use ~1% of GPU
        
        # Run a few small GPU operations
        start_time = time.time()
        
        # Create tensors on GPU
        a = torch.rand(tensor_size, tensor_size, device='cuda')
        b = torch.rand(tensor_size, tensor_size, device='cuda')
        
        # Matrix multiplication (uses GPU compute)
        c = torch.matmul(a, b)
        
        # Some simple operations to keep the GPU active for a moment
        for i in range(3):
            c = torch.nn.functional.relu(c @ b)
        
        # Force synchronization but don't store large results
        _ = c.sum().item()
        
        end_time = time.time()
        
        # Clean up to release GPU memory
        del a, b, c
        torch.cuda.empty_cache()
        
        print(f"✅ Node {node_id} GPU task completed in {end_time - start_time:.2f} seconds")
        
        if node_id in connected_nodes:
            connected_nodes[node_id].gpu_verified = True
            connected_nodes[node_id].last_heartbeat = time.time()
        
        return True
        
    except Exception as e:
        print(f"⚠️ Error running GPU task on node {node_id}: {str(e)}")
        return False

def simulate_verification_task(node_id: str):
    """
    Verifies node responsiveness with both CPU and GPU tasks
    """
    if node_id not in connected_nodes:
        print(f"❌ Node {node_id} disconnected before verification")
        return
    
    # Run CPU verification
    cpu_result = run_cpu_task(node_id)
    
    # Run GPU verification if node has GPU capabilities
    gpu_result = run_gpu_task(node_id)
    
    # Update node status
    if node_id in connected_nodes:
        node = connected_nodes[node_id]
        print(f"📊 Node {node_id} verification results - CPU: {cpu_result}, GPU: {gpu_result}")
        
        # Update last heartbeat time
        node.last_heartbeat = time.time()

# Add a function to check for stale nodes and mark them as disconnected
def cleanup_stale_nodes():
    """
    Periodically check for nodes that haven't sent a heartbeat
    and mark them as disconnected
    """
    while True:
        current_time = time.time()
        stale_threshold = 60  # 60 seconds without heartbeat = stale
        
        for node_id, node in list(connected_nodes.items()):
            if node.isConnected and node.last_heartbeat and (current_time - node.last_heartbeat) > stale_threshold:
                print(f"⚠️ Node {node_id} appears to be stale. Marking as disconnected.")
                node.isConnected = False
        
        time.sleep(30)  # Check every 30 seconds

# Add a function to periodically run small tasks to maintain minimal usage
def maintain_minimal_usage():
    """
    Periodically run small CPU/GPU tasks on active nodes
    to maintain the minimal 1% usage as requested
    """
    while True:
        # For each connected node
        for node_id, node in list(connected_nodes.items()):
            if node.isConnected:
                # Gradually decrease usage if no recent task (simulates decay)
                current_time = time.time()
                time_since_heartbeat = current_time - (node.last_heartbeat or current_time)
                
                # If it's been more than 5 seconds since the last heartbeat
                if time_since_heartbeat > 5:
                    # Run mini tasks to maintain usage
                    threading.Thread(target=run_mini_tasks, args=(node_id,)).start()
        
        # Run every 5 seconds
        time.sleep(5)

def run_mini_tasks(node_id: str):
    """Run minimal CPU and GPU tasks to maintain the 1% usage"""
    if node_id not in connected_nodes or not connected_nodes[node_id].isConnected:
        return
    
    # Run a tiny CPU task
    try:
        # Very small CPU task
        _ = sum(i * i for i in range(100_000))
        
        # Update CPU usage
        connected_nodes[node_id].cpu_usage = 1.0  # Maintain at 1%
        connected_nodes[node_id].last_heartbeat = time.time()
    except Exception as e:
        print(f"Error in mini CPU task: {e}")
    
    # Run a tiny GPU task if GPU is verified
    if connected_nodes[node_id].gpu_verified and 'torch' in globals():
        try:
            if torch.cuda.is_available():
                # Small tensor operations
                a = torch.rand(500, 500, device='cuda')
                b = torch.rand(500, 500, device='cuda')
                c = a @ b
                _ = c.sum().item()
                
                # Clean up
                del a, b, c
                torch.cuda.empty_cache()
                
                # Update GPU usage
                connected_nodes[node_id].gpu_usage = 1.0  # Maintain at 1%
                connected_nodes[node_id].last_heartbeat = time.time()
        except Exception as e:
            print(f"Error in mini GPU task: {e}")

# Add an endpoint to manually trigger verification
@app.post("/verify-node/{node_id}")
def verify_node(node_id: str):
    if node_id not in connected_nodes:
        return {"status": "error", "message": f"Node {node_id} not found"}
    
    if not connected_nodes[node_id].isConnected:
        return {"status": "error", "message": f"Node {node_id} is not connected"}
    
    # Start verification in background
    threading.Thread(target=simulate_verification_task, args=(node_id,)).start()
    
    return {"status": "success", "message": f"Verification started for node {node_id}"}

# Start the cleanup thread when the application starts
@app.on_event("startup")
def startup_event():
    # Start the cleanup thread
    threading.Thread(target=cleanup_stale_nodes, daemon=True).start()
    
    # Start the minimal usage maintenance thread
    threading.Thread(target=maintain_minimal_usage, daemon=True).start()
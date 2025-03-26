import os
import socket
import psutil
import platform
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import requests
import GPUtil
from backend.shared.types import ComputationTask

app = FastAPI()

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Function to get the local IP address
def get_local_ip():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        print(f"Error getting IP address: {e}")
        return "127.0.0.1"  # Default to localhost if there is an error

# Function to get CPU and GPU information dynamically
def get_system_capabilities():
    # Get CPU info dynamically
    cpu_info = psutil.cpu_freq()
    cpu = {
        "brand": platform.processor(),  # Get the processor name from the system
        "cores": psutil.cpu_count(logical=False),  # Physical cores
        "threads": psutil.cpu_count(logical=True),  # Logical cores (threads)
        "max_freq": cpu_info.max if cpu_info else "Unknown",
        "min_freq": cpu_info.min if cpu_info else "Unknown"
    }

    # Get GPU information dynamically
    try:
        gpus = GPUtil.getGPUs()
        gpu = gpus[0].name if gpus else "No GPU found"
    except Exception as e:
        gpu = "No GPU available"

    return {"cpu": cpu, "gpu": gpu}

# Session-based node information
node_info = {
    "node_id": f"node_{os.getpid()}",  # Generate a unique node ID based on process ID
    "ip": get_local_ip(),  # Get the actual local IP address
    "port": "9100",  # This can be changed dynamically as needed
    "capabilities": get_system_capabilities(),  # Get the system's capabilities dynamically
    "connected": False  # Set initial connection status to False
}

@app.get("/")
def get_node_status():
    # Dynamically check if the node is connected
    is_connected = node_info["connected"]
    return {
        "status": "online",
        "connected": is_connected,
        "node": node_info
    }

@app.post("/connect-node")
async def connect_node():
    if node_info["connected"]:
        return {"status": "Node is already connected.", "connected": node_info["connected"]}

    payload = {
        "node_id": node_info["node_id"],
        "ip": node_info["ip"],
        "port": node_info["port"],
        "capabilities": node_info["capabilities"]
    }

    try:
        # Connect the node with the coordinator at 'http://127.0.0.1:8100/connect-node'
        res = requests.post("http://127.0.0.1:8100/connect-node", json=payload)

        if res.status_code == 200:
            node_info["connected"] = True  # Mark the node as connected after successful connection
            return {"status": "Node connected successfully!", "connected": node_info["connected"]}
        else:
            error_message = res.text if res.text else "Unknown error occurred."
            raise HTTPException(status_code=400, detail=f"Connection failed: {error_message}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to coordinator: {str(e)}")


@app.get("/usage")
def get_usage():
    # Get CPU usage (percentage)
    cpu_usage = psutil.cpu_percent(interval=1)  # Get CPU usage over 1 second
    # Get GPU usage (percentage)
    gpus = GPUtil.getGPUs()
    gpu_usage = gpus[0].load * 100 if gpus else 0  # Assuming one GPU for simplicity
    
    return {
        "cpu_usage": cpu_usage,
        "gpu_usage": gpu_usage
    }

@app.post("/compute")
def compute(task: ComputationTask):
    result = task.data.get("message", "").upper()  # Process the task (for example, convert message to uppercase)
    return {"task_id": task.task_id, "result": result}

@app.post("/receive-data")
async def receive_data(request: Request):
    data = await request.json()  # Parse the incoming JSON data
    print("Got data:", data)
    return {"status": "received"}

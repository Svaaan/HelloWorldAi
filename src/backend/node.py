import os
import socket
import psutil
import platform
from fastapi import FastAPI, Request
import requests
from backend.shared.types import ComputationTask

app = FastAPI()

# Function to get the local IP address
def get_local_ip():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except Exception as e:
        print(f"Error getting IP address: {e}")
        return "127.0.0.1"  # Default to localhost if there is an error

# Function to get CPU and GPU information
def get_system_capabilities():
    cpu = psutil.cpu_info().brand if hasattr(psutil, 'cpu_info') else "Unknown CPU"
    gpu = "Nvidia RTX 3060"  # This can be improved further by using GPU libraries, if needed.
    return {"cpu": cpu, "gpu": gpu}

# Dynamically set node information
node_info = {
    "node_id": f"node_{os.getpid()}",  # Unique node ID based on process ID
    "ip": get_local_ip(),  # Get the actual local IP address
    "port": "9100",  # This can be changed dynamically as needed
    "capabilities": get_system_capabilities(),  # Get the system's capabilities
    "registered": False  # Track the registration status of the node
}

@app.get("/")
def get_node_status():
    return {
        "status": "online",
        "registered": node_info["registered"],
        "node": node_info
    }

@app.post("/register-node")
async def register_node():
    if node_info["registered"]:
        return {"status": "Node already registered."}

    payload = {
        "node_id": node_info["node_id"],
        "ip": node_info["ip"],
        "port": node_info["port"],
        "capabilities": node_info["capabilities"]
    }

    try:
        res = requests.post("http://127.0.0.1:8100/register-node", json=payload)

        if res.status_code == 200:
            node_info["registered"] = True
            return {"status": "Node registered successfully!"}
        else:
            return {"status": "Registration failed.", "error": res.text}
    except Exception as e:
        return {"status": "Error", "message": str(e)}

@app.post("/compute")
def compute(task: ComputationTask):
    result = task.data.get("message", "").upper()  # Process the task (for example, convert message to uppercase)
    return {"task_id": task.task_id, "result": result}

@app.post("/receive-data")
async def receive_data(request: Request):
    data = await request.json()  # Parse the incoming JSON data
    print("Got data:", data)
    return {"status": "received"}

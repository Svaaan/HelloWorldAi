from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define node connection models
class NodeConnection(BaseModel):
    node_id: str
    ip: str
    port: str
    capabilities: dict
    isConnected: bool = False  # Track the connection status of the node

# In-memory storage for connected nodes
connected_nodes = {}

@app.post("/connect-node")
def connect_node(node: NodeConnection):
    if node.node_id in connected_nodes:
        connected_nodes[node.node_id].isConnected = True  # Update connection status
        return {"status": "Node already connected", "node": connected_nodes[node.node_id]}

    node.isConnected = True
    connected_nodes[node.node_id] = node
    return {"status": "Node connected successfully", "node": node.dict()}

@app.get("/nodes")  # This is the route you're fetching from the frontend
def get_connected_nodes():
    return [node.dict() for node in connected_nodes.values() if node.isConnected]

@app.get("/get-connected-nodes-count")
def get_connected_nodes_count():
    connected_nodes_count = len([node for node in connected_nodes.values() if node.isConnected])
    return {"connected_nodes_count": connected_nodes_count}


@app.get("/get-total-power")
def get_total_power():
    total_cpu_power = 0
    total_gpu_power = 0
    cpu_weight = 1  # Weight of 1 per CPU core
    gpu_weight = 5  # Weight of 5 for each GPU (or adjust as needed)

    for node in connected_nodes.values():
        if node.isConnected:
            total_cpu_power += node.capabilities['cpu']['cores'] * cpu_weight
            if node.capabilities['gpu'] != "No GPU found":
                total_gpu_power += gpu_weight

    total_ai_power = total_cpu_power + total_gpu_power
    return {
        "total_power": total_ai_power,
        "cpu_power": total_cpu_power,
        "gpu_power": total_gpu_power,
        "ai_power": total_ai_power
    }
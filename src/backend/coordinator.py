from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NodeRegistration(BaseModel):
    node_id: str
    ip: str
    port: str
    capabilities: dict
    registered: bool = False

class ComputationTask(BaseModel):
    task_id: str
    data: dict

registered_nodes = {}

@app.post("/register-node")
def register_node(node: NodeRegistration):
    if node.node_id in registered_nodes:
        raise HTTPException(status_code=400, detail="Node already registered")

    registered_nodes[node.node_id] = node
    return {"status": "registered", "node": node.dict()}

@app.post("/send-task/{node_id}")
def send_task(node_id: str):
    if node_id not in registered_nodes:
        raise HTTPException(status_code=404, detail="Node not found")
    
    task = ComputationTask(task_id="t1", data={"message": "Run AI logic"})
    node_url = f"http://{registered_nodes[node_id].ip}:{registered_nodes[node_id].port}/compute"
    
    try:
        res = requests.post(node_url, json=task.dict())
        res.raise_for_status()  # Ensure successful response (status code 2xx)
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to send task to node: {e}")
    
    return {"sent_to": node_url, "response": res.json()}

@app.get("/nodes")
def get_registered_nodes():
    return [node.dict() for node in registered_nodes.values()]

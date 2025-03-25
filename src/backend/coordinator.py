from fastapi import FastAPI
from backend.shared.types import NodeRegistration, ComputationTask
import requests

app = FastAPI()
registered_nodes = {}

@app.post("/register-node")
def register_node(node: NodeRegistration):
    registered_nodes[node.node_id] = node
    return {"status": "registered", "node": node}

@app.post("/send-task/{node_id}")
def send_task(node_id: str):
    if node_id not in registered_nodes:
        return {"error": "Node not found"}
    
    task = ComputationTask(task_id="t1", data={"message": "Run AI logic"})
    node_url = f"http://{registered_nodes[node_id].ip}:{registered_nodes[node_id].port}/compute"
    res = requests.post(node_url, json=task.dict())
    return {"sent_to": node_url, "response": res.json()}

# 💡 Nytt endpoint för dashboarden:
@app.get("/nodes")
def get_registered_nodes():
    return [node.dict() for node in registered_nodes.values()]

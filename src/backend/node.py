from fastapi import FastAPI, Request
from backend.shared.types import ComputationTask


app = FastAPI()

# 🔧 Info om denna nod
node_info = {
    "node_id": "node_1",
    "ip": "127.0.0.1",
    "port": 9100,
    "capabilities": {"cpu": "Intel i7", "gpu": "Nvidia RTX 3060"},
    "registered": False
}

@app.get("/")
def get_node_status():
    return {
        "status": "online",
        "registered": node_info["registered"],
        "node": node_info
    }

@app.post("/compute")
def compute(task: ComputationTask):
    result = task.data.get("message", "").upper()
    return {"task_id": task.task_id, "result": result}

@app.post("/receive-data")
async def receive_data(request: Request):
    data = await request.json()
    print("Got data:", data)
    return {"status": "received"}

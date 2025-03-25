from pydantic import BaseModel

class NodeRegistration(BaseModel):
    node_id: str
    ip: str
    port: int
    capabilities: dict

class ComputationTask(BaseModel):
    task_id: str
    data: dict

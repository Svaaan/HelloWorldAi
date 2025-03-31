import os
import httpx
from fastapi import APIRouter
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Detect if we're running inside Docker
USE_DOCKER = os.getenv("USE_DOCKER", "false").lower() == "true"

# Set dynamic base URLs based on environment
if USE_DOCKER:
    COORDINATOR_BASE = "http://coordinator:8100"
    NODE_BASE = "http://node:9100"
else:
    COORDINATOR_BASE = "http://127.0.0.1:8100"
    NODE_BASE = "http://127.0.0.1:9100"

@router.get("/get-total-power")
async def proxy_total_power():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_BASE}/get-total-power")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to reach coordinator: {e}"}


@router.get("/get-connected-nodes-count")
async def proxy_nodes_count():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_BASE}/get-connected-nodes-count")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to reach coordinator: {e}"}


@router.get("/nodes")
async def proxy_nodes():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_BASE}/nodes")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch nodes: {e}"}


@router.get("/usage")
async def proxy_usage():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{NODE_BASE}/usage")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch usage info: {e}"}


@router.post("/connect-node")
async def proxy_connect_node():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_BASE}/connect-node")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to reach node at {NODE_BASE}: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
    
@router.post("/verify-node/{node_id}/cpu")
async def proxy_verify_cpu(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_BASE}/verify-node/{node_id}/cpu")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to start CPU test: {e}"}
    
@router.post("/verify-node/{node_id}/gpu")
async def proxy_verify_gpu(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_BASE}/verify-node/{node_id}/gpu")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to start GPU test: {e}"}
    
@router.get("/node-performance/{node_id}")
async def proxy_node_performance(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_BASE}/node-performance/{node_id}")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch node performance: {e}"}

import os
import httpx
from fastapi import APIRouter
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Load base URLs from environment variables
COORDINATOR_BASE = os.getenv("COORDINATOR_BASE", "http://127.0.0.1:8100")
NODE_BASE = os.getenv("NODE_BASE", "http://127.0.0.1:9100")


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

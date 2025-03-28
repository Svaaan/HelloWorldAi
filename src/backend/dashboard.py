import os
from fastapi import APIRouter
import httpx
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Load base URL from environment variable
COORDINATOR_BASE = os.getenv("COORDINATOR_BASE", "http://127.0.0.1:8100")

async def fetch_node_data():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_BASE}/nodes", timeout=5)
            res.raise_for_status()
            nodes = res.json()
            return nodes
    except httpx.RequestError as e:
        print(f"Error fetching node data: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing response: {e}")
        return []

@router.get("/fetch-node")
async def get_node_info():
    nodes = await fetch_node_data()
    return {"nodes": nodes}

@router.get("/node-info")
async def get_node():
    nodes = await fetch_node_data()
    if len(nodes) > 0:
        node = nodes[0]
        return {
            "node_id": node.get("node_id"),
            "ip": node.get("ip"),
            "port": node.get("port"),
            "capabilities": node.get("capabilities"),
            "isConnected": node.get("isConnected")
        }
    else:
        return {"message": "No nodes connected"}
import os
from fastapi import APIRouter
import httpx
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# Load base URL from environment variable
COORDINATOR_BASE = os.getenv("COORDINATOR_BASE", "http://127.0.0.1:8100")

async def fetch_node_data():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_BASE}/nodes", timeout=5)
            res.raise_for_status()
            nodes = res.json()
            return nodes
    except httpx.RequestError as e:
        print(f"Error fetching node data: {e}")
        return []
    except ValueError as e:
        print(f"Error parsing response: {e}")
        return []

@router.get("/fetch-node")
async def get_node_info():
    nodes = await fetch_node_data()
    return {"nodes": nodes}

@router.get("/node-info")
async def get_node():
    nodes = await fetch_node_data()
    if len(nodes) > 0:
        node = nodes[0]
        return {
            "node_id": node.get("node_id"),
            "ip": node.get("ip"),
            "port": node.get("port"),
            "capabilities": node.get("capabilities"),
            "isConnected": node.get("isConnected")
        }
    else:
        return {"message": "No nodes connected"}
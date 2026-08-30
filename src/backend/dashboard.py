import os
from fastapi import APIRouter
import httpx
from backend.utils.config import COORDINATOR_URL  # ✅ Use shared config

router = APIRouter()

# ✅ Optional: Log to verify
print(f"[Dashboard] Using COORDINATOR_URL: {COORDINATOR_URL}")

# ✅ API endpoint
@router.get("/node-info")
async def get_node_info():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/nodes")
            res.raise_for_status()
            nodes = res.json()

            if nodes:
                node = nodes[0]
                return {
                    "node_id": node.get("node_id"),
                    "ip": node.get("ip"),
                    "capabilities": node.get("capabilities"),
                    "isConnected": node.get("isConnected", True)
                }
            else:
                return {"message": "No nodes connected"}

    except Exception as e:
        return {"error": f"Failed to retrieve node info: {e}"}

import os
from fastapi import APIRouter
import httpx
from dotenv import load_dotenv

load_dotenv()
router = APIRouter()

COORDINATOR_BASE = os.getenv("COORDINATOR_BASE", "http://127.0.0.1:8100")

#PROXY PAGE IS THE GENERAL PAGE FOR THE ROUTING OF THE NODE AND COORDINATOR ETC.
#THIS CAN BE USED FOR EXPLICIT DATA. Minimal views etc.
@router.get("/node-info")
async def get_node_info():

    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_BASE}/nodes")
            res.raise_for_status()
            nodes = res.json()
            if nodes:
                node = nodes[0]
                return {
                    "node_id": node.get("node_id"),
                    "ip": node.get("ip"),
                    "country": node.get("country", "Unknown"),
                    "capabilities": node.get("capabilities"),
                    "isConnected": node.get("isConnected", True)
                }
            else:
                return {"message": "No nodes connected"}
    except Exception as e:
        return {"error": f"Failed to retrieve node info: {e}"}

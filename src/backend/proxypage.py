import os
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from backend.config import NODE_URL, COORDINATOR_URL  # ✅ Clean import from config

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "template")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

router = APIRouter()

# ✅ Optional: Log for verification
print(f"[ProxyPage] NODE_URL: {NODE_URL}")
print(f"[ProxyPage] COORDINATOR_URL: {COORDINATOR_URL}")

@router.patch("/toggle-availability/{node_id}")
async def proxy_toggle_availability(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.patch(f"{COORDINATOR_URL}/toggle-availability/{node_id}")
            res.raise_for_status()
            return res.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.RequestError as e:
        return {"error": f"Failed to toggle availability: {e}"}

@router.get("/get-connected-nodes-count")
async def proxy_nodes_count():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/get-connected-nodes-count")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to reach coordinator: {e}"}

@router.get("/nodes")
async def proxy_nodes():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/nodes")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch nodes: {e}"}

@router.get("/usage")
async def proxy_usage():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{NODE_URL}/usage")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch usage info: {e}"}

@router.post("/connect-node")
async def proxy_connect_node():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/connect-node")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to reach node at {NODE_URL}: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@router.post("/verify-node/{node_id}/cpu")
async def proxy_verify_cpu(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_URL}/verify-node/{node_id}/cpu")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to start CPU test: {e}"}

@router.post("/verify-node/{node_id}/gpu")
async def proxy_verify_gpu(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_URL}/verify-node/{node_id}/gpu")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to start GPU test: {e}"}

@router.get("/node-performance/{node_id}")
async def proxy_node_performance(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/node-performance/{node_id}")
            res.raise_for_status()
            return res.json()
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch node performance: {e}"}

@router.get("/distribution", response_class=HTMLResponse)
async def proxy_distribution_page(request: Request):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/nodes")
            res.raise_for_status()
            all_nodes = res.json()

            available_nodes = [
                node for node in all_nodes
                if node.get("isConnected") and node.get("isAvailable")
            ]

    except httpx.RequestError as e:
        available_nodes = []
        print(f"Failed to fetch nodes: {e}")

    return templates.TemplateResponse("distribution.html", {
        "request": request,
        "available_nodes": available_nodes
    })

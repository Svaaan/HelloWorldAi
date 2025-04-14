import os
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from backend.utils.config import NODE_URL, COORDINATOR_URL

TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "template")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

router = APIRouter()

print(f"[ProxyPage] NODE_URL: {NODE_URL}")
print(f"[ProxyPage] COORDINATOR_URL: {COORDINATOR_URL}")

# 🧩 Helper to process responses safely
def safe_json(res):
    try:
        return res.json()
    except Exception:
        return {"error": "Invalid JSON response"}

@router.patch("/toggle-availability/{node_id}")
async def proxy_toggle_availability(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.patch(f"{COORDINATOR_URL}/toggle-availability/{node_id}")
            res.raise_for_status()
            return safe_json(res)
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
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to reach coordinator: {e}"}

@router.get("/nodes")
async def proxy_nodes(request: Request):
    node_id = request.query_params.get("node_id")
    try:
        async with httpx.AsyncClient() as client:
            url = f"{COORDINATOR_URL}/nodes"
            if node_id:
                url += f"?node_id={node_id}"
            res = await client.get(url)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch nodes: {e}"}

@router.post("/finalize-connection")
async def proxy_finalize_connection():
    try:
        print("[Proxy] Forwarding finalize-connection request to Node")
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/finalize-connection")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to reach node at {NODE_URL}: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
    
@router.post("/process-task/{task_id}")
async def proxy_process_task(task_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/process-task/{task_id}")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to process task: {e}"}

@router.post("/reject-task/{task_id}")
async def proxy_reject_task(task_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/reject-task/{task_id}")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to reject task: {e}"}

@router.get("/get-task-results")
async def proxy_get_task_results():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/get-task-results")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch task results: {e}"}

@router.post("/receive-task-result")
async def proxy_receive_task_result(result: dict):
    try:
        print(f"📥 Task result received: {result}")
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_URL}/receive-task-result", json=result)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to forward task result to coordinator: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@router.post("/node-heartbeat/{node_id}")
async def proxy_node_heartbeat(node_id: str, request: Request):
    try:
        status_payload = await request.json()
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_URL}/node-heartbeat/{node_id}", json=status_payload)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to send heartbeat: {e}"}

@router.delete("/node/{node_id}")
async def proxy_delete_node(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.delete(f"{COORDINATOR_URL}/node/{node_id}")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to delete node: {e}"}

@router.post("/execute-task/{node_id}")
async def proxy_execute_task(node_id: str, request: Request):
    try:
        task_data = await request.json()
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/execute-task/{node_id}", json=task_data)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to send task to node: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@router.get("/usage")
async def proxy_usage():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{NODE_URL}/usage")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch usage info: {e}"}

@router.post("/connect-node")
async def proxy_connect_node(request: Request):
    try:
        payload = await request.json()
        print(f"[Proxy] Forwarding connect-node with payload: {payload}")  # ✅ Add this line!
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/connect-node", json=payload)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to reach node at {NODE_URL}: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

@router.post("/queue-task/{node_id}")
async def proxy_queue_task(node_id: str, request: Request):
    try:
        task_data = await request.json()
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/queue-task/{node_id}", json=task_data)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"status": "error", "message": f"Failed to queue task: {e}"}

@router.get("/get-pending-tasks")
async def proxy_get_pending_tasks():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{NODE_URL}/get-pending-tasks")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch pending tasks: {e}"}

@router.post("/verify-node/{node_id}/cpu")
async def proxy_verify_cpu(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_URL}/verify-node/{node_id}/cpu")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to start CPU test: {e}"}

@router.post("/verify-node/{node_id}/gpu")
async def proxy_verify_gpu(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_URL}/verify-node/{node_id}/gpu")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to start GPU test: {e}"}

@router.get("/generate-challenge/{node_id}")
async def proxy_generate_challenge(node_id: str):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/generate-challenge/{node_id}")
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to generate challenge: {e}"}


@router.post("/verify-challenge/{node_id}")
async def proxy_verify_challenge(node_id: str, request: Request):
    try:
        signature_payload = await request.json()
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{COORDINATOR_URL}/verify-challenge/{node_id}",
                json=signature_payload
            )
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to verify challenge: {e}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
    
@router.get("/distribution", response_class=HTMLResponse)
async def proxy_distribution_page(request: Request):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/nodes")
            res.raise_for_status()
            all_nodes = safe_json(res)

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

@router.get("/available-nodes")
async def proxy_available_nodes():
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/nodes")
            res.raise_for_status()
            all_nodes = safe_json(res)

            available_nodes = [
                node for node in all_nodes
                if node.get("isConnected") and node.get("isAvailable")
            ]

            return available_nodes
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch available nodes: {e}"}

@router.post("/find-node-id")
async def proxy_find_node_id(request: Request):
    try:
        payload = await request.json()
        print(f"[Proxy] Forwarding find-node-id with payload: {payload}")  # Optional: log

        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_URL}/find-node-id", json=payload)
            res.raise_for_status()
            return safe_json(res)

    except httpx.RequestError as e:
        return {"error": f"Failed to reach coordinator for find-node-id: {str(e)}"}

    except Exception as e:
        return {"error": f"Unexpected error during find-node-id: {str(e)}"}
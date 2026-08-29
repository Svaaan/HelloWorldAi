import os
import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
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


# 🔑 Pass the caller's credentials through to the upstream service.
#
# Two kinds now: a node's bearer token, and the submitter key that identifies
# whoever sent a job. Dropping either here silently breaks the feature it
# belongs to -- a job submitted through this proxy without its key arrives
# with no owner, so nobody can ever collect the model it produces.
def auth_headers(request: Request) -> dict:
    headers = {}

    token = request.headers.get("authorization")
    if token:
        headers["Authorization"] = token

    submitter = request.headers.get("x-submitter-key")
    if submitter:
        headers["X-Submitter-Key"] = submitter

    return headers

@router.patch("/toggle-availability/{node_id}")
async def proxy_toggle_availability(node_id: str, request: Request):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.patch(
                f"{COORDINATOR_URL}/toggle-availability/{node_id}",
                headers=auth_headers(request),
            )
            res.raise_for_status()
            return safe_json(res)
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=safe_json(e.response))
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
async def proxy_finalize_connection(request: Request):
    try:
        print("[Proxy] Forwarding finalize-connection request to Node")
        body = await request.json()

        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/finalize-connection", json=body)
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
            res = await client.post(
                f"{COORDINATOR_URL}/node-heartbeat/{node_id}",
                json=status_payload,
                headers=auth_headers(request),
            )
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

async def _post_to_node(path: str, body=None):
    async with httpx.AsyncClient() as client:
        res = await client.post(f"{NODE_URL}{path}", json=body, timeout=20)
        if res.status_code >= 400:
            raise HTTPException(status_code=res.status_code, detail=safe_json(res))
        return safe_json(res)


@router.post("/approve-task/{task_id}")
async def proxy_approve_task(task_id: str):
    try:
        return await _post_to_node(f"/approve-task/{task_id}")
    except httpx.RequestError as e:
        return {"error": f"Failed to reach node: {e}"}


@router.post("/decline-task/{task_id}")
async def proxy_decline_task(task_id: str):
    try:
        return await _post_to_node(f"/decline-task/{task_id}")
    except httpx.RequestError as e:
        return {"error": f"Failed to reach node: {e}"}


@router.post("/approval-mode")
async def proxy_approval_mode(request: Request):
    try:
        return await _post_to_node("/approval-mode", await request.json())
    except httpx.RequestError as e:
        return {"error": f"Failed to reach node: {e}"}


@router.get("/current-task")
async def proxy_current_task():
    """Live progress and thermal state from the node."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{NODE_URL}/current-task", timeout=10)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to reach node: {e}"}


@router.post("/self-test")
async def proxy_run_self_test(request: Request):
    """Ask the node to train something small and report what it managed."""
    try:
        async with httpx.AsyncClient() as client:
            # The body is the owner's optional CSV. Forwarding it is the whole
            # point of the option: without this the node always saw an empty
            # request and quietly trained on synthetic data instead.
            body = await request.body()
            headers = auth_headers(request)
            if body:
                headers["Content-Type"] = request.headers.get("content-type", "text/csv")

            # Generous: this trains a real model before it answers.
            res = await client.post(f"{NODE_URL}/self-test", content=body,
                                    params=dict(request.query_params),
                                    headers=headers, timeout=1200)
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach node: {e}")


@router.post("/self-test/stop")
async def proxy_stop_self_test(request: Request):
    """Ask a running test to stop."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/self-test/stop",
                                    headers=auth_headers(request), timeout=20)
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach node: {e}")


@router.get("/self-test")
async def proxy_get_self_test(request: Request):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{NODE_URL}/self-test",
                                   headers=auth_headers(request), timeout=20)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach node: {e}")


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

@router.post("/node-session")
async def proxy_node_session(request: Request):
    """Forward the browser-obtained session token to the node process."""
    try:
        body = await request.json()
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{NODE_URL}/node-session", json=body)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to reach node at {NODE_URL}: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@router.post("/verify-task/{task_id}")
async def proxy_verify_task(task_id: str):
    """Score a completed task's model against the withheld holdout."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(f"{COORDINATOR_URL}/verify-task/{task_id}", timeout=120)
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to reach coordinator: {e}"}


# A corpus worth training a language model on is measured in tens of
# megabytes. Two minutes was comfortable for a spreadsheet and not for that:
# the upload has to cross the browser's connection, then this hop, and the
# coordinator has to convert it before answering.
ARTIFACT_UPLOAD_TIMEOUT = int(os.getenv("ARTIFACT_UPLOAD_TIMEOUT", 900))


@router.post("/artifacts")
async def proxy_upload_artifact(request: Request):
    """Forward a dataset (or weights) blob to the coordinator's store."""
    try:
        body = await request.body()
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{COORDINATOR_URL}/artifacts",
                params=dict(request.query_params),
                content=body,
                headers={"Content-Type": "application/octet-stream"},
                timeout=ARTIFACT_UPLOAD_TIMEOUT,
            )
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        return {"status": "error", "message": f"Failed to reach coordinator: {e}"}


@router.post("/artifacts/{artifact_id}/append")
async def proxy_append_artifact(artifact_id: str, request: Request):
    """Add another file to a dataset that was already uploaded."""
    body = await request.body()
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{COORDINATOR_URL}/artifacts/{artifact_id}/append",
                params=dict(request.query_params),
                content=body,
                headers={"Content-Type": "application/octet-stream"},
                timeout=ARTIFACT_UPLOAD_TIMEOUT,
            )
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        return {"status": "error", "message": f"Failed to reach coordinator: {e}"}


@router.get("/artifacts/{artifact_id}")
async def proxy_download_artifact(artifact_id: str, request: Request):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{COORDINATOR_URL}/artifacts/{artifact_id}",
                headers=auth_headers(request),
                timeout=120,
            )
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail="Artifact not available.")
            return Response(content=res.content, media_type="application/octet-stream")
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach coordinator: {e}")


@router.post("/submit-task")
async def proxy_submit_task_anywhere(request: Request):
    """Queue work and let the coordinator choose the node."""
    try:
        task_data = await request.json()
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{COORDINATOR_URL}/submit-task",
                json=task_data,
                headers=auth_headers(request),
                timeout=60,
            )
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach coordinator: {e}")


@router.post("/submit-task/{node_id}")
async def proxy_submit_task(node_id: str, request: Request):
    """Queue work for a node. Goes to the coordinator, which holds the queue.

    Unlike /queue-task this never tries to reach the node directly - the node
    polls for its own work, so contributors behind home routers can take part.
    """
    try:
        task_data = await request.json()
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{COORDINATOR_URL}/submit-task/{node_id}",
                json=task_data,
                headers=auth_headers(request),
            )
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        return {"status": "error", "message": f"Failed to reach coordinator: {e}"}


@router.post("/cancel-task/{task_id}")
async def proxy_cancel_task(task_id: str, request: Request):
    """Stop a job the caller submitted."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{COORDINATOR_URL}/cancel-task/{task_id}",
                headers=auth_headers(request), timeout=20,
            )
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach coordinator: {e}")


@router.post("/retry-task/{task_id}")
async def proxy_retry_task(task_id: str, request: Request):
    """Queue the same job again, with any changed settings.

    The body carries the changes. Forwarding only the headers -- which this did
    -- meant the coordinator saw an empty request and re-ran the original
    settings while reporting success, so the page showed a tuned run that had
    quietly ignored every value typed into it.
    """
    body = await request.body()
    headers = auth_headers(request)
    if body:
        headers["Content-Type"] = "application/json"

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{COORDINATOR_URL}/retry-task/{task_id}",
                headers=headers, content=body or None, timeout=20,
            )
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach coordinator: {e}")


@router.get("/job-schema")
async def proxy_job_schema():
    """The fields a job may contain, used to build the submit form."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/job-schema", timeout=15)
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach coordinator: {e}")


@router.get("/my-tasks")
async def proxy_my_tasks(request: Request):
    """The jobs submitted from this browser, for collecting finished models."""
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"{COORDINATOR_URL}/my-tasks",
                params=dict(request.query_params),
                headers=auth_headers(request),
                timeout=30,
            )
            if res.status_code >= 400:
                raise HTTPException(status_code=res.status_code, detail=safe_json(res))
            return safe_json(res)
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach coordinator: {e}")


@router.get("/tasks")
async def proxy_tasks(request: Request):
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/tasks", params=dict(request.query_params))
            res.raise_for_status()
            return safe_json(res)
    except httpx.RequestError as e:
        return {"error": f"Failed to fetch tasks: {e}"}


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
    """Nodes offering their GPUs, straight from the coordinator.

    This used to fetch /nodes and re-apply the availability filter here. That
    is the same question asked twice, and the two answers had drifted: the
    coordinator's /available-nodes fills in each node's throughput, so going
    the long way round reported every machine as 0 TFLOPS.
    """
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/available-nodes", timeout=20)
            res.raise_for_status()
            return safe_json(res)
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
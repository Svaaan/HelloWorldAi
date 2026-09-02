"""The dashboard's proxy to the coordinator and the local node.

Why this is a table and not forty handlers
------------------------------------------
It used to be forty-two hand-written routes, each deciding for itself whether
to pass on the query string, the body and the caller's credentials. Measured
before this was rewritten: 21 of them dropped the credentials, 34 dropped the
query string, and 8 dropped the body.

Nothing about that fails loudly, which is the real problem. The known cases:

    next-task, task-result, task-cancelled   never forwarded at all, so a node
                                             installed from the setup page
                                             registered, reported in, showed as
                                             Connected with the right graphics
                                             card, and never received a job
    artifacts                                set its own Content-Type by
                                             replacing the header dict, so the
                                             submitter key never arrived and
                                             every upload came back 401
    execute-task                             forwarded to a path the node does
                                             not serve
    self-test, available-nodes, retry-task   headers passed on, body dropped

Every one of those is the same mistake, and writing the next route by hand is
an invitation to make it again. So there is one forwarder, it passes on
everything it was given, and a route is a line in a table saying which service
it belongs to. Adding a route can no longer mean forgetting a header.

tests/test_proxy_forwarding.py drives every entry against a fake upstream and
checks what actually came out.
"""

import json
import os
import time

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates

from backend.utils.config import NODE_URL, COORDINATOR_URL

TEMPLATE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "frontend", "template")
templates = Jinja2Templates(directory=TEMPLATE_DIR)

router = APIRouter()

# Datasets are large and slow to send; the rest of the calls are small.
ARTIFACT_UPLOAD_TIMEOUT = int(os.getenv("ARTIFACT_UPLOAD_TIMEOUT", 900))
DEFAULT_TIMEOUT = 30


def safe_json(res):
    """The upstream's JSON, or a description of why there wasn't any."""
    try:
        return res.json()
    except Exception:
        return {"error": "Invalid JSON response"}


def auth_headers(request: Request, extra: dict = None) -> dict:
    """The caller's credentials, plus whatever else the request needs.

    Two kinds: a node's bearer token, and the submitter key identifying whoever
    sent a job. Dropping either silently breaks the feature it belongs to -- a
    job submitted without its key arrives with no owner, so nobody can collect
    the model it produces.

    `extra` exists so a caller that must set its own Content-Type can add to
    the credentials rather than replace them.
    """
    headers = dict(extra or {})

    token = request.headers.get("authorization")
    if token:
        headers["Authorization"] = token

    submitter = request.headers.get("x-submitter-key")
    if submitter:
        headers["X-Submitter-Key"] = submitter

    return headers


# Response headers worth repeating to the browser. Content-Type so a zip
# arrives as a zip and a CSV as a CSV; Content-Disposition so the download
# keeps the filename the coordinator chose. Hop-by-hop headers such as
# Content-Length and Transfer-Encoding are deliberately not copied -- they
# describe the upstream connection, not this one.
PASSED_BACK = ("content-type", "content-disposition")


async def forward(request: Request, upstream: str, timeout: float):
    """Send this request on unchanged, and hand the answer back unchanged.

    Everything travels: method, path, query string, body and credentials. The
    reply keeps its status code, so a 404 from the coordinator reaches the
    browser as a 404 rather than as a 200 carrying the word "error" -- which is
    what the pages already check for with `response.ok`.
    """
    body = await request.body()

    headers = auth_headers(request)
    content_type = request.headers.get("content-type")
    if content_type:
        headers["Content-Type"] = content_type

    try:
        async with httpx.AsyncClient() as client:
            res = await client.request(
                request.method,
                f"{upstream}{request.url.path}",
                params=dict(request.query_params),
                content=body if body else None,
                headers=headers,
                timeout=timeout,
            )
    except httpx.RequestError as e:
        # A shape that suits both ways the pages check for failure: a real
        # status for `response.ok`, and an `error` key for the few places that
        # read one off the body.
        #
        # A node that cannot be reached is worth saying plainly, because on the
        # central deployment it is not a fault at all: there is no node there
        # and there never will be. A contributor's agent runs on their own
        # machine alongside its own copy of this dashboard, and that is where
        # it is registered and managed from. Leaking "Temporary failure in name
        # resolution" invites somebody to debug DNS for an hour over a
        # container that was never meant to exist here.
        if upstream == NODE_URL:
            message = (
                "No node agent is running on this machine, so there is nothing "
                "here to register yet. A graphics card is offered from the "
                "machine it is in: install the agent there, and register it "
                "from the dashboard that starts alongside it."
            )
            status = 503
        else:
            message = "Could not reach %s: %s" % (upstream, e)
            status = 502

        payload = {"error": message, "detail": message}
        if status == 503:
            # A flag rather than a sentence to match on. The pages offer the
            # setup guide when they see this, and wording that a page has to
            # recognise by its text is wording nobody can edit safely.
            payload["no_local_node"] = True

        return Response(
            content=json.dumps(payload).encode(),
            status_code=status,
            media_type="application/json",
        )

    return Response(
        content=res.content,
        status_code=res.status_code,
        headers={k: v for k, v in res.headers.items() if k.lower() in PASSED_BACK},
    )


# --- what goes where -------------------------------------------------------
#
# (methods, path, upstream, timeout). The path is used unchanged on both sides:
# what the browser asks this service for is what this service asks the other
# one for, so there is no second place for the two to disagree.

COORDINATOR_ROUTES = [
    # Looking at the network
    (["GET"], "/nodes", DEFAULT_TIMEOUT),
    (["GET"], "/available-nodes", DEFAULT_TIMEOUT),
    (["GET"], "/get-connected-nodes-count", DEFAULT_TIMEOUT),
    (["GET"], "/tasks", DEFAULT_TIMEOUT),
    (["GET"], "/get-task-results", DEFAULT_TIMEOUT),
    (["GET"], "/job-schema", 15),
    # What the network has actually managed on jobs like this one, so the form
    # can turn a step count into a number of minutes.
    (["GET"], "/throughput", 15),
    (["PATCH"], "/toggle-availability/{node_id}", DEFAULT_TIMEOUT),
    (["DELETE"], "/node/{node_id}", DEFAULT_TIMEOUT),

    # What a node agent needs. The setup page hands contributors an install
    # command pointing at this origin, because one public address is one thing
    # to expose and one certificate to keep -- and the production compose binds
    # the coordinator to localhost, so its own port does not exist from
    # outside. These three were the ones missing.
    (["GET"], "/next-task/{node_id}", DEFAULT_TIMEOUT),
    (["POST"], "/task-result/{task_id}", 60),
    (["GET"], "/task-cancelled/{task_id}", 15),
    (["POST"], "/node-heartbeat/{node_id}", DEFAULT_TIMEOUT),
    (["POST"], "/receive-task-result", DEFAULT_TIMEOUT),

    # Registering and proving a node's identity
    (["GET"], "/generate-challenge/{node_id}", DEFAULT_TIMEOUT),
    (["POST"], "/verify-challenge/{node_id}", DEFAULT_TIMEOUT),
    (["POST"], "/find-node-id", DEFAULT_TIMEOUT),
    # How an agent puts itself in the database. Distinct from /connect-node in
    # NODE_ROUTES below, which the browser calls on the agent running beside
    # it -- the two used to share a path, and this one lost, so no
    # contributor's node could register through the public address at all.
    (["POST"], "/register-node", 60),
    (["POST"], "/verify-node/{node_id}/cpu", 120),
    (["POST"], "/verify-node/{node_id}/gpu", 120),
    (["POST"], "/verify-task/{task_id}", 120),

    # Sending work and getting it back
    (["POST"], "/submit-task", 60),
    (["POST"], "/submit-task/{node_id}", 60),
    (["POST"], "/cancel-task/{task_id}", DEFAULT_TIMEOUT),
    (["POST"], "/retry-task/{task_id}", DEFAULT_TIMEOUT),
    (["GET"], "/my-tasks", DEFAULT_TIMEOUT),
    (["POST"], "/my-tasks/{task_id}/sample", 120),

    # Downloads: the reply is a zip or a CSV rather than JSON, which the
    # forwarder handles by keeping the upstream's content type.
    (["GET"], "/my-tasks/{task_id}/bundle", 120),
    (["POST"], "/my-tasks/{task_id}/predict", 300),
    (["GET"], "/artifacts/{artifact_id}", 120),

    # Storing a dataset
    (["POST"], "/artifacts", ARTIFACT_UPLOAD_TIMEOUT),
    (["POST"], "/artifacts/{artifact_id}/append", ARTIFACT_UPLOAD_TIMEOUT),
]

NODE_ROUTES = [
    (["POST"], "/connect-node", 60),
    (["POST"], "/finalize-connection", 60),
    (["POST"], "/node-session", DEFAULT_TIMEOUT),
    (["GET"], "/current-task", 10),
    (["GET"], "/usage", DEFAULT_TIMEOUT),
    (["POST"], "/approve-task/{task_id}", DEFAULT_TIMEOUT),
    (["POST"], "/decline-task/{task_id}", DEFAULT_TIMEOUT),
    (["POST"], "/approval-mode", DEFAULT_TIMEOUT),
    (["GET"], "/self-test", DEFAULT_TIMEOUT),
    # A self-test runs the GPU flat out for a while; it is the one call here
    # that legitimately takes minutes.
    (["POST"], "/self-test", 1200),
    (["POST"], "/self-test/stop", DEFAULT_TIMEOUT),
]


def _register(methods, path, upstream, timeout):
    async def handler(request: Request):
        return await forward(request, upstream, timeout)

    # A readable name in tracebacks and in the OpenAPI schema.
    handler.__name__ = "proxy_" + path.strip("/").replace("/", "_").replace(
        "{", "").replace("}", "")
    router.add_api_route(path, handler, methods=methods, include_in_schema=False)


for _methods, _path, _timeout in COORDINATOR_ROUTES:
    _register(_methods, _path, COORDINATOR_URL, _timeout)

for _methods, _path, _timeout in NODE_ROUTES:
    _register(_methods, _path, NODE_URL, _timeout)


# --- what kind of dashboard this is ----------------------------------------

_local_node_seen = {"answer": None, "at": 0.0}

# Asymmetric on purpose, because the two answers are not equally safe to be
# wrong about.
#
# "There is an agent" is stable: a contributor's node agent runs for as long as
# their machine is on, so this can be held for a while and saves a call on every
# page render.
#
# "There is no agent" is the dangerous one. The front door is rendered from it,
# and a false negative on a contributor's own machine hides the buttons they
# need and offers them a setup guide for software they have already installed.
# That happened the first time this shipped: `docker compose up` starts the
# dashboard and the agent together, the dashboard answered a page load first,
# and the machine that was running a graphics card was told to go and install
# one -- for thirty seconds, with no way to tell it was lying.
#
# So a negative is re-checked often. It costs an outbound call that fails fast:
# on the central server there is no `node` host at all, so it ends in DNS
# resolution rather than a timeout.
LOCAL_NODE_CACHE_SECONDS = 30
LOCAL_NODE_MISS_CACHE_SECONDS = 5


async def has_local_node() -> bool:
    """Is a node agent listening beside this dashboard?

    Shared by the /local-node endpoint and by the front door, which is rendered
    differently for the two deployments -- see the template. One function so
    the page and the endpoint can never disagree about it.
    """
    now = time.time()
    cached = _local_node_seen["answer"]
    ttl = LOCAL_NODE_CACHE_SECONDS if cached else LOCAL_NODE_MISS_CACHE_SECONDS
    if cached is not None and now - _local_node_seen["at"] < ttl:
        return cached

    present = False
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{NODE_URL}/current-task", timeout=3)
            present = res.status_code < 500
    except Exception:
        present = False

    _local_node_seen.update({"answer": present, "at": now})
    return present


@router.get("/local-node")
async def local_node():
    """Whether a node agent runs alongside this dashboard.

    The same dashboard image serves two very different jobs. On a contributor's
    machine it sits beside their node agent and is how they register and manage
    it. On the central server there is no node and there cannot be one -- a
    graphics card is offered from the machine it is in.

    The front door used to ask this on load and rearrange itself around the
    answer. It does not any more: hiding the register and connect buttons put a
    second link to the setup guide where the two doors had been, on a card that
    already links the guide in its corner. The dialogs handle the missing agent
    instead, at the point where somebody actually asks for it -- forward() marks
    that 503 with `no_local_node`, and connect/nodeErrors.js explains it there.

    Kept because it answers the question directly, without registering a node to
    find out, which is what you want when checking a deployment: a fresh install
    reports `present: true` from the machine holding the card, and the central
    server reports `present: false`.

    Cached briefly; the answer changes about as often as the deployment does.
    """
    return {"present": await has_local_node()}


# --- the one route that is not a forward -----------------------------------

@router.get("/distribution", response_class=HTMLResponse)
async def proxy_distribution_page(request: Request):
    """The send-work page, rendered with the nodes that can take a job now."""
    available_nodes = []
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/nodes",
                                   timeout=DEFAULT_TIMEOUT)
            res.raise_for_status()
            all_nodes = safe_json(res)

            if isinstance(all_nodes, list):
                available_nodes = [
                    node for node in all_nodes
                    if node.get("isConnected") and node.get("isAvailable")
                ]
    except (httpx.RequestError, httpx.HTTPStatusError) as e:
        # The page is still worth serving; it polls for nodes once it loads.
        print(f"Failed to fetch nodes for /distribution: {e}")

    return templates.TemplateResponse("distribution.html", {
        "request": request,
        "available_nodes": available_nodes,
    })

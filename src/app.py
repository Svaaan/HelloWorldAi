import os
import sys
import multiprocessing
import time

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from backend.utils.config import USE_DOCKER, NODE_PORT, COORDINATOR_PORT, DASHBOARD_PORT  # ✅ import from config

# Paths
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SRC_DIR, "frontend", "template")
STATIC_DIR = os.path.join(SRC_DIR, "frontend", "static")

# Add src dir to path
sys.path.insert(0, SRC_DIR)

# Kill ports (if not Docker)
if not USE_DOCKER:
    from backend.terminate_port import kill_process_on_port
    kill_process_on_port(COORDINATOR_PORT)
    kill_process_on_port(NODE_PORT)
    kill_process_on_port(DASHBOARD_PORT)

# The dashboard's own pieces, and nothing else.
#
# The node and coordinator apps used to be imported here as well, so that
# run_node() and run_coordinator() below could reach them. That made importing
# this module import all three services, including a database driver the
# dashboard has no use for -- it proxies to the coordinator over HTTP and never
# opens a connection of its own. A contributor's dashboard, built from a
# requirements file with no driver in it, crash-looped on:
#
#     File "/app/src/backend/routes/artifacts.py", line 19
#     ModuleNotFoundError: No module named 'bson'
#
# for a coordinator it was never going to run. They are imported inside the
# functions that need them instead, and tests/test_image_requirements.py checks
# each image's requirements file against what its entry point actually imports.
from backend.dashboard import router as dashboard_router
from backend.proxypage import COORDINATOR_URL, has_local_node
from backend.proxypage import router as proxy_router

# Dashboard app
dashboard_app = FastAPI(title="AI Node Dashboard")

# Routers
dashboard_app.include_router(dashboard_router)
dashboard_app.include_router(proxy_router)

# Mount static + template files
templates = Jinja2Templates(directory=TEMPLATE_DIR)

dashboard_app.mount("/template", StaticFiles(directory=TEMPLATE_DIR), name="template")
dashboard_app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# CORS middleware
dashboard_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Without Cache-Control, browsers apply heuristic caching and keep serving
# stale pages, JS modules and stylesheets after an edit. `no-cache` means
# "revalidate before use", not "never store": the browser still gets 304s, it
# just never shows out-of-date code.
#
# This covers the HTML routes as well as /static -- the page templates were
# being cached too, so a rebuilt layout kept rendering from the old markup.
@dashboard_app.middleware("http")
async def revalidate_pages_and_assets(request, call_next):
    response = await call_next(request)

    is_asset = request.url.path.startswith(("/static", "/template"))
    is_page = response.headers.get("content-type", "").startswith("text/html")

    if is_asset or is_page:
        response.headers["Cache-Control"] = "no-cache, must-revalidate"
    return response


# What a browser is allowed to do with these pages.
#
# This matters more here than on an ordinary site. A person's node private key
# and their submitter key live in localStorage -- that is the design, the key
# file is the account -- and localStorage is readable by any script running on
# the page. So a single injected script is not a defacement, it is the loss of
# the key that identifies them.
#
# script-src 'self' with no 'unsafe-inline' is the part that earns its keep:
# every page bootstrap was moved into /static/js/page/ so that this can be
# stated without an exception. connect-src stays 'self' because the browser
# only ever talks to this origin; the coordinator and the node are reached
# through the proxy on this same host.
CSP = "; ".join([
    "default-src 'self'",
    "script-src 'self'",
    "style-src 'self'",
    "img-src 'self' data:",
    "connect-src 'self'",
    "font-src 'self'",
    # Nothing here is meant to be embedded, and nothing embeds anything.
    "frame-ancestors 'none'",
    "form-action 'self'",
    "base-uri 'self'",
    "object-src 'none'",
])


@dashboard_app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)

    response.headers.setdefault("Content-Security-Policy", CSP)
    # frame-ancestors covers this for modern browsers; the older header costs
    # one line and covers the rest.
    response.headers.setdefault("X-Frame-Options", "DENY")
    # Stops a browser deciding for itself that an uploaded .csv is really HTML.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    # A node id or task id in a path should not travel to another site.
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault(
        "Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    return response


# Whether the coordinator has a GitHub OAuth application configured.
#
# Asked rather than assumed, for the same reason has_local_node is: the same
# image serves a central deployment and a contributor's own machine, and only
# one of them has any business offering a sign-in. Somebody running the stack
# at home should not be shown a button that ends in "GitHub sign-in is not set
# up on this deployment".
#
# Cached for a while because the answer changes when the server is restarted
# with new settings, and not otherwise.
_github_signin = {"answer": None, "at": 0.0}
GITHUB_SIGNIN_CACHE_SECONDS = 300


async def github_signin() -> bool:
    # Never on a contributor's own dashboard, whatever the coordinator says.
    #
    # Sign-in only works on the origin the OAuth application's callback URL
    # names, and that is the central deployment. A contributor's dashboard has
    # no coordinator of its own -- it proxies to the central one over the
    # internet -- so pressing the button there sent the browser to GitHub, and
    # GitHub sent it on to the *public* domain. They ended up signed in on a
    # different site than the one they were reading, and the dashboard in front
    # of them still showed them signed out, with nothing to explain it.
    if await has_local_node():
        return False

    now = time.time()
    if (_github_signin["answer"] is not None
            and now - _github_signin["at"] < GITHUB_SIGNIN_CACHE_SECONDS):
        return _github_signin["answer"]

    configured = False
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(f"{COORDINATOR_URL}/auth/config", timeout=5)
            configured = bool((res.json() or {}).get("github"))
    except Exception:
        # An older coordinator has no /auth routes. Not an error: it means this
        # deployment has no sign-in, which is what the front door will show.
        configured = False

    _github_signin.update({"answer": configured, "at": now})
    return configured


# === Frontend routes ===
@dashboard_app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def start_page(request: Request):
    """The front door.

    This used to redirect to /connect, which asks for a GPU. Somebody arriving
    with a dataset and an ordinary laptop was told their hardware was
    unsuitable for something they had not asked to do. The two sides of the
    network get a door each.

    The GPU door differs between the two deployments, and is rendered here
    rather than adjusted by script after the page loads. "Create key file" and
    "I have a key file" both end in a call to the node agent running beside
    this dashboard. On a contributor's machine that is the whole point. On the
    central server there is no agent and cannot be one -- not even for somebody
    browsing from the machine that has one, because this page is served over
    HTTPS and their agent listens on plain HTTP on their own localhost. There,
    those buttons can only ever produce an error, so the card offers the guide
    instead.

    Rendered server-side so the right markup arrives on the first paint. Doing
    it in the browser meant the page visibly rearranged itself a moment after
    it appeared, which reads as a glitch even when the end state is right.
    """
    if not os.path.exists(os.path.join(TEMPLATE_DIR, "start.html")):
        return HTMLResponse("<h1>404 - start.html not found</h1>", status_code=404)

    local = await has_local_node()

    return templates.TemplateResponse("start.html", {
        "request": request,
        "has_local_node": local,
        "github_signin": await github_signin(),
        # Where training actually happens, for a contributor's dashboard to
        # point at rather than offer itself.
        #
        # localStorage is per-origin, so a submitter key made on
        # http://localhost:3000 is a different identity from one made on the
        # public site -- same person, same browser, two workspaces, and nothing
        # saying so. The card still earns its place here: it is where somebody
        # who came to lend a card learns the other half exists. It just should
        # not be the place they take it up.
        "main_site": COORDINATOR_URL if local else None,
    })

@dashboard_app.get("/connect", include_in_schema=False)
def render_connect():
    """Registering a node happens on the front door now.

    This was a second page doing the same job the front door already did for
    the other side of the network: one screen said "here are the two things
    you can be", the next said "and here is how you become one of them". The
    data side never had that step -- it made a key and went straight to a
    workspace -- so the GPU side took two screens for one decision.

    Kept as a redirect rather than deleted: the setup guide, the installer's
    instructions and anybody's bookmarks all point here.
    """
    return RedirectResponse("/", status_code=308)

@dashboard_app.get("/setup", response_class=HTMLResponse)
def render_setup_page():
    path = os.path.join(TEMPLATE_DIR, "setup.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - setup.html not found</h1>", status_code=404)

@dashboard_app.get("/node", response_class=HTMLResponse)
def render_node_page():
    path = os.path.join(TEMPLATE_DIR, "node.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - node.html not found</h1>", status_code=404)

@dashboard_app.get("/workspace", response_class=HTMLResponse)
async def workspace_page():
    path = os.path.join(TEMPLATE_DIR, "workspace.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - workspace.html not found</h1>", status_code=404)

@dashboard_app.get("/distribution", response_class=HTMLResponse)
def render_distribution_page():
    path = os.path.join(TEMPLATE_DIR, "distribution.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - distribution.html not found</h1>", status_code=404)

# === Run services ===

def run_node():
    # Imported here so that starting the dashboard does not require the node's
    # dependencies, and vice versa.
    from backend.node import app as node_app
    uvicorn.run(app=node_app, host="0.0.0.0" if USE_DOCKER else "127.0.0.1", port=NODE_PORT)

def run_coordinator():
    from backend.coordinator import app as coordinator_app
    uvicorn.run(app=coordinator_app, host="0.0.0.0" if USE_DOCKER else "127.0.0.1", port=COORDINATOR_PORT)

def run_dashboard():
    uvicorn.run(app=dashboard_app, host="0.0.0.0" if USE_DOCKER else "127.0.0.1", port=DASHBOARD_PORT)

# Final exportable app
app = dashboard_app

# Multiprocessing entrypoint
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    processes = [
        multiprocessing.Process(target=run_node),
        multiprocessing.Process(target=run_coordinator),
        multiprocessing.Process(target=run_dashboard),
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

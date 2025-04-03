import os
import sys
import multiprocessing
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Docker flag and host
USE_DOCKER = os.getenv("USE_DOCKER", "false").lower() == "true"
HOST = "0.0.0.0" if USE_DOCKER else "127.0.0.1"

# Path config
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(SRC_DIR, "frontend", "template")
STATIC_DIR = os.path.join(SRC_DIR, "frontend", "static")
SCRIPT_FILE_PATH = "/mnt/data/setup-node.sh"

# Add src dir to path
sys.path.insert(0, SRC_DIR)

# Kill ports (if not Docker)
if not USE_DOCKER:
    from backend.terminate_port import kill_process_on_port
    kill_process_on_port(8100)
    kill_process_on_port(9100)
    kill_process_on_port(3000)

# Import apps and routers
from backend.node import app as node_app
from backend.coordinator import app as coordinator_app
from backend.dashboard import router as dashboard_router
from backend.proxypage import router as proxy_router

# Dashboard app
dashboard_app = FastAPI(title="AI Node Dashboard")

# Routers
dashboard_app.include_router(dashboard_router)
dashboard_app.include_router(proxy_router)

# Mount static + template files
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

# Frontend routes
@dashboard_app.get("/", include_in_schema=False)
def redirect_to_connect():
    return RedirectResponse(url="/connect")

@dashboard_app.get("/connect", response_class=HTMLResponse)
def render_connect():
    path = os.path.join(TEMPLATE_DIR, "connect.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - connect.html not found</h1>", status_code=404)

@dashboard_app.get("/setup", include_in_schema=False)
def redirect_to_setup_html():
    return RedirectResponse(url="/setup.html")

@dashboard_app.get("/setup.html", response_class=HTMLResponse)
def render_setup_page():
    path = os.path.join(TEMPLATE_DIR, "setup.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - setup.html not found</h1>", status_code=404)

@dashboard_app.get("/static/scripts/setup-node.sh")
def serve_setup_script():
    return FileResponse(SCRIPT_FILE_PATH, media_type="application/x-sh") if os.path.exists(SCRIPT_FILE_PATH) else HTMLResponse("<h1>404 - setup script not found</h1>", status_code=404)

@dashboard_app.get("/node", include_in_schema=False)
def redirect_to_node_html():
    return RedirectResponse(url="/node.html")

@dashboard_app.get("/node.html", response_class=HTMLResponse)
def render_node_page():
    path = os.path.join(TEMPLATE_DIR, "node.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - node.html not found</h1>", status_code=404)

@dashboard_app.get("/distribution", include_in_schema=False)
def redirect_to_distribution_html():
    return RedirectResponse(url="/distribution.html")

@dashboard_app.get("/distribution.html", response_class=HTMLResponse)
def render_distribution_page():
    path = os.path.join(TEMPLATE_DIR, "distribution.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - distribution.html not found</h1>", status_code=404)

# Run each service
def run_node():
    uvicorn.run(app=node_app, host=HOST, port=int(os.getenv("NODE_PORT", 9100)))

def run_coordinator():
    uvicorn.run(app=coordinator_app, host=HOST, port=int(os.getenv("COORDINATOR_PORT", 8100)))

def run_dashboard():
    uvicorn.run(app=dashboard_app, host=HOST, port=int(os.getenv("DASHBOARD_PORT", 3000)))

# Final exportable app
app = dashboard_app

# Run processes
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

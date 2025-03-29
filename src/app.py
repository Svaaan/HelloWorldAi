import os
import sys
import multiprocessing
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Determine host based on environment
USE_DOCKER = os.getenv("USE_DOCKER", "false").lower() == "true"
HOST = "0.0.0.0" if USE_DOCKER else "127.0.0.1"

# Add src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

# ✅ Import from backend utility
from backend.terminate_port import kill_process_on_port

# ✅ Kill ports before starting (only if not using Docker)
if not USE_DOCKER:
    kill_process_on_port(8100)
    kill_process_on_port(9100)
    kill_process_on_port(3000)

# Path to templates
template_dir = os.path.join(src_dir, "frontend", "template")

# Create FastAPI dashboard app
dashboard_app = FastAPI()

# Mount static files (e.g., CSS, JS)
dashboard_app.mount("/template", StaticFiles(directory=template_dir), name="template")

# CORS
dashboard_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import backend services and routers
from backend.node import app as node_app
from backend.coordinator import app as coordinator_app
from backend.dashboard import router as dashboard_router
from backend.proxypage import router as proxy_router

# Register routers
dashboard_app.include_router(dashboard_router)
dashboard_app.include_router(proxy_router)

# Frontend routes
@dashboard_app.get("/", include_in_schema=False)
def redirect_to_connect():
    return RedirectResponse(url="/connect")

@dashboard_app.get("/connect", response_class=HTMLResponse)
def render_connect():
    path = os.path.join(template_dir, "connect.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - connect.html not found</h1>", status_code=404)

@dashboard_app.get("/node", include_in_schema=False)
def redirect_to_node_html():
    return RedirectResponse(url="/node.html")

@dashboard_app.get("/node.html", response_class=HTMLResponse)
def render_node_page():
    path = os.path.join(template_dir, "node.html")
    return FileResponse(path) if os.path.exists(path) else HTMLResponse("<h1>404 - node.html not found</h1>", status_code=404)

# Functions to run each FastAPI app
def run_node():
    port = int(os.getenv("NODE_PORT", 9100))
    uvicorn.run(app=node_app, host=HOST, port=port)

def run_coordinator():
    port = int(os.getenv("COORDINATOR_PORT", 8100))
    uvicorn.run(app=coordinator_app, host=HOST, port=port)

def run_dashboard():
    port = int(os.getenv("DASHBOARD_PORT", 3000))
    uvicorn.run(app=dashboard_app, host=HOST, port=port)

# Final app for deployment
app = dashboard_app

# Run locally
if __name__ == "__main__":
    processes = [
        multiprocessing.Process(target=run_node),
        multiprocessing.Process(target=run_coordinator),
        multiprocessing.Process(target=run_dashboard),
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

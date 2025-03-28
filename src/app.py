import os
import sys
import multiprocessing
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from dotenv import load_dotenv

# Add the src directory to Python path
src_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_dir)

# Now import backend modules
from backend.node import app as node_app
from backend.coordinator import app as coordinator_app
from backend.dashboard import router as dashboard_router
from backend.proxypage import router as proxy_router  # ✅ NEW: proxy routes

# Load environment variables from .env file
load_dotenv()

# Path to templates
template_dir = os.path.abspath(os.path.join(src_dir, "frontend", "template"))
print("📂 Template path:", template_dir)

# Create dashboard-app and add router
dashboard_app = FastAPI()

# Add CORS middleware to allow requests from frontend (adjust the origins as needed)
dashboard_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
dashboard_app.include_router(dashboard_router)
dashboard_app.include_router(proxy_router)  # ✅ Include proxy endpoints

# Redirect root to /connect
@dashboard_app.get("/", include_in_schema=False)
def redirect_to_connect():
    return RedirectResponse(url="/connect")

# Handle GET requests to render the dashboard (connect.html)
@dashboard_app.get("/connect", response_class=HTMLResponse)
def render_connect():
    path = os.path.join(template_dir, "connect.html")
    print("🔍 Serving file:", path)
    if not os.path.exists(path):
        return HTMLResponse("<h1>404 - connect.html not found</h1>", status_code=404)
    return FileResponse(path, media_type="text/html")

# Handle GET requests to render the node page (node.html)
@dashboard_app.get("/node.html", response_class=HTMLResponse)
def render_node_page():
    path = os.path.join(template_dir, "node.html")
    print("🔍 Serving file:", path)
    if not os.path.exists(path):
        return HTMLResponse("<h1>404 - node.html not found</h1>", status_code=404)
    return FileResponse(path, media_type="text/html")

# Functions to run each app separately (local development)
def run_node():
    port = os.getenv("NODE_PORT", 9100)
    uvicorn.run(app=node_app, host="127.0.0.1", port=int(port))

def run_coordinator():
    port = os.getenv("COORDINATOR_PORT", 8100)
    uvicorn.run(app=coordinator_app, host="127.0.0.1", port=int(port))

def run_dashboard():
    port = os.getenv("DASHBOARD_PORT", 3000)
    uvicorn.run(app=dashboard_app, host="127.0.0.1", port=int(port))

# 👇 This is what Render sees — serve dashboard app for deployment
app = dashboard_app

# Local multiprocessing (not used in deployment)
if __name__ == "__main__":
    processes = [
        multiprocessing.Process(target=run_node),
        multiprocessing.Process(target=run_coordinator),
        multiprocessing.Process(target=run_dashboard)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

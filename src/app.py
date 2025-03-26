import os
import sys
import multiprocessing
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from backend.node import app as node_app
from backend.coordinator import app as coordinator_app
from backend.dashboard import router as dashboard_router
import backend.terminate_port

# Lägg till sökvägar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Pekar på src/frontend
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend", "template"))
print("📂 Template path:", template_dir)

# Create dashboard-app and add router
dashboard_app = FastAPI()

# Add CORS middleware to allow requests from frontend (adjust the origins as needed)
dashboard_app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000"],  # Allow only the frontend URL
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

dashboard_app.include_router(dashboard_router)

# Handle GET requests to render the dashboard (connect.html)
@dashboard_app.get("/", response_class=HTMLResponse)
def render_dashboard():
    path = os.path.join(template_dir, "connect.html")
    print("🔍 Servar fil:", path)
    if not os.path.exists(path):
        return HTMLResponse("<h1>404 - connect.html not found</h1>", status_code=404)
    return FileResponse(path, media_type="text/html")

# Handle GET requests to render the node page (node.html)
@dashboard_app.get("/node.html", response_class=HTMLResponse)
def render_node_page():
    path = os.path.join(template_dir, "node.html")
    print("🔍 Servar fil:", path)
    if not os.path.exists(path):
        return HTMLResponse("<h1>404 - node.html not found</h1>", status_code=404)
    return FileResponse(path, media_type="text/html")

# Run node, coordinator, and dashboard on different ports
def run_node():
    uvicorn.run(app=node_app, host="127.0.0.1", port=9100)

def run_coordinator():
    uvicorn.run(app=coordinator_app, host="127.0.0.1", port=8100)

def run_dashboard():
    uvicorn.run(app=dashboard_app, host="127.0.0.1", port=3000)

if __name__ == "__main__":
    # Automatically kill processes using ports 8100, 9100, and 3000
    backend.terminate_port.kill_process_on_port(8100)
    backend.terminate_port.kill_process_on_port(9100)
    backend.terminate_port.kill_process_on_port(3000)

    # Start the backend services
    processes = [
        multiprocessing.Process(target=run_node),
        multiprocessing.Process(target=run_coordinator),
        multiprocessing.Process(target=run_dashboard)
    ]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

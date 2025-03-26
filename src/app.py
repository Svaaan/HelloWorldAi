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
from dotenv import load_dotenv  # Import dotenv

# Load environment variables from .env file
load_dotenv()  # This loads variables from the .env file into the environment

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
    allow_origins=["*"],  # Allow all origins (adjust if necessary)
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

def run_node():
    port = os.getenv("NODE_PORT", 9100)  # Use the environment variable or fallback to 9100
    uvicorn.run(app=node_app, host="127.0.0.1", port=int(port))  # Change this to 127.0.0.1 for localhost

def run_coordinator():
    port = os.getenv("COORDINATOR_PORT", 8100)  # Use the environment variable or fallback to 8100
    uvicorn.run(app=coordinator_app, host="127.0.0.1", port=int(port))  # Change this to 127.0.0.1 for localhost

def run_dashboard():
    port = os.getenv("DASHBOARD_PORT", 3000)  # Use the environment variable or fallback to 3000
    uvicorn.run(app=dashboard_app, host="127.0.0.1", port=int(port))  # Change this to 127.0.0.1 for localhost


if __name__ == "__main__":
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

import multiprocessing
import uvicorn
import os
import sys
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from backend.node import app as node_app
from backend.coordinator import app as coordinator_app
from backend.dashboard import router as dashboard_router, fetch_node_data

# Lägg till sökvägar
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Pekar på src/frontend/home.html
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "frontend"))
print("📂 Template path:", template_dir)

# Skapa dashboard-app och koppla in router
dashboard_app = FastAPI()
dashboard_app.include_router(dashboard_router)

@dashboard_app.get("/", response_class=HTMLResponse)
def render_dashboard():
    path = os.path.join(template_dir, "home.html")
    print("🔍 Servar fil:", path)
    if not os.path.exists(path):
        return HTMLResponse("<h1>404 - home.html not found</h1>", status_code=404)
    return FileResponse(path, media_type="text/html")


def run_node():
    uvicorn.run(app=node_app, host="127.0.0.1", port=9100)

def run_coordinator():
    uvicorn.run(app=coordinator_app, host="127.0.0.1", port=8100)

def run_dashboard():
    uvicorn.run(app=dashboard_app, host="127.0.0.1", port=3000)

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

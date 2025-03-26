# backend/dashboard.py
from fastapi import APIRouter
import requests

router = APIRouter()  # Ensure the router object is created

# Function to fetch node data from the coordinator
def fetch_node_data():
    try:
        # Fetching node data from coordinator
        res = requests.get("http://127.0.0.1:8100/nodes", timeout=5)  # Adding timeout to avoid hanging
        res.raise_for_status()  # Will raise an error for bad responses (e.g., 404, 500)
        nodes = res.json()
        return nodes
    except requests.exceptions.RequestException as e:
        # Handle various request errors (e.g., connection error, timeout, etc.)
        print(f"Error fetching node data: {e}")
        return []  # Return an empty list if the request fails
    except ValueError as e:
        # Handle JSON parsing errors
        print(f"Error parsing response: {e}")
        return []  # Return an empty list if the JSON parsing fails

# Add the appropriate route to get node info
@router.get("/fetch-node")
def get_node_info():
    nodes = fetch_node_data()
    return {"nodes": nodes}

# backend/dashboard.py
from fastapi import APIRouter
import requests

router = APIRouter()

def fetch_node_data():
    try:
        res = requests.get("http://127.0.0.1:8100/nodes")
        nodes = res.json()
    except Exception as e:
        print("Kunde inte hämta noder:", e)
        nodes = []
    return nodes

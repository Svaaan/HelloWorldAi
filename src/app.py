import multiprocessing
import uvicorn
import sys
import os
from node.node import app as node
from coordinator.coordinator import app as coordinator

# Lägg till src som sökväg
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_node():
    uvicorn.run(app=node, host="127.0.0.1", port=9100)

def run_coordinator():
    uvicorn.run(app=coordinator, host="127.0.0.1", port=8100)

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_node)
    p2 = multiprocessing.Process(target=run_coordinator)
    p1.start()
    p2.start()
    p1.join()
    p2.join()

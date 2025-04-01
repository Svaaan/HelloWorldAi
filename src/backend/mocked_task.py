import time
import numpy as np
import psutil
from GPUtil import getGPUs
import torch

def verify_cpu_connected(node_id: str, connected_nodes: dict):
    if node_id not in connected_nodes:
        return False

    print(f"🔍 Verifying CPU connection for node: {node_id}")
    _ = sum(i * i for i in range(10_000))  # Small load
    usage = psutil.cpu_percent(interval=0.1)

    node = connected_nodes[node_id]
    node.cpu_usage = usage
    node.cpu_verified = True
    node.cpu_benchmark = "Verified connection"

    print(f"✅ CPU verified for {node_id} — usage: {usage:.2f}%")
    return True

def verify_gpu_connected(node_id: str, connected_nodes: dict, torch_module=torch):
    if node_id not in connected_nodes or not torch_module.cuda.is_available():
        return False

    node = connected_nodes[node_id]
    gpus = node.capabilities.get("gpu", [])

    if not isinstance(gpus, list) or not gpus:
        print(f"⚠️ Node {node_id} has no reported GPUs.")
        return False

    print(f"🔍 Verifying GPU connection for node: {node_id}")

    for gpu_index, gpu in enumerate(gpus):
        gpu_name = gpu.get("name", "Unknown GPU")
        if "No GPU" in gpu_name or "Detection" in gpu_name:
            continue
        try:
            torch_module.cuda.set_device(gpu_index)
            a = torch_module.rand(50, 50, device='cuda')
            b = torch_module.rand(50, 50, device='cuda')
            c = torch_module.matmul(a, b)
            _ = c.sum().item()
            torch_module.cuda.empty_cache()

            usage = sum(g.load for g in getGPUs()) / len(getGPUs()) * 100

            node.gpu_usage = usage
            node.gpu_verified = True
            node.gpu_benchmark = "Verified connection"

            print(f"✅ GPU verified for {node_id} — usage: {usage:.2f}%")
            return True
        except Exception as e:
            print(f"⚠️ Error testing GPU {gpu_name} on {node_id}: {e}")
            return False

    return False

def simulate_verification_task(node_id: str, connected_nodes: dict, torch_module=torch):
    if node_id not in connected_nodes:
        print(f"❌ Node {node_id} disconnected before verification")
        return

    cpu_result = verify_cpu_connected(node_id, connected_nodes)
    gpu_result = verify_gpu_connected(node_id, connected_nodes, torch_module)

    if node_id in connected_nodes:
        node = connected_nodes[node_id]
        print(f"📊 Node {node_id} verification summary — CPU: {cpu_result}, GPU: {gpu_result}")

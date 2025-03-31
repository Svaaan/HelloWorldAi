import time
import threading
import numpy as np
import psutil
from GPUtil import getGPUs

try:
    import torch
except ImportError:
    torch = None


def run_cpu_task(node_id: str, connected_nodes: dict):
    if node_id not in connected_nodes:
        return False
    print(f"🧪 Running CPU verification task on node: {node_id}")
    connected_nodes[node_id].cpu_usage = psutil.cpu_percent(interval=0.2)
    matrix_size = 1000
    a = np.random.rand(matrix_size, matrix_size)
    b = np.random.rand(matrix_size, matrix_size)
    start_time = time.time()
    c = np.matmul(a, b)
    _ = np.linalg.det(a[:100, :100])
    for _ in range(5):
        c = c @ b
    end_time = time.time()
    print(f"✅ Node {node_id} CPU task completed in {end_time - start_time:.2f} seconds")
    connected_nodes[node_id].cpu_verified = True
    connected_nodes[node_id].last_heartbeat = time.time()
    return True

def run_gpu_task(node_id: str, connected_nodes: dict, torch):
    if node_id not in connected_nodes or not torch.cuda.is_available():
        return False
    node = connected_nodes[node_id]
    gpus = node.capabilities.get('gpu', [])
    if not isinstance(gpus, list) or not gpus:
        print(f"⚠️ Node {node_id} has no usable GPUs.")
        return False
    print(f"🧪 Running minimal GPU task(s) on node: {node_id}")
    success_count = 0
    for gpu_index, gpu in enumerate(gpus):
        gpu_name = gpu.get("name", "Unknown GPU")
        if "No GPU" in gpu_name or "Detection" in gpu_name:
            continue
        try:
            torch.cuda.set_device(gpu_index)
            tensor_size = 1000
            a = torch.rand(tensor_size, tensor_size, device='cuda')
            b = torch.rand(tensor_size, tensor_size, device='cuda')
            c = torch.matmul(a, b)
            for _ in range(2):
                c = torch.nn.functional.relu(c @ b)
            _ = c.sum().item()
            del a, b, c
            torch.cuda.empty_cache()
            print(f"✅ GPU {gpu_name} (index {gpu_index}) verified on node {node_id}")
            success_count += 1
        except Exception as e:
            print(f"⚠️ Error verifying GPU {gpu_name} (index {gpu_index}) on node {node_id}: {e}")
    if success_count > 0:
        try:
            live_gpus = getGPUs()
            if live_gpus:
                avg_load = sum(gpu.load for gpu in live_gpus) / len(live_gpus)
                node.gpu_usage = round(avg_load * 100, 2)
        except:
            pass
        node.gpu_verified = True
        node.last_heartbeat = time.time()
        return True
    return False

def run_mini_tasks(node_id: str, connected_nodes: dict, torch):
    if node_id not in connected_nodes or not connected_nodes[node_id].isConnected:
        return
    node = connected_nodes[node_id]
    try:
        _ = sum(i * i for i in range(100_000))
        node.cpu_usage = psutil.cpu_percent(interval=0.1)
        node.last_heartbeat = time.time()
    except Exception as e:
        print(f"⚠️ Error in mini CPU task on node {node_id}: {e}")
    if node.gpu_verified and torch.cuda.is_available():
        gpus = node.capabilities.get('gpu', [])
        for gpu_index, gpu in enumerate(gpus):
            gpu_name = gpu.get("name", "Unknown GPU")
            if "No GPU" in gpu_name or "Detection" in gpu_name:
                continue
            try:
                torch.cuda.set_device(gpu_index)
                a = torch.rand(500, 500, device='cuda')
                b = torch.rand(500, 500, device='cuda')
                c = a @ b
                _ = c.sum().item()
                del a, b, c
                torch.cuda.empty_cache()
                node.gpu_usage = 1.0
                node.last_heartbeat = time.time()
            except Exception as e:
                print(f"⚠️ Error in mini GPU task on {gpu_name} (index {gpu_index}): {e}")

def simulate_verification_task(node_id: str, connected_nodes: dict, torch):
    if node_id not in connected_nodes:
        print(f"❌ Node {node_id} disconnected before verification")
        return
    cpu_result = run_cpu_task(node_id, connected_nodes)
    gpu_result = run_gpu_task(node_id, connected_nodes, torch)
    if node_id in connected_nodes:
        node = connected_nodes[node_id]
        print(f"📊 Node {node_id} verification results - CPU: {cpu_result}, GPU: {gpu_result}")
        node.last_heartbeat = time.time()

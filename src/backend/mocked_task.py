import time
import threading
import numpy as np
import psutil
from GPUtil import getGPUs
import torch  # torch is now guaranteed to be available

def run_cpu_task(node_id: str, connected_nodes: dict):
    if node_id not in connected_nodes:
        return False

    print(f"🧪 Running CPU verification task on node: {node_id}")

    node = connected_nodes[node_id]
    usage = 0
    intensity = 100_000  # Start small
    max_intensity = 5_000_000
    target_usage = 1.0

    while usage < target_usage and intensity <= max_intensity:
        _ = sum(i * i for i in range(intensity))
        usage = psutil.cpu_percent(interval=0.1)
        print(f"  🔄 Tried intensity {intensity}, got usage {usage}%")
        if usage >= target_usage:
            break
        intensity += 100_000

    node.cpu_usage = usage
    node.cpu_verified = True
    node.cpu_benchmark = intensity

    print(f"✅ Node {node_id} CPU benchmark completed: {intensity} ops to reach {usage:.2f}% usage")
    return True

def run_gpu_task(node_id: str, connected_nodes: dict, torch_module=torch):
    if node_id not in connected_nodes or not torch_module.cuda.is_available():
        return False

    node = connected_nodes[node_id]
    gpus = node.capabilities.get("gpu", [])

    if not isinstance(gpus, list) or not gpus:
        print(f"⚠️ Node {node_id} has no usable GPUs.")
        return False

    print(f"🧪 Running GPU verification task on node: {node_id}")
    success_count = 0
    tensor_size = 100
    max_size = 3000
    usage = 0
    target_usage = 1.0

    for gpu_index, gpu in enumerate(gpus):
        gpu_name = gpu.get("name", "Unknown GPU")
        if "No GPU" in gpu_name or "Detection" in gpu_name:
            continue
        while usage < target_usage and tensor_size <= max_size:
            try:
                torch_module.cuda.set_device(gpu_index)
                a = torch_module.rand(tensor_size, tensor_size, device='cuda')
                b = torch_module.rand(tensor_size, tensor_size, device='cuda')
                c = torch_module.matmul(a, b)
                _ = c.sum().item()
                del a, b, c
                torch_module.cuda.empty_cache()

                live_gpus = getGPUs()
                if live_gpus:
                    usage = sum(g.load for g in live_gpus) / len(live_gpus) * 100
                    print(f"  🔄 Tensor {tensor_size}x{tensor_size} got usage: {usage:.2f}%")
                if usage < target_usage:
                    tensor_size += 100
            except Exception as e:
                print(f"⚠️ Error testing GPU {gpu_name} on node {node_id}: {e}")
                break

        if usage >= target_usage:
            success_count += 1
            break

    if success_count > 0:
        node.gpu_usage = usage
        node.gpu_verified = True
        node.gpu_benchmark = tensor_size
        print(f"✅ GPU benchmark done — {tensor_size}x{tensor_size} to reach {usage:.2f}%")
        return True

    return False

# 🚦 Triggered Task (both CPU and GPU)
def simulate_verification_task(node_id: str, connected_nodes: dict, torch_module=torch):
    if node_id not in connected_nodes:
        print(f"❌ Node {node_id} disconnected before verification")
        return

    cpu_result = run_cpu_task(node_id, connected_nodes)
    gpu_result = run_gpu_task(node_id, connected_nodes, torch_module)

    if node_id in connected_nodes:
        node = connected_nodes[node_id]
        print(f"📊 Node {node_id} verification results - CPU: {cpu_result}, GPU: {gpu_result}")

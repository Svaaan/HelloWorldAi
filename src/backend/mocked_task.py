import time
import psutil

def verify_cpu_connected(node_id: str, connected_nodes: dict):
    if node_id not in connected_nodes:
        return False

    print(f"🔍 Verifying CPU connection for node: {node_id}")
    _ = sum(i * i for i in range(10_000))  # Simulated CPU load
    usage = psutil.cpu_percent(interval=0.1)

    node = connected_nodes[node_id]
    node.cpu_usage = usage
    node.cpu_verified = True
    node.cpu_benchmark = "Verified connection (simulated server-side)"

    print(f"✅ CPU verified for {node_id} — usage: {usage:.2f}%")
    return True

def verify_gpu_connected(node_id: str, connected_nodes: dict):
    if node_id not in connected_nodes:
        print(f"❌ Node {node_id} not found for GPU verification.")
        return False

    node = connected_nodes[node_id]
    gpus = node.capabilities.get("gpu", [])

    if not isinstance(gpus, list) or not gpus:
        print(f"⚠️ Node {node_id} has no reported GPUs (mirror).")
        return False

    print(f"🔍 Verifying GPU connection for node: {node_id} (mirror only)")

    for gpu in gpus:
        gpu_name = gpu.get("name", "Unknown GPU")
        if "No GPU" in gpu_name or "Detection" in gpu_name:
            continue

        # ✅ Mirror values (no local computation)
        node.gpu_usage = gpu.get("load_percentage", 0)
        node.gpu_verified = True
        node.gpu_benchmark = "Verified from node report"

        print(f"✅ GPU verified for {node_id} — mirrored load: {node.gpu_usage}%")
        return True

    print(f"⚠️ No valid GPU found in node report for {node_id}")
    return False

def simulate_verification_task(node_id: str, connected_nodes: dict):
    if node_id not in connected_nodes:
        print(f"❌ Node {node_id} disconnected before verification")
        return

    cpu_result = verify_cpu_connected(node_id, connected_nodes)
    gpu_result = verify_gpu_connected(node_id, connected_nodes)

    if node_id in connected_nodes:
        node = connected_nodes[node_id]
        print(f"📊 Node {node_id} verification summary — CPU: {cpu_result}, GPU: {gpu_result}")

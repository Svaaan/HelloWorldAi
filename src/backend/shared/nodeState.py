# File: backend/nodeState.py
# Create a new shared module for common elements used by both files

import socket
import logging
import pynvml
from typing import Dict, Any

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Node info state (moved from node.py)
public_ip = socket.gethostbyname(socket.gethostname())
node_info = {
    "country": "Unknown",
    "connected": False,
    "accept_tasks": True,
}

# GPU info function (moved from node.py)
def get_gpu_info_list():
    try:
        pynvml.nvmlInit()
        gpu_info = []

        for i in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_info.append({
                "name": pynvml.nvmlDeviceGetName(handle),
                "total_memory": round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / 1024**2),
                "free_memory": round(pynvml.nvmlDeviceGetMemoryInfo(handle).free / 1024**2),
                "used_memory": round(pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**2),
                "load_percentage": pynvml.nvmlDeviceGetUtilizationRates(handle).gpu,
                "temperature": pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            })

        return gpu_info

    except pynvml.NVMLError as e:
        logger.warning(f"⚠️ NVML error: {e}")
        return []
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass

# === Payload Builder === (moved from node.py)
def build_node_payload(system_capabilities) -> Dict[str, Any]:
    total_flops = system_capabilities.get("total_gpu_tflops", 0)

    # Also update node_info in memory
    node_info["total_gpu_tflops"] = total_flops

    return {
        "ip": public_ip,
        "country": node_info["country"],
        "capabilities": system_capabilities,
        "isConnected": node_info["connected"],
        "isAvailable": node_info.get("isAvailable", False),
        "total_gpu_tflops": total_flops,  
        "public_key": node_info.get("public_key")  
    }
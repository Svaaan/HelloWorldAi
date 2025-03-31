# systemInfoService.py

import psutil
import platform
import GPUtil
import logging

logger = logging.getLogger(__name__)

def get_node_ip():
    import socket, os
    try:
        return os.getenv('NODE_HOSTNAME', socket.gethostname())
    except Exception as e:
        logger.warning(f"IP retrieval error: {e}")
        return "localhost"

def get_system_capabilities():
    try:
        cpu_info = psutil.cpu_freq()
        cpu = {
            "brand": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "max_freq": round(cpu_info.max, 2) if cpu_info else None,
            "min_freq": round(cpu_info.min, 2) if cpu_info else None,
            "current_freq": round(cpu_info.current, 2) if cpu_info else None
        }

        gpus = []
        try:
            detected_gpus = GPUtil.getGPUs()
            for gpu in detected_gpus:
                gpus.append({
                    "name": gpu.name,
                    "total_memory": gpu.memoryTotal,
                    "free_memory": gpu.memoryFree,
                    "used_memory": gpu.memoryUsed,
                    "load_percentage": round(gpu.load * 100, 2),
                    "temperature": gpu.temperature
                })
        except Exception as gpu_error:
            logger.error(f"GPU detection error: {gpu_error}")
            gpus = [{"name": "GPU Detection Limited"}]

        return {
            "cpu": cpu,
            "gpu": gpus if gpus else [{"name": "No GPU Detected"}]
        }

    except Exception as e:
        logger.error(f"System capabilities error: {e}")
        return {"error": "Limited system capabilities"}

# systemInfoService.py

import psutil
import platform
import pynvml
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
        # CPU info
        cpu_info = psutil.cpu_freq()
        cpu = {
            "brand": platform.processor(),
            "cores": psutil.cpu_count(logical=False),
            "threads": psutil.cpu_count(logical=True),
            "max_freq": round(cpu_info.max, 2) if cpu_info else None,
            "min_freq": round(cpu_info.min, 2) if cpu_info else None,
            "current_freq": round(cpu_info.current, 2) if cpu_info else None
        }

        # GPU info via pynvml
        gpus = []
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                gpus.append({
                    "name": name,
                    "total_memory": round(memory_info.total / 1024 ** 2),
                    "free_memory": round(memory_info.free / 1024 ** 2),
                    "used_memory": round(memory_info.used / 1024 ** 2),
                    "load_percentage": utilization.gpu,
                    "temperature": temperature
                })

        except pynvml.NVMLError as gpu_error:
            logger.error(f"GPU detection error: {gpu_error}")
            gpus = [{"name": "No GPU Detected"}]

        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

        return {
            "cpu": cpu,
            "gpu": gpus if gpus else [{"name": "No GPU Detected"}]
        }

    except Exception as e:
        logger.error(f"System capabilities error: {e}")
        return {"error": "Limited system capabilities"}

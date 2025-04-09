# backend/service/usage.py
import pynvml
from psutil import cpu_percent, virtual_memory
import logging

logger = logging.getLogger(__name__)

async def get_usage():
    try:
        cpu_usage = cpu_percent(interval=1)
        gpu_usage = 0

        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_usage = utilization.gpu
            pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"GPU usage retrieval failed: {e}")
            gpu_usage = "N/A"

        return {
            "cpu_usage": cpu_usage,
            "gpu_usage": gpu_usage,
            "memory_usage": virtual_memory().percent
        }

    except Exception as e:
        logger.error(f"Usage retrieval error: {e}")
        return {"error": "Failed to retrieve usage information"}

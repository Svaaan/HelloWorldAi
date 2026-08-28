# backend/service/usage.py

import pynvml
from psutil import cpu_percent, virtual_memory
import logging

logger = logging.getLogger(__name__)

async def get_usage():
    try:
        # interval=None measures since the previous call instead of sleeping.
        # interval=1 blocked this coroutine -- and therefore the node's whole
        # event loop -- for a full second per request, which delayed heartbeats
        # and task polling whenever the dashboard was open.
        # The very first call after start-up returns 0.0; every later one is real.
        cpu_usage = cpu_percent(interval=None)
        memory_usage = virtual_memory().percent

        gpu_data = []

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                gpu_data.append({
                    "index": i,
                    "gpu_usage": utilization.gpu,
                    "gpu_temperature": temperature,
                    "critical_temperature": 85  
                })

            pynvml.nvmlShutdown()

        except Exception as e:
            logger.warning(f"GPU usage/temperature retrieval failed: {e}")
            gpu_data.append({
                "error": str(e)
            })

        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "gpu_data": gpu_data
        }

    except Exception as e:
        logger.error(f"Usage retrieval error: {e}")
        return {"error": "Failed to retrieve usage information"}
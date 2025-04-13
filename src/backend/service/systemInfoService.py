# systemInfoService.py

import traceback
import psutil
import platform
import pynvml
import logging
import json
import os

logger = logging.getLogger(__name__)

# === Load GPU database once ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_DB_PATH = os.path.join(BASE_DIR, 'gpu-db.json')

try:
    with open(GPU_DB_PATH, 'r') as f:
        gpu_db = json.load(f)
except Exception as e:
    logger.error(f"Failed to load GPU database: {e}")
    gpu_db = []

# === Helper functions ===

def get_node_ip():
    import socket, os
    try:
        return os.getenv('NODE_HOSTNAME', socket.gethostname())
    except Exception as e:
        logger.warning(f"IP retrieval error: {e}")
        return "localhost"

def calculate_tflops(cores, clock_hz, flops_per_cycle=2):
    try:
        return round((cores * clock_hz * flops_per_cycle) / 1_000_000_000_000, 3)
    except Exception as e:
        logger.warning(f"TFLOPS calculation error: {e}")
        return 0

def get_cuda_cores(gpu_name: str):
    gpu_name_clean = gpu_name.lower().replace("nvidia", "").strip()

    for gpu in gpu_db:
        db_gpu_name = gpu['name'].lower()
        if gpu_name_clean in db_gpu_name or db_gpu_name in gpu_name_clean:
            return gpu.get('shaders')  # ✅ THIS IS THE FIX 🔥

    logger.warning(f"No CUDA cores found for GPU: '{gpu_name}'")
    return None



def get_system_capabilities():
    total_tflops = 0
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
                clock_mhz = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS)

                # ✅ Check if clock is suspiciously low, if so, use your scraped GPU database
                boost_clock_mhz = clock_mhz
                matching_gpu = next((gpu for gpu in gpu_db if gpu['name'].lower() in name.lower()), None)

                if clock_mhz < 500 and matching_gpu:
                    try:
                        gpu_clock_str = matching_gpu.get('gpu_clock', '0 MHz').replace(' MHz', '')
                        boost_clock_mhz = int(float(gpu_clock_str))
                        logger.info(f"🧩 Boosted GPU clock from database: {boost_clock_mhz} MHz (original: {clock_mhz} MHz)")
                    except Exception as e:
                        logger.warning(f"Error parsing boost clock from database: {e}")

                clock_hz = boost_clock_mhz * 1_000_000  # Convert MHz to Hz

                # Match CUDA cores
                cuda_cores = get_cuda_cores(name)
                if not cuda_cores:
                    logger.warning(f"No CUDA core count found for GPU: '{name}'")
                else:
                    logger.info(f"Found CUDA cores from database: {cuda_cores}")

                tflops = calculate_tflops(cuda_cores, clock_hz) if cuda_cores else 0
                total_tflops += tflops

                # Log details
                logger.info(f"Detected GPU: {name}, CUDA Cores: {cuda_cores or 'Unknown'}, Clock: {boost_clock_mhz} MHz, TFLOPS: {tflops}")

                gpus.append({
                    "name": name,
                    "total_memory": round(memory_info.total / 1024 ** 2),
                    "free_memory": round(memory_info.free / 1024 ** 2),
                    "used_memory": round(memory_info.used / 1024 ** 2),
                    "load_percentage": utilization.gpu,
                    "temperature": temperature,
                    "core_clock_mhz": boost_clock_mhz,
                    "cuda_cores": cuda_cores or "Unknown",
                    "theoretical_tflops": tflops
                })

        except pynvml.NVMLError as gpu_error:
            logger.error(f"GPU detection error: {gpu_error}")
            gpus = [{"name": "No GPU Detected"}]

        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception as shutdown_error:
                logger.warning(f"NVML shutdown error: {shutdown_error}")

        # Log total TFLOPS
        logger.info(f"Total theoretical GPU compute power: {total_tflops} TFLOPS")

        return {
            "cpu": cpu,
            "gpu": gpus if gpus else [{"name": "No GPU Detected"}],
            "total_gpu_tflops": total_tflops
        }

    except Exception as e:
        logger.error(f"System capabilities error: {e}")
        logger.error(traceback.format_exc())  # ✅ Full traceback for deep debugging
        return {"error": "Limited system capabilities"}


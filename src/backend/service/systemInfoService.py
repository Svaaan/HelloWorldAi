# systemInfoService.py

import traceback
import psutil
import platform
import pynvml
import logging

from backend.service.gpuMatch import find_gpu_entry, get_database_clock_mhz

logger = logging.getLogger(__name__)

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

                # Resolve the card once — core count and reference clock must come
                # from the same database entry, or the TFLOPS figure is meaningless.
                db_entry = find_gpu_entry(name)
                cuda_cores = db_entry.get('shaders') if db_entry else None

                # NVML reports the *current* clock, which is near-idle on a quiet GPU.
                # Fall back to the database's reference clock so an idle card is not
                # rated at zero.
                boost_clock_mhz = clock_mhz
                if clock_mhz < 500:
                    db_clock = get_database_clock_mhz(db_entry)
                    if db_clock:
                        boost_clock_mhz = db_clock
                        logger.info(f"🧩 Using database clock {boost_clock_mhz} MHz (NVML reported {clock_mhz} MHz)")

                clock_hz = boost_clock_mhz * 1_000_000  # Convert MHz to Hz

                if cuda_cores:
                    logger.info(f"Found CUDA cores from database: {cuda_cores}")
                else:
                    logger.warning(f"Could not resolve '{name}' in the GPU database — reporting 0 TFLOPS for it.")

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


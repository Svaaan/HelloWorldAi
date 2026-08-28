"""Decide when a GPU is too hot to keep working.

Why this exists
---------------
Nothing in the work path used to look at temperature. A submitted job ran flat
out regardless of how hot a contributor's card got, on hardware the contributor
still has to live with.

The card protects *itself* — an RTX 3070 throttles at 95 C and cuts power at
98 C — but by then it is already degraded and loud. The point of this module is
to stop earlier, of our own accord, and to be honest about why.

Thresholds come from the hardware rather than a guess. The old code hard-coded
85 C as "critical" for every GPU, which is wrong in both directions: too low for
a card that happily runs at 83, too high for one that throttles at 80.

    warn  = stop - WARN_MARGIN     shown in the UI, work continues
    stop  = min(gpu_max, slowdown - STOP_MARGIN)     work is paused

Both can be overridden with NODE_TEMP_WARN / NODE_TEMP_STOP for a contributor
who wants their card kept cooler than the hardware minimum requires.
"""

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# How far below the hardware slowdown point we stop, and how far below that we
# start warning.
STOP_MARGIN_C = int(os.getenv("NODE_TEMP_STOP_MARGIN", 5))
WARN_MARGIN_C = int(os.getenv("NODE_TEMP_WARN_MARGIN", 8))

# Used only when the driver will not tell us the real limits.
FALLBACK_SLOWDOWN_C = 90
FALLBACK_GPU_MAX_C = 88

STATE_OK = "ok"
STATE_WARN = "warn"
STATE_STOP = "stop"


def _env_int(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning(f"{name}={raw!r} is not a whole number; ignoring it.")
        return None


def limits_for(slowdown: Optional[int], gpu_max: Optional[int]) -> Dict[str, int]:
    """Work out warn/stop points from what the hardware reports."""
    slowdown = slowdown or FALLBACK_SLOWDOWN_C
    gpu_max = gpu_max or FALLBACK_GPU_MAX_C

    stop = min(gpu_max, slowdown - STOP_MARGIN_C)
    override_stop = _env_int("NODE_TEMP_STOP")
    if override_stop is not None:
        stop = override_stop

    warn = stop - WARN_MARGIN_C
    override_warn = _env_int("NODE_TEMP_WARN")
    if override_warn is not None:
        warn = override_warn

    # A warn point at or above the stop point would never fire.
    if warn >= stop:
        warn = stop - 1

    return {"warn": int(warn), "stop": int(stop),
            "slowdown": int(slowdown), "gpu_max": int(gpu_max)}


def classify(temperature: Optional[float], limits: Dict[str, int]) -> str:
    """ok / warn / stop for a single reading.

    An unknown temperature is treated as OK rather than as an emergency: a
    driver that stops reporting should not halt every job on the machine.
    """
    if temperature is None:
        return STATE_OK
    if temperature >= limits["stop"]:
        return STATE_STOP
    if temperature >= limits["warn"]:
        return STATE_WARN
    return STATE_OK


def worst(states: List[str]) -> str:
    """The most severe state in a list — one hot GPU is enough to stop."""
    if STATE_STOP in states:
        return STATE_STOP
    if STATE_WARN in states:
        return STATE_WARN
    return STATE_OK


def read_gpu_thermals() -> List[Dict[str, Any]]:
    """Per-GPU temperature and limits, straight from the driver."""
    try:
        import pynvml
    except ImportError:
        return []

    try:
        pynvml.nvmlInit()
    except Exception as e:
        logger.debug(f"NVML unavailable: {e}")
        return []

    out: List[Dict[str, Any]] = []
    try:
        for index in range(pynvml.nvmlDeviceGetCount()):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)

            def threshold(name):
                try:
                    return int(pynvml.nvmlDeviceGetTemperatureThreshold(
                        handle, getattr(pynvml, name)))
                except Exception:
                    return None

            temperature = int(pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU))
            limits = limits_for(
                threshold("NVML_TEMPERATURE_THRESHOLD_SLOWDOWN"),
                threshold("NVML_TEMPERATURE_THRESHOLD_GPU_MAX"),
            )

            out.append({
                "index": index,
                "name": pynvml.nvmlDeviceGetName(handle),
                "temperature": temperature,
                "state": classify(temperature, limits),
                **limits,
            })
    except Exception as e:
        logger.warning(f"Could not read GPU thermals: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

    return out


def thermal_status() -> Dict[str, Any]:
    """Machine-wide thermal state, for the UI and for the work loop."""
    gpus = read_gpu_thermals()
    if not gpus:
        return {"state": STATE_OK, "gpus": [], "hottest": None, "reason": None}

    state = worst([g["state"] for g in gpus])
    hottest = max(gpus, key=lambda g: g["temperature"])

    reason = None
    if state == STATE_STOP:
        reason = (f"{hottest['name']} reached {hottest['temperature']}C "
                  f"(limit {hottest['stop']}C)")
    elif state == STATE_WARN:
        reason = (f"{hottest['name']} is at {hottest['temperature']}C "
                  f"(warning from {hottest['warn']}C)")

    return {"state": state, "gpus": gpus, "hottest": hottest, "reason": reason}


class ThermalAbort(RuntimeError):
    """Raised to stop a running job because the hardware got too hot."""

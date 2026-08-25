"""Plan how to split one workload across a set of heterogeneous GPUs.

The problem this solves
-----------------------
PyTorch's DataParallel and DistributedDataParallel split a batch *evenly* across
devices. On identical GPUs that is optimal. On a mixed pool -- say an RTX 3070
next to a 3050 and a 1660 -- every device gets the same share, so every step
runs at the pace of the slowest card and the fastest one idles most of the time.
An even split over N devices delivers `N * slowest` throughput, no matter how
fast the other cards are.

Splitting the batch *proportionally* to each device's measured throughput lets
every device finish its share at the same moment, delivering `sum(throughput)`.

Everything here is deliberately pure Python with no torch dependency, so the
scheduling maths can be tested without a GPU. Measurement lives in
gpuBenchmark.py.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _usable(devices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Devices with a positive measured throughput."""
    return [d for d in devices if (d.get('measured_tflops') or 0) > 0]


def pool_summary(devices: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Report what this pool can actually deliver, evenly split vs proportionally.

    `pooled_tflops` is the honest headline number: the sum of measured
    throughput, which is only achievable when work is split proportionally.
    """
    usable = _usable(devices)

    if not usable:
        return {
            "device_count": 0,
            "pooled_tflops": 0.0,
            "even_split_tflops": 0.0,
            "speedup_vs_even_split": 1.0,
            "total_vram_mb": 0,
            "bottleneck": None,
        }

    throughputs = [float(d['measured_tflops']) for d in usable]
    pooled = sum(throughputs)
    slowest = min(throughputs)

    # An even split runs every device at the slowest device's pace.
    even_split = slowest * len(usable)

    slowest_device = min(usable, key=lambda d: float(d['measured_tflops']))

    return {
        "device_count": len(usable),
        "pooled_tflops": round(pooled, 3),
        "even_split_tflops": round(even_split, 3),
        "speedup_vs_even_split": round(pooled / even_split, 3) if even_split else 1.0,
        "total_vram_mb": sum(int(d.get('total_memory_mb') or 0) for d in usable),
        "bottleneck": slowest_device.get('name'),
    }


def plan_batch_split(
    devices: List[Dict[str, Any]],
    global_batch_size: int,
    memory_per_sample_mb: Optional[float] = None,
    memory_headroom: float = 0.9,
) -> List[Dict[str, Any]]:
    """Divide `global_batch_size` across devices in proportion to throughput.

    When `memory_per_sample_mb` is given, no device is handed more samples than
    its VRAM can hold (times `memory_headroom`); the overflow is redistributed
    to devices that still have room.

    Returns one entry per device that received work, each with `batch_size`.
    The batch sizes always sum to `global_batch_size`, or to the largest total
    the pool's memory can hold if that is smaller.
    """
    if global_batch_size <= 0:
        return []

    usable = _usable(devices)
    if not usable:
        logger.warning("No device reported a positive throughput; cannot plan a split.")
        return []

    total_throughput = sum(float(d['measured_tflops']) for d in usable)

    # Per-device ceiling from VRAM, if we know how much a sample costs.
    def capacity(device: Dict[str, Any]) -> Optional[int]:
        if not memory_per_sample_mb or memory_per_sample_mb <= 0:
            return None
        available = device.get('free_memory_mb') or device.get('total_memory_mb') or 0
        return max(0, int((float(available) * memory_headroom) / memory_per_sample_mb))

    # Largest-remainder apportionment, so the batch sizes sum exactly.
    exact = [
        (float(d['measured_tflops']) / total_throughput) * global_batch_size
        for d in usable
    ]
    sizes = [int(x) for x in exact]
    remainder = global_batch_size - sum(sizes)

    # Hand leftover samples to the devices with the largest fractional part,
    # breaking ties towards the faster device.
    order = sorted(
        range(len(usable)),
        key=lambda i: (exact[i] - sizes[i], float(usable[i]['measured_tflops'])),
        reverse=True,
    )
    for i in order[:remainder]:
        sizes[i] += 1

    # Apply VRAM ceilings and redistribute whatever does not fit.
    caps = [capacity(d) for d in usable]
    overflow = 0
    for i, cap in enumerate(caps):
        if cap is not None and sizes[i] > cap:
            overflow += sizes[i] - cap
            sizes[i] = cap

    while overflow > 0:
        # Fastest device with remaining headroom takes the next sample.
        room = [
            i for i, cap in enumerate(caps)
            if cap is None or sizes[i] < cap
        ]
        if not room:
            logger.warning(
                f"Pool VRAM cannot hold a batch of {global_batch_size}; "
                f"scheduling {global_batch_size - overflow} instead."
            )
            break
        room.sort(key=lambda i: float(usable[i]['measured_tflops']), reverse=True)
        sizes[room[0]] += 1
        overflow -= 1

    plan = []
    for device, size in zip(usable, sizes):
        if size <= 0:
            continue
        plan.append({
            "device_index": device.get('index'),
            "name": device.get('name'),
            "measured_tflops": device.get('measured_tflops'),
            "batch_size": size,
            "share": round(size / global_batch_size, 4),
        })

    return plan

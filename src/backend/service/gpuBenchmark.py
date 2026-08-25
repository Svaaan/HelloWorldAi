"""Measure what each GPU in this machine can actually do, rather than looking it up.

Why measure instead of reading a spec sheet
-------------------------------------------
`cuda_cores * clock * 2` is theoretical peak FP32. It is the wrong number for AI
work in two directions at once:

  * Real workloads reach roughly 30-60% of peak, so it overstates throughput.
  * Training runs on tensor cores in FP16/BF16/TF32, where an Ampere card does
    several times its FP32 figure, so it badly understates useful throughput.

It also requires correctly identifying the card, which needs a scraped database,
VRAM disambiguation for memory variants, and still fails on anything new.

Measuring sidesteps all of that: it works on any device, including ones no
database has heard of, and it produces the per-device weights poolPlanner needs
to split work proportionally across a mixed pool.

Run directly to see the pool report:

    python -m backend.service.gpuBenchmark
"""

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Measured results are cached: benchmarking takes seconds and must never run on
# every heartbeat.
_cache: Optional[List[Dict[str, Any]]] = None

WARMUP_ITERS = 5
MIN_ITERS = 10
TARGET_SECONDS = 1.5
MAX_MATRIX = 8192
MIN_MATRIX = 1024


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        logger.warning("torch is not installed - GPU benchmarking unavailable.")
        return None


def _matrix_size(free_bytes: int, bytes_per_element: int) -> int:
    """Largest power-of-two square matrix that comfortably fits in VRAM.

    Three matrices are live at once (A, B, result), and we stay well inside free
    memory so a benchmark never competes with real work for VRAM.
    """
    budget = max(free_bytes, 0) * 0.25
    n = MIN_MATRIX
    while n * 2 <= MAX_MATRIX and 3 * (n * 2) ** 2 * bytes_per_element <= budget:
        n *= 2
    return n


def benchmark_device(index: int, dtype_name: str = "float16") -> Optional[float]:
    """Sustained matmul throughput for one device, in TFLOPS. None on failure."""
    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        return None

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get(dtype_name)

    if dtype is None:
        raise ValueError("Unsupported dtype %r" % dtype_name)

    device = torch.device("cuda:%d" % index)

    try:
        free_bytes, _total = torch.cuda.mem_get_info(device)
        n = _matrix_size(free_bytes, torch.finfo(dtype).bits // 8)

        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)

        # Warm up: CUDA context creation and clock ramp would otherwise be timed.
        for _ in range(WARMUP_ITERS):
            a @ b
        torch.cuda.synchronize(device)

        # Time one iteration to decide how many to run.
        start = time.perf_counter()
        a @ b
        torch.cuda.synchronize(device)
        single = max(time.perf_counter() - start, 1e-6)
        iters = max(MIN_ITERS, int(TARGET_SECONDS / single))

        start = time.perf_counter()
        for _ in range(iters):
            a @ b
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start

        del a, b
        torch.cuda.empty_cache()

        if elapsed <= 0:
            return None

        # A dense N x N matmul is 2 * N^3 floating point operations.
        flops = 2.0 * (n ** 3) * iters
        return round(flops / elapsed / 1e12, 3)

    except Exception as e:
        logger.error("Benchmark failed on cuda:%d: %s" % (index, e))
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        return None


def benchmark_all(force: bool = False) -> List[Dict[str, Any]]:
    """Benchmark every visible GPU. Cached after the first run.

    Each entry is shaped for poolPlanner: it carries `measured_tflops` plus the
    memory figures the planner uses to cap per-device batch sizes.
    """
    global _cache
    if _cache is not None and not force:
        return _cache

    torch = _torch()
    if torch is None or not torch.cuda.is_available():
        logger.warning("No CUDA device available - nothing to benchmark.")
        _cache = []
        return _cache

    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free_bytes, total_bytes = torch.cuda.mem_get_info(torch.device("cuda:%d" % index))

        # FP16 is what mixed-precision training actually uses; FP32 is reported
        # alongside it for comparison against spec-sheet figures.
        fp16 = benchmark_device(index, "float16")
        fp32 = benchmark_device(index, "float32")

        devices.append({
            "index": index,
            "name": props.name,
            "compute_capability": "%d.%d" % (props.major, props.minor),
            "total_memory_mb": round(total_bytes / 1024 ** 2),
            "free_memory_mb": round(free_bytes / 1024 ** 2),
            "measured_tflops": fp16,          # the weight poolPlanner schedules on
            "measured_tflops_fp16": fp16,
            "measured_tflops_fp32": fp32,
        })
        logger.info(
            "Benchmarked cuda:%d %s: %s TFLOPS fp16, %s TFLOPS fp32"
            % (index, props.name, fp16, fp32)
        )

    _cache = devices
    return _cache


def _main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    torch = _torch()
    if torch is None:
        print("torch is not installed. Install it inside the node environment first.")
        return 1
    if not torch.cuda.is_available():
        print("No CUDA device visible to torch. Check the driver and the --gpus flag.")
        return 1

    from backend.service.poolPlanner import plan_batch_split, pool_summary

    devices = benchmark_all(force=True)
    if not devices:
        print("No GPUs found.")
        return 1

    print("")
    print("%-3s%-30s%9s%10s%10s" % ("#", "GPU", "VRAM", "fp16", "fp32"))
    print("-" * 62)
    for d in devices:
        print("%-3d%-30s%9s%10s%10s" % (
            d["index"],
            d["name"][:29],
            "%s MB" % d["total_memory_mb"],
            d["measured_tflops_fp16"],
            d["measured_tflops_fp32"],
        ))

    summary = pool_summary(devices)
    print("")
    print("  Pooled (proportional split) : %s TFLOPS" % summary["pooled_tflops"])
    print("  Even split (naive DDP)      : %s TFLOPS" % summary["even_split_tflops"])
    print("  Gain from proportional split: %sx" % summary["speedup_vs_even_split"])
    if summary["bottleneck"] and summary["speedup_vs_even_split"] > 1.05:
        print("  Slowest device              : %s" % summary["bottleneck"])
        print("                                (an even split runs everything at this pace)")

    print("")
    print("  Example split of a 256-sample batch:")
    for p in plan_batch_split(devices, 256):
        print("    cuda:%s %-27s%5d samples  (%.1f%%)" % (
            p["device_index"], p["name"][:26], p["batch_size"], p["share"] * 100))
    print("")
    return 0


if __name__ == '__main__':
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
    sys.exit(_main())

"""Tests for heterogeneous GPU work splitting.

Pure scheduling maths -- no torch and no GPU required.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from backend.service.poolPlanner import plan_batch_split, pool_summary  # noqa: E402


def gpu(index, name, tflops, total_mb=8192, free_mb=None):
    return {
        "index": index,
        "name": name,
        "measured_tflops": tflops,
        "total_memory_mb": total_mb,
        "free_memory_mb": total_mb if free_mb is None else free_mb,
    }


# A realistic mixed pool: a fast card, a midrange card and an older one.
MIXED = [
    gpu(0, "RTX 3070", 17.7, 8192),
    gpu(1, "RTX 3050 8 GB", 7.9, 8192),
    gpu(2, "GTX 1660", 4.3, 6144),
]


# --- batch splitting -----------------------------------------------------

def test_split_sums_exactly_to_the_global_batch():
    for batch in [1, 7, 8, 64, 100, 256, 1000]:
        plan = plan_batch_split(MIXED, batch)
        assert sum(p['batch_size'] for p in plan) == batch, batch


def test_faster_devices_receive_proportionally_more_work():
    plan = {p['name']: p['batch_size'] for p in plan_batch_split(MIXED, 300)}
    assert plan["RTX 3070"] > plan["RTX 3050 8 GB"] > plan["GTX 1660"]
    # 3070 is ~2.24x the 3050; its share should track that within rounding
    ratio = plan["RTX 3070"] / plan["RTX 3050 8 GB"]
    assert 2.0 < ratio < 2.5, ratio


def test_identical_devices_get_an_even_split():
    same = [gpu(i, f"RTX 3060 #{i}", 12.0) for i in range(4)]
    sizes = [p['batch_size'] for p in plan_batch_split(same, 128)]
    assert sizes == [32, 32, 32, 32], sizes


def test_single_device_receives_everything():
    plan = plan_batch_split([gpu(0, "RTX 3070", 17.7)], 64)
    assert len(plan) == 1 and plan[0]['batch_size'] == 64


def test_devices_without_measured_throughput_are_skipped():
    pool = MIXED + [gpu(3, "Unbenchmarked", 0), gpu(4, "Unknown", None)]
    plan = plan_batch_split(pool, 128)
    names = {p['name'] for p in plan}
    assert "Unbenchmarked" not in names and "Unknown" not in names
    assert sum(p['batch_size'] for p in plan) == 128


def test_empty_or_unusable_pool_returns_no_plan():
    assert plan_batch_split([], 64) == []
    assert plan_batch_split([gpu(0, "dead", 0)], 64) == []
    assert plan_batch_split(MIXED, 0) == []


def test_tiny_batch_goes_to_the_fastest_devices():
    plan = plan_batch_split(MIXED, 1)
    assert len(plan) == 1
    assert plan[0]['name'] == "RTX 3070"
    assert plan[0]['batch_size'] == 1


# --- VRAM limits ---------------------------------------------------------

def test_a_device_is_never_given_more_samples_than_its_vram_holds():
    # 512 MB per sample: the 6 GB card can hold ~10, the 8 GB cards ~14.
    plan = plan_batch_split(MIXED, 64, memory_per_sample_mb=512)
    by_name = {p['name']: p['batch_size'] for p in plan}
    assert by_name.get("GTX 1660", 0) <= int(6144 * 0.9 / 512)
    assert by_name.get("RTX 3070", 0) <= int(8192 * 0.9 / 512)


def test_overflow_from_a_small_card_is_redistributed_not_dropped():
    # By throughput the 1660 earns ~8 of 40 samples, but 2560 MB only holds 4.
    # The 4 it cannot take must move to the 3070, not vanish from the batch.
    pool = [
        gpu(0, "RTX 3070", 17.7, 24576),   # holds 43 samples
        gpu(1, "GTX 1660", 4.3, 2560),     # holds 4
    ]
    plan = plan_batch_split(pool, 40, memory_per_sample_mb=512)
    by_name = {p['name']: p['batch_size'] for p in plan}

    assert sum(by_name.values()) == 40, by_name
    assert by_name["GTX 1660"] == 4, by_name
    assert by_name["RTX 3070"] == 36, by_name


def test_batch_larger_than_total_pool_capacity_is_truncated_to_capacity():
    pool = [
        gpu(0, "RTX 3070", 17.7, 24576),   # 43 samples
        gpu(1, "GTX 1660", 4.3, 2560),     # 4 samples
    ]
    plan = plan_batch_split(pool, 48, memory_per_sample_mb=512)
    assert sum(p['batch_size'] for p in plan) == 47  # 43 + 4, all the pool can hold


def test_batch_is_reduced_when_the_pool_simply_cannot_hold_it():
    pool = [gpu(0, "RTX 3050 4 GB", 6.0, 4096)]
    plan = plan_batch_split(pool, 10_000, memory_per_sample_mb=512)
    scheduled = sum(p['batch_size'] for p in plan)
    assert 0 < scheduled < 10_000
    assert scheduled <= int(4096 * 0.9 / 512)


def test_free_memory_is_preferred_over_total_when_present():
    busy = [gpu(0, "RTX 3070", 17.7, total_mb=8192, free_mb=1024)]
    plan = plan_batch_split(busy, 64, memory_per_sample_mb=512)
    assert sum(p['batch_size'] for p in plan) <= int(1024 * 0.9 / 512)


# --- pool summary --------------------------------------------------------

def test_pooled_throughput_is_the_sum_of_measured_throughput():
    s = pool_summary(MIXED)
    assert abs(s['pooled_tflops'] - (17.7 + 7.9 + 4.3)) < 0.01
    assert s['device_count'] == 3


def test_even_split_is_capped_by_the_slowest_device():
    s = pool_summary(MIXED)
    # three devices all running at the 1660's pace
    assert abs(s['even_split_tflops'] - 3 * 4.3) < 0.01
    assert s['bottleneck'] == "GTX 1660"


def test_speedup_reflects_the_gain_from_proportional_splitting():
    s = pool_summary(MIXED)
    expected = (17.7 + 7.9 + 4.3) / (3 * 4.3)
    assert abs(s['speedup_vs_even_split'] - expected) < 0.01
    assert s['speedup_vs_even_split'] > 2.2


def test_identical_devices_gain_nothing_from_proportional_splitting():
    same = [gpu(i, f"RTX 3060 #{i}", 12.0) for i in range(3)]
    assert abs(pool_summary(same)['speedup_vs_even_split'] - 1.0) < 0.001


def test_empty_pool_summary_is_zeroed_not_an_error():
    s = pool_summary([])
    assert s['device_count'] == 0 and s['pooled_tflops'] == 0.0
    assert s['speedup_vs_even_split'] == 1.0


# --- standalone runner ---------------------------------------------------

def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as e:
            failed.append(name)
            print("  FAIL  %s: %s" % (name, e))
        except Exception as e:
            failed.append(name)
            print("  ERROR %s: %s: %s" % (name, type(e).__name__, e))
    print("")
    summary = "%d/%d passed" % (len(tests) - len(failed), len(tests))
    if failed:
        summary += " -- FAILED: %s" % ", ".join(failed)
    print(summary)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())

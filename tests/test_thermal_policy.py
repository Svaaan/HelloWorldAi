"""Tests for the thermal safety policy.

Pure threshold logic — no GPU and no NVML needed. The point is that a
contributor's card is protected before the hardware has to protect itself.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from backend.service.thermalPolicy import (  # noqa: E402
    FALLBACK_GPU_MAX_C,
    STATE_OK,
    STATE_STOP,
    STATE_WARN,
    classify,
    limits_for,
    thermal_status,
    worst,
)

# What an RTX 3070 actually reports.
RTX_3070 = {"slowdown": 95, "gpu_max": 93}


def clear_overrides():
    for name in ("NODE_TEMP_WARN", "NODE_TEMP_STOP"):
        os.environ.pop(name, None)


# --- deriving limits from hardware ---------------------------------------

def test_limits_come_from_the_cards_own_thresholds():
    clear_overrides()
    limits = limits_for(**RTX_3070)
    # stop below the point the card starts throttling itself
    assert limits["stop"] < RTX_3070["slowdown"], limits
    assert limits["stop"] <= RTX_3070["gpu_max"], limits
    assert limits["warn"] < limits["stop"], limits


def test_a_cooler_card_gets_lower_limits():
    clear_overrides()
    hot = limits_for(slowdown=95, gpu_max=93)
    cool = limits_for(slowdown=80, gpu_max=78)
    assert cool["stop"] < hot["stop"], (cool, hot)


def test_missing_driver_values_fall_back_to_something_safe():
    clear_overrides()
    limits = limits_for(None, None)
    assert limits["stop"] <= FALLBACK_GPU_MAX_C
    assert limits["warn"] < limits["stop"]


def test_a_contributor_can_demand_a_cooler_card():
    os.environ["NODE_TEMP_STOP"] = "70"
    os.environ["NODE_TEMP_WARN"] = "60"
    try:
        limits = limits_for(**RTX_3070)
        assert limits["stop"] == 70 and limits["warn"] == 60
    finally:
        clear_overrides()


def test_a_warn_point_above_the_stop_point_is_corrected():
    # Otherwise the warning could never fire before the stop.
    os.environ["NODE_TEMP_STOP"] = "80"
    os.environ["NODE_TEMP_WARN"] = "90"
    try:
        limits = limits_for(**RTX_3070)
        assert limits["warn"] < limits["stop"], limits
    finally:
        clear_overrides()


def test_nonsense_overrides_are_ignored_not_crashed_on():
    os.environ["NODE_TEMP_STOP"] = "hot"
    try:
        limits = limits_for(**RTX_3070)
        assert limits["stop"] < RTX_3070["slowdown"]
    finally:
        clear_overrides()


# --- classifying a reading ------------------------------------------------

def test_normal_load_temperature_is_fine():
    clear_overrides()
    limits = limits_for(**RTX_3070)
    # A 3070 sitting at 72 C under load is healthy and must not be stopped.
    assert classify(72, limits) == STATE_OK


def test_warning_fires_before_the_stop():
    clear_overrides()
    limits = limits_for(**RTX_3070)
    assert classify(limits["warn"], limits) == STATE_WARN
    assert classify(limits["stop"] - 1, limits) == STATE_WARN


def test_stop_fires_at_the_limit_and_above():
    clear_overrides()
    limits = limits_for(**RTX_3070)
    assert classify(limits["stop"], limits) == STATE_STOP
    assert classify(limits["stop"] + 10, limits) == STATE_STOP


def test_we_stop_before_the_hardware_has_to_throttle():
    """The whole point: act before the card starts protecting itself."""
    clear_overrides()
    limits = limits_for(**RTX_3070)
    assert classify(RTX_3070["slowdown"], limits) == STATE_STOP
    assert limits["stop"] < RTX_3070["slowdown"]


def test_an_unreadable_temperature_does_not_halt_everything():
    # A driver that stops reporting should not look like an emergency.
    clear_overrides()
    assert classify(None, limits_for(**RTX_3070)) == STATE_OK


# --- combining several GPUs ----------------------------------------------

def test_one_hot_gpu_stops_the_machine():
    assert worst([STATE_OK, STATE_OK, STATE_STOP]) == STATE_STOP
    assert worst([STATE_OK, STATE_WARN]) == STATE_WARN
    assert worst([STATE_OK, STATE_OK]) == STATE_OK
    assert worst([]) == STATE_OK


# --- status without a GPU -------------------------------------------------

def test_status_on_a_machine_with_no_gpu_is_ok_and_empty():
    status = thermal_status()
    assert status["state"] in (STATE_OK, STATE_WARN, STATE_STOP)
    assert isinstance(status["gpus"], list)
    if not status["gpus"]:
        assert status["state"] == STATE_OK
        assert status["reason"] is None


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

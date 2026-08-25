"""Tests for GPU name -> database entry resolution.

Runs standalone (`python tests/test_gpu_match.py`) with no test dependencies,
and is also collectable by pytest if it is ever added to the project.

The cases named "regression" below were all real mismatches produced by the
previous substring-based lookup.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from backend.service.gpuMatch import (  # noqa: E402
    find_gpu_entry,
    get_cuda_cores,
    get_database_clock_mhz,
    normalize,
)


def _name(gpu_name):
    entry = find_gpu_entry(gpu_name)
    return entry['name'] if entry else None


def _tflops(entry):
    """Theoretical FP32 peak, the same formula systemInfoService uses."""
    clock_hz = get_database_clock_mhz(entry) * 1_000_000
    return round(entry['shaders'] * clock_hz * 2 / 1_000_000_000_000, 3)


# --- normalisation -------------------------------------------------------

def test_normalize_strips_vendor_and_punctuation():
    assert normalize("NVIDIA GeForce RTX 4090") == ["geforce", "rtx", "4090"]


def test_normalize_maps_nvml_laptop_naming_to_database_mobile_naming():
    assert normalize("NVIDIA GeForce RTX 4090 Laptop GPU") == ["geforce", "rtx", "4090", "mobile"]


def test_normalize_handles_empty_and_none():
    assert normalize("") == []
    assert normalize(None) == []


# --- exact matches -------------------------------------------------------

def test_common_desktop_cards_resolve_exactly():
    for probe, expected in [
        ("NVIDIA GeForce RTX 4090", "GeForce RTX 4090"),
        ("NVIDIA GeForce RTX 3080", "GeForce RTX 3080"),
        ("NVIDIA GeForce GTX 1660", "GeForce GTX 1660"),
        ("NVIDIA A2", "A2"),
    ]:
        assert _name(probe) == expected, "%s -> %s" % (probe, _name(probe))


def test_ti_and_super_variants_are_distinct_from_their_base_cards():
    assert _name("NVIDIA GeForce GTX 1660 Ti") == "GeForce GTX 1660 Ti"
    assert _name("NVIDIA GeForce GTX 1660") == "GeForce GTX 1660"
    assert get_cuda_cores("NVIDIA GeForce GTX 1660 Ti") != get_cuda_cores("NVIDIA GeForce GTX 1660")


# --- regressions ---------------------------------------------------------

def test_regression_a4000_does_not_match_a400():
    # Substring matching resolved "rtx a4000" to the entry "RTX A400" (-76% TFLOPS).
    assert _name("NVIDIA RTX A4000") == "RTX A4000"


def test_regression_a2000_does_not_match_a2():
    # "a2" is a substring of "rtx a2000".
    assert _name("NVIDIA RTX A2000") == "RTX A2000"


def test_regression_2060_super_does_not_fall_back_to_2060():
    assert _name("NVIDIA GeForce RTX 2060 SUPER") == "GeForce RTX 2060 SUPER"


def test_regression_laptop_gpu_never_inherits_desktop_specs():
    # The worst case: a mobile 4090 advertised desktop 4090 compute (+182%).
    for probe in [
        "NVIDIA GeForce RTX 4090 Laptop GPU",
        "NVIDIA GeForce RTX 4060 Laptop GPU",
        "NVIDIA GeForce RTX 3080 Laptop GPU",
    ]:
        entry = find_gpu_entry(probe)
        if entry is None:
            continue  # no mobile entry in the database is an acceptable outcome

        assert "Mobile" in entry['name'], \
            "%s resolved to desktop part %s" % (probe, entry['name'])

        # Core count alone is not a discriminator: the mobile 4060 genuinely has
        # the same 3072 shaders as the desktop part. The clock is what differs,
        # so compare effective compute instead.
        desktop = find_gpu_entry(probe.replace(" Laptop GPU", ""))
        assert desktop is not None
        assert _tflops(entry) < _tflops(desktop), \
            "%s reports %s TFLOPS, not below desktop %s" % (
                probe, _tflops(entry), _tflops(desktop))


def test_desktop_card_never_resolves_to_a_mobile_entry():
    for probe in ["NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4060"]:
        entry = find_gpu_entry(probe)
        assert entry is not None and "Mobile" not in entry['name']


# --- safe failure --------------------------------------------------------

def test_unknown_gpu_returns_none_rather_than_a_wrong_guess():
    assert find_gpu_entry("Some Made Up Accelerator 9000") is None
    assert get_cuda_cores("Some Made Up Accelerator 9000") is None


def test_blank_input_returns_none():
    for probe in ["", "   ", None]:
        assert find_gpu_entry(probe) is None


def test_non_nvidia_names_do_not_match():
    assert find_gpu_entry("AMD Radeon RX 7900 XTX") is None
    assert find_gpu_entry("Intel Arc A770") is None


# --- clock parsing -------------------------------------------------------

def test_database_clock_parses_to_int_mhz():
    entry = find_gpu_entry("NVIDIA GeForce RTX 4090")
    clock = get_database_clock_mhz(entry)
    assert isinstance(clock, int) and 100 < clock < 5000, clock


def test_database_clock_handles_missing_and_malformed_values():
    assert get_database_clock_mhz(None) is None
    assert get_database_clock_mhz({'name': 'x', 'gpu_clock': 'unknown'}) is None
    assert get_database_clock_mhz({'name': 'x'}) is None


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

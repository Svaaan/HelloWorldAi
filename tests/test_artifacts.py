"""Tests for network payload serialisation.

The security tests here are the point of the module: datasets arrive from
strangers and trained weights come back from strangers, so a payload must never
be able to execute code while being loaded.
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

if HAVE_NUMPY:
    from backend.service.artifacts import (
        ArtifactError,
        MAX_ARTIFACT_BYTES,
        pack_arrays,
        pack_dataset,
        parse_csv_dataset,
        pack_state_dict,
        unpack_arrays,
        unpack_dataset,
        unpack_state_dict,
    )


def expect_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ArtifactError:
        return
    raise AssertionError("expected ArtifactError from %s" % getattr(fn, "__name__", fn))


# --- security ------------------------------------------------------------

def test_an_object_array_payload_is_refused():
    """The core defence: object arrays are pickled, and pickle executes code."""
    buffer = io.BytesIO()
    # allow_pickle defaults on for savez, so this writes a real pickle payload.
    np.savez(buffer, evil=np.array([{"payload": "arbitrary"}], dtype=object))
    expect_error(unpack_arrays, buffer.getvalue())


class Marker:
    """Module level so it is picklable -- a local class cannot be pickled at all,
    which would make the test pass for the wrong reason."""


def test_a_class_instance_payload_is_refused():
    buffer = io.BytesIO()
    np.savez(buffer, evil=np.array([Marker()], dtype=object))
    expect_error(unpack_arrays, buffer.getvalue())


def test_packing_an_object_array_is_refused_too():
    # Refuse in both directions, so we never emit something a peer must reject.
    expect_error(pack_arrays, {"evil": np.array([{"a": 1}], dtype=object)})


def test_a_mixed_archive_is_refused_whole():
    """One bad array poisons the archive; partial loads are not offered."""
    buffer = io.BytesIO()
    np.savez(buffer,
             good=np.arange(10, dtype=np.float32),
             evil=np.array([{"a": 1}], dtype=object))
    expect_error(unpack_arrays, buffer.getvalue())


def test_garbage_and_truncated_input_is_rejected_cleanly():
    for payload in [b"not an npz", b"PK\x03\x04truncated", b"\x00" * 64]:
        expect_error(unpack_arrays, payload)


def test_empty_and_wrong_typed_input_is_rejected():
    expect_error(unpack_arrays, b"")
    expect_error(unpack_arrays, "a string")
    expect_error(unpack_arrays, None)


def test_oversized_payloads_are_rejected_without_loading():
    expect_error(unpack_arrays, b"\x00" * (MAX_ARTIFACT_BYTES + 1))


def test_too_many_arrays_is_rejected():
    expect_error(pack_arrays, {"a%d" % i: np.zeros(1) for i in range(600)})


# --- round trips ---------------------------------------------------------

def test_numeric_arrays_survive_a_round_trip():
    original = {
        "floats": np.random.rand(20, 4).astype(np.float32),
        "ints": np.arange(12, dtype=np.int64),
        "bools": np.array([True, False, True]),
    }
    restored = unpack_arrays(pack_arrays(original))

    assert set(restored) == set(original)
    for name, array in original.items():
        assert restored[name].dtype == array.dtype, name
        assert np.array_equal(restored[name], array), name


def test_dataset_round_trip_preserves_values_and_shape():
    x = np.random.rand(50, 8).astype(np.float32)
    y = np.random.randint(0, 3, size=50).astype(np.int64)

    rx, ry = unpack_dataset(pack_dataset(x, y))
    assert rx.shape == x.shape and ry.shape == y.shape
    assert np.allclose(rx, x) and np.array_equal(ry, y)


def test_state_dict_round_trip():
    state = {"layer.weight": np.random.rand(4, 4).astype(np.float32),
             "layer.bias": np.zeros(4, dtype=np.float32)}
    restored = unpack_state_dict(pack_state_dict(state))
    assert set(restored) == set(state)
    assert np.allclose(restored["layer.weight"], state["layer.weight"])


def test_state_dict_accepts_torch_tensors():
    try:
        import torch
    except ImportError:
        return  # torch-free environments still exercise the numpy path above

    state = {"w": torch.randn(3, 3), "b": torch.zeros(3)}
    restored = unpack_state_dict(pack_state_dict(state))
    assert np.allclose(restored["w"], state["w"].numpy())
    assert restored["b"].shape == (3,)


# --- dataset validation --------------------------------------------------

def test_mismatched_feature_and_label_counts_are_rejected():
    expect_error(pack_dataset, np.zeros((10, 3)), np.zeros(9))


def test_empty_dataset_is_rejected():
    expect_error(pack_dataset, np.zeros((0, 3)), np.zeros(0))


def test_dataset_missing_a_required_array_is_rejected():
    payload = pack_arrays({"x": np.zeros((4, 2))})
    expect_error(unpack_dataset, payload)


def test_dataset_with_desynced_arrays_is_caught_on_load():
    # Bypasses pack_dataset's own check, mimicking a hostile or corrupt peer.
    payload = pack_arrays({"x": np.zeros((10, 2)), "y": np.zeros(4)})
    expect_error(unpack_dataset, payload)


def test_empty_artifact_is_rejected():
    expect_error(pack_arrays, {})


def test_non_string_array_names_are_rejected():
    expect_error(pack_arrays, {1: np.zeros(2)})


# --- CSV intake ----------------------------------------------------------

SIMPLE_CSV = "1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n"


def test_plain_numeric_csv_parses():
    x, y, classes, _names, _label = parse_csv_dataset(SIMPLE_CSV)
    assert x.shape == (3, 2)
    assert y.tolist() == [0, 1, 0]
    assert classes is None


def test_header_row_is_detected_and_skipped():
    x, y, _, _names, _label = parse_csv_dataset("width,height,label\n1,2,0\n3,4,1\n")
    assert x.shape == (2, 2)
    assert y.tolist() == [0, 1]


def test_text_labels_are_mapped_to_indices():
    x, y, classes, _names, _label = parse_csv_dataset("1,2,cat\n3,4,dog\n5,6,cat\n")
    assert classes == ["cat", "dog"]
    assert y.tolist() == [0, 1, 0]
    assert x.shape == (3, 2)


def test_semicolon_delimiter_is_handled():
    x, y, _, _names, _label = parse_csv_dataset("1;2;0\n3;4;1\n")
    assert x.shape == (2, 2) and y.tolist() == [0, 1]


def test_blank_lines_and_comments_are_ignored():
    x, _y, _c, _names, _label = parse_csv_dataset("# notes\n1,2,0\n\n3,4,1\n")
    assert x.shape == (2, 2)


def test_bytes_input_with_bom_is_accepted():
    x, _y, _c, _names, _label = parse_csv_dataset("\ufeff1,2,0\n3,4,1\n".encode("utf-8"))
    assert x.shape == (2, 2)


def test_ragged_rows_are_rejected_with_the_row_number():
    try:
        parse_csv_dataset("1,2,0\n3,4,5,1\n")
        assert False, "expected ArtifactError"
    except ArtifactError as e:
        assert "Row 2" in str(e), e


def test_non_numeric_feature_is_rejected():
    # The bad value must not be on row 1: a first row with non-numeric features
    # is indistinguishable from a header, and is skipped as one.
    try:
        parse_csv_dataset("1,2,0\n3,oops,1\n5,6,0\n")
        assert False, "expected ArtifactError"
    except ArtifactError as e:
        assert "non-numeric" in str(e), e
        assert "Row 2" in str(e), e


def test_a_non_numeric_first_row_is_treated_as_a_header():
    """Documents the ambiguity above: row 1 is a header when it looks like one."""
    x, y, _, _names, _label = parse_csv_dataset("a,b,label\n1,2,0\n3,4,1\n")
    assert x.shape == (2, 2) and y.tolist() == [0, 1]


def test_single_column_csv_is_rejected():
    expect_error(parse_csv_dataset, "1\n2\n3\n")


def test_empty_csv_is_rejected():
    expect_error(parse_csv_dataset, "")
    expect_error(parse_csv_dataset, "\n\n  \n")
    expect_error(parse_csv_dataset, "a,b,label\n")


def test_a_single_class_is_rejected_as_unlearnable():
    expect_error(parse_csv_dataset, "1,2,0\n3,4,0\n5,6,0\n")


def test_fractional_labels_are_rejected():
    expect_error(parse_csv_dataset, "1,2,0.5\n3,4,1.5\n")


def test_negative_class_indices_are_rejected():
    expect_error(parse_csv_dataset, "1,2,-1\n3,4,1\n")


def test_csv_output_round_trips_through_the_wire_format():
    x, y, _, _names, _label = parse_csv_dataset(SIMPLE_CSV)
    rx, ry = unpack_dataset(pack_dataset(x, y))
    assert np.allclose(rx, x) and np.array_equal(ry, y)


# --- standalone runner ---------------------------------------------------

def _main():
    if not HAVE_NUMPY:
        print("  SKIP  numpy is not installed - artifact tests not run")
        return 0

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

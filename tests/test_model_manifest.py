"""The description packed with a trained model.

A finished job used to hand back arrays named "0.weight", "2.weight",
"4.bias" -- positions in an nn.Sequential, usable only by someone who already
knew the module list they referred to. The submitter had to reverse-engineer
the model they had asked the network to build.

What matters here: the description survives the round trip, it is enough to
rebuild the network, it never breaks the no-pickle rule that makes these files
safe to accept from a stranger, and files written before manifests existed
still load.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from backend.service.artifacts import (  # noqa: E402
        MANIFEST_KEY,
        ArtifactError,
        pack_arrays,
        pack_state_dict,
        read_manifest,
        unpack_state_dict,
    )
    from backend.service.modelManifest import build_manifest  # noqa: E402


MLP_SPEC = {"architecture": "mlp", "input_dim": 3, "hidden_dim": 4,
            "depth": 2, "output_dim": 3}


def fake_mlp_state():
    """The tensors _build_mlp would produce for MLP_SPEC."""
    return {
        "0.weight": np.zeros((4, 3), dtype=np.float32),
        "0.bias": np.zeros(4, dtype=np.float32),
        "2.weight": np.zeros((4, 4), dtype=np.float32),
        "2.bias": np.zeros(4, dtype=np.float32),
        "4.weight": np.zeros((3, 4), dtype=np.float32),
        "4.bias": np.zeros(3, dtype=np.float32),
    }


# --- the description itself -----------------------------------------------

def test_the_module_list_matches_the_weights_that_were_trained():
    """The point: the layer list must line up with the state dict keys, or a
    loader rebuilds a model the weights do not fit."""
    manifest = build_manifest(MLP_SPEC, fake_mlp_state())
    linears = [m for m in manifest["modules"] if m["type"] == "Linear"]

    # Three Linear layers -> keys 0, 2, 4 (a ReLU sits between each pair).
    assert len(linears) == 3, manifest["modules"]
    assert [m["out_features"] for m in linears] == [4, 4, 3]
    assert linears[0]["in_features"] == 3


def test_the_described_shapes_match_the_actual_tensors():
    state = fake_mlp_state()
    manifest = build_manifest(MLP_SPEC, state)
    for name, described in manifest["tensors"].items():
        assert tuple(described["shape"]) == state[name].shape, name


def test_a_transformer_is_described_by_its_spec():
    spec = {"architecture": "transformer", "vocab_size": 128, "d_model": 32,
            "n_head": 2, "n_layer": 1, "seq_len": 16}
    manifest = build_manifest(spec, {"head.weight": np.zeros((128, 32), dtype=np.float32)})
    assert manifest["container"] == "TinyLM"
    assert manifest["spec"]["vocab_size"] == 128
    assert manifest["input"]["dtype"] == "int64"


def test_the_output_says_what_the_numbers_mean():
    # "logits, argmax for the label" is the difference between a usable model
    # and a tensor of unexplained floats.
    manifest = build_manifest(MLP_SPEC, fake_mlp_state())
    assert "logit" in manifest["output"]["meaning"].lower()


def test_class_names_are_carried_when_known():
    manifest = build_manifest(MLP_SPEC, fake_mlp_state(),
                              class_names=["setosa", "versicolor", "virginica"])
    assert manifest["class_names"][2] == "virginica"


def test_training_figures_are_summarised_not_dumped():
    metrics = {"steps": 300, "final_loss": 0.01, "parameters": 771,
               "devices": ["cuda:0"], "ran_hot": False, "secret": "x"}
    manifest = build_manifest(MLP_SPEC, fake_mlp_state(), metrics=metrics)
    assert manifest["training"]["steps"] == 300
    assert "secret" not in manifest["training"]
    assert "devices" not in manifest["training"]


# --- surviving the round trip ---------------------------------------------

def test_the_description_survives_pack_and_unpack():
    manifest = build_manifest(MLP_SPEC, fake_mlp_state(), model_name="iris-ish")
    blob = pack_state_dict(fake_mlp_state(), manifest)
    recovered = read_manifest(blob)

    assert recovered["model_name"] == "iris-ish"
    assert recovered["modules"] == manifest["modules"]
    assert recovered["spec"]["hidden_dim"] == 4


def test_the_description_is_not_returned_as_a_weight():
    """Otherwise load_state_dict is handed a tensor that is not a parameter."""
    blob = pack_state_dict(fake_mlp_state(), build_manifest(MLP_SPEC, fake_mlp_state()))
    weights = unpack_state_dict(blob)

    assert MANIFEST_KEY not in weights
    assert set(weights) == set(fake_mlp_state())


def test_weights_round_trip_unchanged_alongside_a_description():
    state = fake_mlp_state()
    state["0.weight"] = np.arange(12, dtype=np.float32).reshape(4, 3)

    blob = pack_state_dict(state, build_manifest(MLP_SPEC, state))
    recovered = unpack_state_dict(blob)

    assert np.array_equal(recovered["0.weight"], state["0.weight"])


# --- safety ---------------------------------------------------------------

def test_the_description_does_not_smuggle_in_an_object_array():
    """The whole file format exists to keep executable payloads out. Adding a
    description must not open a door: it is stored as plain bytes."""
    blob = pack_state_dict(fake_mlp_state(), {"architecture": "mlp"})

    with np.load(__import__("io").BytesIO(blob), allow_pickle=False) as archive:
        assert archive[MANIFEST_KEY].dtype.kind == "u"
        assert not archive[MANIFEST_KEY].dtype.hasobject


def test_a_caller_cannot_overwrite_the_description_with_a_fake_tensor():
    state = fake_mlp_state()
    state[MANIFEST_KEY] = np.zeros(4, dtype=np.float32)
    try:
        pack_state_dict(state, {"architecture": "mlp"})
    except ArtifactError:
        return
    raise AssertionError("a tensor was allowed to take the description's name")


def test_an_oversized_description_is_refused():
    huge = {"architecture": "mlp", "junk": "x" * 100_000}
    try:
        pack_state_dict(fake_mlp_state(), huge)
    except ArtifactError:
        return
    raise AssertionError("packed an unbounded description")


# --- older files ----------------------------------------------------------

def test_weights_written_before_manifests_still_load():
    """Files already in storage have no description; they must not break."""
    blob = pack_state_dict(fake_mlp_state())        # no manifest
    assert read_manifest(blob) is None
    assert set(unpack_state_dict(blob)) == set(fake_mlp_state())


def test_an_unreadable_description_does_not_block_the_weights():
    # Bytes that are not valid UTF-8 JSON, in the manifest slot.
    arrays = dict(fake_mlp_state())
    arrays[MANIFEST_KEY] = np.array([255, 254, 253], dtype=np.uint8)
    blob = pack_arrays(arrays)

    assert read_manifest(blob) is None            # reported as absent
    assert set(unpack_state_dict(blob)) == set(fake_mlp_state())


# --- standalone runner ---------------------------------------------------

def _main():
    if np is None:
        print("  SKIP  numpy is not installed - manifest tests not run")
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

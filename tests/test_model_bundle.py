"""The finished model, packaged the way models are normally packaged.

The download used to be a single `.npz` of numpy arrays with a description
hidden inside it. Safe, self-describing, and a format nothing else reads -- so
a developer could not open it in their own application, and the person who had
uploaded a spreadsheet could not open it at all.

A trained model normally arrives as weights in a binary file, a small JSON
saying how they are laid out, and whatever turns real things into the numbers
the model eats. That is what these check.

safetensors rather than `.pt` for the reason this project never used
`torch.save`: loading a pickle executes whatever is inside it, and these files
are meant to be passed between people.
"""

import io
import json
import os
import sys
import zipfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service.modelBundle import (        # noqa: E402
    CONFIG_NAME, LOADER_NAME, README_NAME, TOKENIZER_NAME, WEIGHTS_NAME,
    build_bundle,
)

WEIGHTS = {
    "0.weight": np.arange(12, dtype=np.float32).reshape(4, 3),
    "0.bias": np.zeros(4, dtype=np.float32),
}

CLASSIFIER = {
    "architecture": "mlp",
    "model_name": "gpu-state",
    "spec": {"architecture": "mlp", "input_dim": 3, "output_dim": 4},
    "input": {"shape": ["batch", 3], "dtype": "float32",
              "names": ["temperature_c", "power_watts", "fan_percent"]},
    "class_names": ["idle", "light", "throttling", "training"],
    "label_name": "state",
}

LANGUAGE = {
    "architecture": "transformer",
    "model_name": "writer",
    "spec": {"architecture": "transformer", "seq_len": 64, "vocab_size": 256},
    "input": {"shape": ["batch", 64], "dtype": "int64"},
    "tokenizer": {"kind": "bytes", "vocab_size": 256},
}


def opened(manifest, loader="# loader\n"):
    return zipfile.ZipFile(io.BytesIO(build_bundle(WEIGHTS, manifest, loader)))


# --- what is in the box --------------------------------------------------

def test_a_classifier_ships_weights_a_config_and_a_loader():
    names = opened(CLASSIFIER).namelist()

    assert WEIGHTS_NAME in names
    assert CONFIG_NAME in names
    assert LOADER_NAME in names
    assert README_NAME in names
    # Nothing to tokenise; a file explaining how would be a lie.
    assert TOKENIZER_NAME not in names


def test_a_language_model_also_ships_its_tokenizer():
    archive = opened(LANGUAGE)
    assert TOKENIZER_NAME in archive.namelist()

    tokenizer = json.loads(archive.read(TOKENIZER_NAME))
    assert tokenizer["kind"] == "bytes"
    assert tokenizer["encoding"] == "utf-8"
    # Not just "bytes" -- how to actually do it.
    assert "encode" in tokenizer and "decode" in tokenizer


def test_the_loader_is_optional_but_the_model_is_not():
    names = zipfile.ZipFile(io.BytesIO(build_bundle(WEIGHTS, CLASSIFIER, None))).namelist()

    assert LOADER_NAME not in names
    assert WEIGHTS_NAME in names and CONFIG_NAME in names


# --- the weights ---------------------------------------------------------

def test_the_weights_round_trip_exactly():
    safetensors = pytest.importorskip("safetensors.numpy")
    archive = opened(CLASSIFIER)

    back = safetensors.load(archive.read(WEIGHTS_NAME))

    assert sorted(back) == sorted(WEIGHTS)
    for name, value in WEIGHTS.items():
        assert np.array_equal(back[name], value)
        assert back[name].dtype == value.dtype


def test_the_description_travels_inside_the_weights_too():
    """config.json and the weights get separated.

    Somebody copies model.safetensors into their project and leaves the folder
    behind. A weights file that cannot say what it is becomes a bag of numbers
    again, so the description goes in both places.
    """
    pytest.importorskip("safetensors")
    from safetensors import safe_open

    archive = opened(CLASSIFIER)
    blob = archive.read(WEIGHTS_NAME)

    # The header is at the front of the file; read it the way the format says.
    import struct
    length = struct.unpack("<Q", blob[:8])[0]
    header = json.loads(blob[8:8 + length])
    carried = json.loads(header["__metadata__"]["manifest"])

    assert carried["class_names"] == CLASSIFIER["class_names"]
    assert carried["input"]["names"] == CLASSIFIER["input"]["names"]


def test_a_non_contiguous_array_is_still_written():
    # safetensors stores raw bytes and will not take a view; a transposed
    # array is the obvious way to arrive with one.
    safetensors = pytest.importorskip("safetensors.numpy")
    awkward = {"w": np.arange(6, dtype=np.float32).reshape(2, 3).T}

    blob = zipfile.ZipFile(
        io.BytesIO(build_bundle(awkward, CLASSIFIER, None))).read(WEIGHTS_NAME)

    assert np.array_equal(safetensors.load(blob)["w"], awkward["w"])


# --- the config ----------------------------------------------------------

def test_the_config_says_what_the_columns_are():
    config = json.loads(opened(CLASSIFIER).read(CONFIG_NAME))

    assert config["input"]["names"] == ["temperature_c", "power_watts", "fan_percent"]
    assert config["class_names"] == CLASSIFIER["class_names"]
    assert config["label_name"] == "state"
    assert config["bundle_version"] >= 1


def test_the_config_is_readable_by_a_person():
    text = opened(CLASSIFIER).read(CONFIG_NAME).decode("utf-8")

    # Indented and newline-terminated: somebody will open this in an editor.
    assert "\n  " in text
    assert text.endswith("\n")


def test_building_a_bundle_does_not_alter_the_manifest():
    before = json.dumps(CLASSIFIER, sort_keys=True)
    build_bundle(WEIGHTS, CLASSIFIER, None)

    assert json.dumps(CLASSIFIER, sort_keys=True) == before


# --- the readme ----------------------------------------------------------

def test_the_readme_lists_the_columns_in_order():
    text = opened(CLASSIFIER).read(README_NAME).decode("utf-8")

    order = [text.index(f"`{n}`") for n in CLASSIFIER["input"]["names"]]
    assert order == sorted(order)
    # And says why the order matters, since getting it wrong does not fail.
    assert "confidently and wrongly" in text


def test_the_readme_gives_a_language_model_the_right_command():
    text = opened(LANGUAGE).read(README_NAME).decode("utf-8")

    assert "--prompt" in text
    assert "--input" not in text


def test_the_readme_gives_a_classifier_the_right_command():
    text = opened(CLASSIFIER).read(README_NAME).decode("utf-8")

    assert "--input" in text
    assert "--prompt" not in text

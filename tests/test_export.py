"""Turning a downloaded model into a format other tools accept.

The download is an .npz -- this project's own arrangement of arrays. It is safe
(nothing in it can execute) and self-describing, and it is not a format any
other tool reads. Somebody who trains a model here and then wants to run it in
their own application had nowhere to go.

So the loader that already rebuilds the model can write it back out: as
TorchScript, which loads in PyTorch with no model code beside it, or as ONNX,
which runs almost anywhere without Python at all.

These import load_model.py the way its users do -- as a standalone file, with
nothing from this project on the path.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

LOADER = os.path.join(HERE, "..", "src", "frontend", "static", "scripts",
                      "load_model.py")


def loader():
    """load_model.py, imported as the standalone script it is meant to be."""
    spec = importlib.util.spec_from_file_location("load_model", LOADER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def a_model(architecture="mlp"):
    """A small trained-shaped model and the manifest that describes it."""
    pytest.importorskip("torch")
    from backend.service.modelManifest import build_manifest
    from backend.service.trainer import build_workload

    if architecture == "mlp":
        spec = {"architecture": "mlp", "input_dim": 6, "output_dim": 4,
                "hidden_dim": 16, "depth": 2}
    else:
        spec = {"architecture": "transformer", "d_model": 32, "n_head": 2,
                "n_layer": 2, "seq_len": 16, "vocab_size": 256}

    model = build_workload(spec)["factory"]()
    model.eval()

    state = {name: value.detach().numpy()
             for name, value in model.state_dict().items()}
    return model, build_manifest(spec, state, model_name="m")


# --- what shape the converter feeds it -----------------------------------

def test_the_example_input_matches_a_classifier():
    torch = pytest.importorskip("torch")
    _model, manifest = a_model("mlp")

    sample = loader().example_input(manifest)

    assert tuple(sample.shape) == (1, 6)
    assert sample.dtype == torch.float32


def test_the_example_input_matches_a_language_model():
    torch = pytest.importorskip("torch")
    _model, manifest = a_model("transformer")

    sample = loader().example_input(manifest)

    assert tuple(sample.shape) == (1, 16)
    # Token ids index an embedding table; a float would not.
    assert sample.dtype == torch.int64


# --- TorchScript, which needs nothing extra ------------------------------

def test_a_classifier_exports_and_still_agrees(tmp_path):
    torch = pytest.importorskip("torch")
    module = loader()
    model, manifest = a_model("mlp")

    out = tmp_path / "m.pt"
    assert module.export_model(model, manifest, str(out)) == "TorchScript"
    assert out.exists()

    x = torch.randn(5, 6)
    with torch.no_grad():
        assert torch.allclose(torch.jit.load(str(out))(x), model(x), atol=1e-6)


def test_a_language_model_exports_and_still_agrees(tmp_path):
    """The one that tracing could not do.

    A transformer encoder picks between code paths at runtime, so tracing
    records one of them and torch refuses the result -- "Graphs differed
    across invocations". Scripting compiles the branches instead.
    """
    torch = pytest.importorskip("torch")
    module = loader()
    model, manifest = a_model("transformer")

    out = tmp_path / "m.pt"
    assert module.export_model(model, manifest, str(out)) == "TorchScript"

    ids = torch.randint(0, 256, (3, 16))
    with torch.no_grad():
        assert torch.allclose(torch.jit.load(str(out))(ids), model(ids), atol=1e-5)


def test_a_saved_model_needs_no_model_code(tmp_path):
    # The whole point: a state dict still needs the class definition beside
    # it, which is the thing you are trying to leave behind.
    torch = pytest.importorskip("torch")
    module = loader()
    model, manifest = a_model("mlp")

    out = tmp_path / "m.pt"
    module.export_model(model, manifest, str(out))

    reloaded = torch.jit.load(str(out))          # no factory, no spec, no class
    assert reloaded(torch.zeros(1, 6)).shape == (1, 4)


def test_pth_is_accepted_as_well_as_pt(tmp_path):
    pytest.importorskip("torch")
    module = loader()
    model, manifest = a_model("mlp")

    assert module.export_model(model, manifest, str(tmp_path / "m.pth")) == "TorchScript"


def test_a_converted_model_that_disagrees_is_not_written(tmp_path, monkeypatch):
    """An exporter that quietly writes a wrong file is worse than none."""
    torch = pytest.importorskip("torch")
    module = loader()
    model, manifest = a_model("mlp")

    class Wrong:
        def __call__(self, x):
            return torch.zeros(x.shape[0], 4) + 99.0

    monkeypatch.setattr(torch.jit, "script", lambda _m: Wrong())

    out = tmp_path / "m.pt"
    with pytest.raises(SystemExit, match="does not agree"):
        module.export_model(model, manifest, str(out))

    assert not out.exists()


# --- and the failures worth reading --------------------------------------

def test_an_unknown_extension_says_what_it_takes(tmp_path):
    pytest.importorskip("torch")
    module = loader()
    model, manifest = a_model("mlp")

    with pytest.raises(SystemExit) as excinfo:
        module.export_model(model, manifest, str(tmp_path / "m.tflite"))

    message = str(excinfo.value)
    assert ".onnx" in message and ".pt" in message


def test_the_missing_package_is_named_from_the_error(tmp_path, monkeypatch):
    """Which package is missing depends on the version of torch.

    Older builds write ONNX through `onnx`; newer ones reach for `onnxscript`
    first. Naming a fixed one sent somebody to install the package they
    already had, so the name is read out of the error instead.
    """
    torch = pytest.importorskip("torch")
    module = loader()
    model, manifest = a_model("mlp")

    for raised, expected in (("No module named 'onnxscript'", "onnxscript"),
                             ("Module onnx is not installed!", "onnx")):
        def refuse(*_args, _msg=raised, **_kwargs):
            raise RuntimeError(_msg)

        monkeypatch.setattr(torch.onnx, "export", refuse)

        with pytest.raises(SystemExit) as excinfo:
            module.export_model(model, manifest, str(tmp_path / "m.onnx"))

        assert f"pip install {expected}" in str(excinfo.value)


def test_other_onnx_failures_are_not_disguised_as_a_missing_package(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    module = loader()
    model, manifest = a_model("mlp")

    def refuse(*_args, **_kwargs):
        raise RuntimeError("unsupported operator: aten::something")

    monkeypatch.setattr(torch.onnx, "export", refuse)

    with pytest.raises(SystemExit) as excinfo:
        module.export_model(model, manifest, str(tmp_path / "m.onnx"))

    assert "pip install" not in str(excinfo.value)
    assert "unsupported operator" in str(excinfo.value)


# --- ONNX itself, when the packages are there ----------------------------

def test_onnx_round_trips_a_classifier(tmp_path):
    pytest.importorskip("torch")
    onnxruntime = pytest.importorskip("onnxruntime")
    import torch

    module = loader()
    model, manifest = a_model("mlp")

    out = tmp_path / "m.onnx"
    try:
        assert module.export_model(model, manifest, str(out)) == "ONNX"
    except SystemExit as e:
        pytest.skip(str(e))

    x = np.random.default_rng(0).normal(size=(4, 6)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()

    session = onnxruntime.InferenceSession(str(out))
    got = session.run(None, {session.get_inputs()[0].name: x})[0]

    assert np.abs(expected - got).max() < 1e-4
    assert (expected.argmax(1) == got.argmax(1)).all()


def test_onnx_keeps_the_batch_size_free(tmp_path):
    pytest.importorskip("torch")
    onnxruntime = pytest.importorskip("onnxruntime")

    module = loader()
    model, manifest = a_model("mlp")
    out = tmp_path / "m.onnx"
    try:
        module.export_model(model, manifest, str(out))
    except SystemExit as e:
        pytest.skip(str(e))

    session = onnxruntime.InferenceSession(str(out))
    key = session.get_inputs()[0].name
    for rows in (1, 7, 33):
        x = np.zeros((rows, 6), dtype=np.float32)
        assert session.run(None, {key: x})[0].shape == (rows, 4)


def test_onnx_is_written_as_a_single_file(tmp_path):
    """Recent torch splits the weights into a sidecar `.onnx.data`.

    Sensible for a model too large for one file, and a trap for everyone
    else: copying the .onnx to another machine on its own leaves a graph with
    no weights in it, which loads and then fails.
    """
    pytest.importorskip("torch")
    module = loader()
    model, manifest = a_model("mlp")

    out = tmp_path / "m.onnx"
    try:
        module.export_model(model, manifest, str(out))
    except SystemExit as e:
        pytest.skip(str(e))

    assert out.exists()
    assert not (tmp_path / "m.onnx.data").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["m.onnx"]


def test_the_single_file_runs_on_its_own(tmp_path):
    pytest.importorskip("torch")
    onnxruntime = pytest.importorskip("onnxruntime")
    import shutil

    module = loader()
    model, manifest = a_model("mlp")

    out = tmp_path / "m.onnx"
    try:
        module.export_model(model, manifest, str(out))
    except SystemExit as e:
        pytest.skip(str(e))

    # Moved somewhere with nothing else in it, the way a person would.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    shutil.copy(out, elsewhere / "m.onnx")

    session = onnxruntime.InferenceSession(str(elsewhere / "m.onnx"))
    got = session.run(None, {session.get_inputs()[0].name:
                             np.zeros((2, 6), dtype=np.float32)})[0]
    assert got.shape == (2, 4)

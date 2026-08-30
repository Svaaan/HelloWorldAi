"""The finished model, packaged the way every other tool packages one.

What the download used to be
----------------------------
A single `.npz` of numpy arrays with a description hidden inside it. Safe, and
self-describing, and a format nothing else reads -- so the person who trained
the model could not open it in their own application, and the person who
uploaded a spreadsheet could not open it at all.

What a trained model normally looks like
----------------------------------------
Weights in a binary file, a small JSON saying how they are laid out, and
whatever is needed to turn real things into the numbers the model eats and
back again. That is what a Hugging Face repository is, and what anything that
loads models expects to find:

    model.safetensors    the weights
    config.json          the architecture, the dimensions, the names
    tokenizer.json       how text becomes token ids  (language models only)
    load_model.py        so it runs without hunting for anything
    README.md            the two commands

safetensors rather than `.pt` for the same reason this project never used
`torch.save`: loading a pickle executes whatever is in it, and these files are
meant to be passed between people. safetensors is a header and raw bytes, so
it cannot run anything -- the property that made `.npz` the right choice here,
in the format the rest of the world settled on.

This module needs numpy and safetensors, not torch. The coordinator serves the
download and has no business building models to do it.
"""

import io
import json
import zipfile
from typing import Any, Dict, Optional

import numpy as np

BUNDLE_VERSION = 1

WEIGHTS_NAME = "model.safetensors"
CONFIG_NAME = "config.json"
TOKENIZER_NAME = "tokenizer.json"
LOADER_NAME = "load_model.py"
README_NAME = "README.md"

# safetensors metadata is a flat map of string to string, so the description
# goes in as JSON under one key rather than spread across many.
MANIFEST_KEY = "manifest"


def _weights(state_dict: Dict[str, Any], manifest: Dict[str, Any]) -> bytes:
    """The tensors, with the description carried alongside them.

    Put in the file's own metadata as well as in config.json, because the two
    get separated: somebody copies the weights into their project and leaves
    the folder behind, and a weights file that cannot say what it is becomes a
    bag of numbers again.
    """
    from safetensors.numpy import save

    arrays = {}
    for name, value in state_dict.items():
        array = np.asarray(value)
        # safetensors stores raw bytes and will not take a view.
        arrays[name] = np.ascontiguousarray(array)

    return save(arrays, metadata={
        MANIFEST_KEY: json.dumps(manifest, separators=(",", ":"), default=str),
    })


def _tokenizer(manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """How text becomes numbers, when the model reads text at all."""
    tokenizer = manifest.get("tokenizer")
    if not tokenizer:
        return None

    described = dict(tokenizer)
    if described.get("kind") == "bytes":
        described.setdefault("encoding", "utf-8")
        described["encode"] = "list(text.encode('utf-8'))"
        described["decode"] = "bytes(ids).decode('utf-8', 'replace')"
    return described


def _readme(manifest: Dict[str, Any]) -> str:
    name = manifest.get("model_name") or "model"
    architecture = str(manifest.get("architecture", "mlp")).lower()
    classifier = architecture in ("mlp", "feedforward")

    lines = [
        f"# {name}",
        "",
        f"Trained on the HelloWorldAi network. Architecture: `{architecture}`.",
        "",
        "## What is in here",
        "",
        f"- `{WEIGHTS_NAME}` — the weights, in safetensors format. Loading it",
        "  cannot execute code, so it is safe to accept from someone else.",
        f"- `{CONFIG_NAME}` — how the weights are laid out, and what the inputs",
        "  and outputs mean.",
    ]

    if manifest.get("tokenizer"):
        lines.append(f"- `{TOKENIZER_NAME}` — how text becomes token ids.")

    lines += [
        f"- `{LOADER_NAME}` — rebuilds the model and runs it. Needs numpy and",
        "  torch, and nothing from the project that trained it.",
        "",
        "## Running it",
        "",
        "```bash",
        f"python {LOADER_NAME} {WEIGHTS_NAME}",
    ]

    names = (manifest.get("input") or {}).get("names")
    if classifier:
        example = " ".join("0" for _ in range(
            int((manifest.get("spec") or {}).get("input_dim", 3))))
        lines.append(f"python {LOADER_NAME} {WEIGHTS_NAME} --input {example}")
    else:
        lines.append(f'python {LOADER_NAME} {WEIGHTS_NAME} --prompt "Once upon a"')

    lines += ["```", ""]

    if classifier and names:
        lines += [
            "## The columns it reads, in this order",
            "",
            *[f"{i + 1}. `{n}`" for i, n in enumerate(names)],
            "",
            "Presenting them in a different order does not fail — it answers",
            "confidently and wrongly. The order above is the one it learned.",
            "",
        ]

    if manifest.get("class_names"):
        answer = manifest.get("label_name") or "answer"
        lines += [
            f"## What it answers ({answer})",
            "",
            *[f"- `{n}`" for n in manifest["class_names"]],
            "",
        ]

    lines += [
        "## Somewhere else",
        "",
        "To use this outside Python:",
        "",
        "```bash",
        f"python {LOADER_NAME} {WEIGHTS_NAME} --export {name}.onnx",
        "```",
        "",
        "ONNX runs in almost anything without Python. Writing it needs one",
        "extra package; the script says which one.",
        "",
    ]

    return "\n".join(lines)


def build_bundle(state_dict: Dict[str, Any], manifest: Dict[str, Any],
                 loader_source: Optional[str] = None) -> bytes:
    """Everything the model needs, in one zip."""
    described = dict(manifest)
    described["bundle_version"] = BUNDLE_VERSION

    buffer = io.BytesIO()
    # Deflated rather than stored: weights are floats and compress poorly, but
    # the JSON and the loader do not, and a download is a download.
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(WEIGHTS_NAME, _weights(state_dict, described))
        archive.writestr(CONFIG_NAME,
                         json.dumps(described, indent=2, default=str) + "\n")

        tokenizer = _tokenizer(described)
        if tokenizer:
            archive.writestr(TOKENIZER_NAME,
                             json.dumps(tokenizer, indent=2) + "\n")

        if loader_source:
            archive.writestr(LOADER_NAME, loader_source)

        archive.writestr(README_NAME, _readme(described))

    return buffer.getvalue()

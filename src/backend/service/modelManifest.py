"""A description of a trained model, packed alongside its weights.

Why this exists
---------------
A finished job used to hand back an .npz of arrays named "0.weight",
"2.weight", "4.bias". Those indices are positions in an nn.Sequential, so the
file is only usable by someone who already knows the exact module list they
refer to -- which is the thing the submitter asked the network to build in the
first place. They had to reverse-engineer their own model.

The manifest records what was built and how the weights map onto it, so the
download can be loaded by a short script that knows nothing about this repo.

It travels inside the weights file rather than beside it, because a description
that can be separated from what it describes eventually is.
"""

from typing import Any, Dict, List, Optional

MANIFEST_VERSION = 1


def _mlp_modules(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The flat module list _build_mlp produces, as plain data.

    Written out explicitly so a loader can rebuild the network without
    re-deriving it from depth/hidden_dim and hoping the two agree.
    """
    input_dim = int(spec.get("input_dim", 256))
    hidden_dim = int(spec.get("hidden_dim", 1024))
    depth = max(1, int(spec.get("depth", 4)))
    output_dim = int(spec.get("output_dim", 10))

    modules: List[Dict[str, Any]] = []
    dim = input_dim
    for _ in range(depth):
        modules.append({"type": "Linear", "in_features": dim, "out_features": hidden_dim})
        modules.append({"type": "ReLU"})
        dim = hidden_dim
    modules.append({"type": "Linear", "in_features": dim, "out_features": output_dim})
    return modules


def build_manifest(
    spec: Dict[str, Any],
    state_dict: Dict[str, Any],
    *,
    model_name: Optional[str] = None,
    class_names: Optional[List[str]] = None,
    tokenizer: Optional[str] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Describe a trained model well enough to rebuild and run it."""
    architecture = str(spec.get("architecture", "mlp")).lower()

    manifest: Dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "produced_by": "HelloWorldAi",
        "framework": "pytorch",
        "architecture": architecture,
        "spec": dict(spec),
        "tensors": {
            name: {
                "shape": [int(d) for d in getattr(value, "shape", [])],
                "dtype": str(getattr(value, "dtype", "float32")),
            }
            for name, value in state_dict.items()
        },
    }

    if model_name:
        manifest["model_name"] = model_name

    if architecture in ("mlp", "feedforward"):
        # A flat Sequential, so the module list is the whole story.
        manifest["container"] = "Sequential"
        manifest["modules"] = _mlp_modules(spec)
        manifest["input"] = {
            "shape": ["batch", int(spec.get("input_dim", 256))],
            "dtype": "float32",
        }
        manifest["output"] = {
            "shape": ["batch", int(spec.get("output_dim", 10))],
            "meaning": "unnormalised class scores (logits); argmax for the label",
        }
    else:
        # The transformer is a custom module, not a flat list; the loader
        # rebuilds it from the spec instead.
        manifest["container"] = "TinyLM"
        manifest["input"] = {
            "shape": ["batch", int(spec.get("seq_len", 64))],
            "dtype": "int64",
            "meaning": "token ids",
        }
        manifest["output"] = {
            "shape": ["batch", int(spec.get("seq_len", 64)), int(spec.get("vocab_size", 1024))],
            "meaning": "next-token logits at each position",
        }

    # What the integer labels in the training data actually meant. Without this
    # a classifier returns "2" and the owner has to remember what 2 was.
    if class_names:
        manifest["class_names"] = list(class_names)

    # And what the token ids meant. A language model that cannot say how to
    # turn text into ids -- and its answer back into text -- is a lookup table
    # for a code nobody wrote down.
    if tokenizer:
        manifest["tokenizer"] = {
            "kind": str(tokenizer),
            "vocab_size": int(spec.get("vocab_size", 256)),
        }
        if str(tokenizer) == "bytes":
            manifest["tokenizer"]["encoding"] = "utf-8"
            manifest["tokenizer"]["note"] = (
                "One id per byte: encode with text.encode('utf-8'), decode with "
                "bytes(ids).decode('utf-8', 'replace')."
            )

    if metrics:
        manifest["training"] = {
            k: metrics[k]
            for k in ("steps", "batch_size", "parameters", "final_loss",
                      "initial_loss", "dataset_rows")
            if k in metrics
        }

    return manifest

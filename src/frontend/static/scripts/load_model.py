#!/usr/bin/env python3
"""Load a model trained on the HelloWorldAi network and make a prediction.

    python load_model.py my-model.npz
    python load_model.py my-model.npz --input 1.2 3.4 5.6

The .npz you downloaded carries a description of itself, so this script
rebuilds the network and loads the weights without needing anything from the
project that trained it. Copy it wherever your model is; the only requirements
are numpy and torch.

It never calls pickle or torch.load on the file. The weights came off someone
else's machine, and both of those execute code while deserialising -- the file
is read as plain arrays, which cannot.
"""

import argparse
import json
import sys

import numpy as np

MANIFEST_KEY = "__manifest__"


def read_model_file(path):
    """Return (manifest, weights) from a downloaded .npz."""
    with np.load(path, allow_pickle=False) as archive:
        names = list(archive.files)

        if MANIFEST_KEY not in names:
            raise SystemExit(
                f"{path} has no model description. It may predate this format; "
                f"the arrays it contains are: {', '.join(names[:8])}"
            )

        manifest = json.loads(bytes(archive[MANIFEST_KEY]).decode("utf-8"))
        weights = {n: archive[n] for n in names if n != MANIFEST_KEY}

    return manifest, weights


def build_model(manifest):
    """Rebuild the network the manifest describes."""
    import torch
    from torch import nn

    architecture = manifest.get("architecture", "mlp")

    if manifest.get("container") == "Sequential":
        # The module list is written out in full, so this needs no knowledge
        # of how depth/hidden_dim were turned into layers.
        layers = []
        for module in manifest["modules"]:
            kind = module["type"]
            if kind == "Linear":
                layers.append(nn.Linear(module["in_features"], module["out_features"]))
            elif kind == "ReLU":
                layers.append(nn.ReLU())
            else:
                raise SystemExit(f"This loader does not know the layer type {kind!r}.")
        return nn.Sequential(*layers)

    if architecture in ("transformer", "tiny_lm", "tinylm", "lm"):
        spec = manifest["spec"]
        vocab_size = int(spec.get("vocab_size", 1024))
        d_model = int(spec.get("d_model", 256))
        n_head = int(spec.get("n_head", 4))
        n_layer = max(1, int(spec.get("n_layer", 2)))
        seq_len = int(spec.get("seq_len", 64))

        class TinyLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.token_embedding = nn.Embedding(vocab_size, d_model)
                self.position_embedding = nn.Embedding(seq_len, d_model)
                layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_head,
                    dim_feedforward=4 * d_model,
                    dropout=float(spec.get("dropout", 0.1)),
                    batch_first=True, norm_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
                self.head = nn.Linear(d_model, vocab_size)
                self.seq_len = seq_len

            def forward(self, tokens):
                _batch, length = tokens.shape
                positions = torch.arange(length, device=tokens.device)
                hidden = (self.token_embedding(tokens)
                          + self.position_embedding(positions))
                mask = torch.triu(
                    torch.full((length, length), float("-inf"), device=tokens.device),
                    diagonal=1,
                )
                return self.head(self.encoder(hidden, mask=mask))

        return TinyLM()

    raise SystemExit(f"This loader does not know the architecture {architecture!r}.")


def load_model(path):
    """The one call worth copying: file in, ready-to-use model out."""
    import torch

    manifest, weights = read_model_file(path)
    model = build_model(manifest)

    state = {name: torch.from_numpy(np.asarray(array)) for name, array in weights.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)

    if missing or unexpected:
        raise SystemExit(
            "The weights do not match the described model.\n"
            f"  missing: {list(missing)[:6]}\n"
            f"  unexpected: {list(unexpected)[:6]}"
        )

    model.eval()
    return model, manifest


def describe(manifest):
    print(f"  model        {manifest.get('model_name', '(unnamed)')}")
    print(f"  architecture {manifest.get('architecture')}")
    print(f"  input        {manifest.get('input', {}).get('shape')}")
    print(f"  output       {manifest.get('output', {}).get('meaning', '')}")

    training = manifest.get("training") or {}
    if training:
        print(f"  trained      {training.get('steps')} steps, "
              f"loss {training.get('initial_loss')} -> {training.get('final_loss')}")

    if manifest.get("class_names"):
        print(f"  classes      {', '.join(manifest['class_names'])}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="the .npz you downloaded")
    parser.add_argument("--input", nargs="*", type=float,
                        help="one row of features to run through the model")
    args = parser.parse_args()

    import torch

    model, manifest = load_model(args.model)
    print(f"\nLoaded {args.model}")
    describe(manifest)

    total = sum(p.numel() for p in model.parameters())
    print(f"  parameters   {total:,}")

    if args.input is None:
        print("\nPass --input <numbers> to run a prediction.")
        return 0

    architecture = manifest.get("architecture", "mlp")
    if architecture not in ("mlp", "feedforward"):
        print("\n--input only makes sense for the mlp architecture.")
        return 1

    expected = int(manifest["spec"].get("input_dim", 0))
    if len(args.input) != expected:
        raise SystemExit(f"This model expects {expected} features, got {len(args.input)}.")

    x = torch.tensor([args.input], dtype=torch.float32)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=-1)[0]

    index = int(torch.argmax(probabilities))
    names = manifest.get("class_names")
    label = names[index] if names and index < len(names) else str(index)

    print(f"\nPrediction: {label}  (confidence {probabilities[index]:.1%})")
    print("  all classes: " + ", ".join(
        f"{(names[i] if names and i < len(names) else i)}={p:.1%}"
        for i, p in enumerate(probabilities.tolist())
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())

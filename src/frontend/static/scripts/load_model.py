#!/usr/bin/env python3
"""Load a model trained on the HelloWorldAi network and make a prediction.

    python load_model.py my-model.npz
    python load_model.py my-model.npz --input 1.2 3.4 5.6
    python load_model.py my-model.npz --prompt "Once upon a"

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


def encode(manifest, text):
    """Turn text into the token ids this model was trained on."""
    kind = (manifest.get("tokenizer") or {}).get("kind")
    if kind != "bytes":
        raise SystemExit(
            f"This model's tokeniser is {kind!r}; this script only knows 'bytes'."
        )
    return list(text.encode("utf-8"))


def decode(manifest, ids):
    """And back again. 'replace' because a partial character is normal here:
    working one byte at a time, the model can stop part-way through a
    multi-byte one."""
    return bytes(int(i) & 0xFF for i in ids).decode("utf-8", "replace")


def generate(model, manifest, prompt, length=200, temperature=0.8, seed=None):
    """Continue the prompt one token at a time.

    Sampled rather than argmax: always taking the most likely token makes a
    small model repeat itself within a sentence or two. Temperature flattens
    the distribution -- lower is more predictable, higher more erratic.
    """
    import torch

    if seed is not None:
        torch.manual_seed(seed)

    seq_len = int(manifest["spec"].get("seq_len", 64))
    ids = encode(manifest, prompt) or [ord(" ")]

    with torch.no_grad():
        for _ in range(length):
            # The position embedding only goes up to seq_len, so the model can
            # only ever look back that far -- over its own output as well as
            # the prompt.
            window = ids[-seq_len:]
            logits = model(torch.tensor([window], dtype=torch.long))[0, -1]

            if temperature <= 0:
                ids.append(int(torch.argmax(logits)))
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                ids.append(int(torch.multinomial(probabilities, 1)))

    return decode(manifest, ids)


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

    tokenizer = manifest.get("tokenizer")
    if tokenizer:
        print(f"  tokeniser    {tokenizer.get('kind')} "
              f"({tokenizer.get('vocab_size')} ids)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="the .npz you downloaded")
    parser.add_argument("--input", nargs="*", type=float,
                        help="one row of features to run through the model")
    parser.add_argument("--prompt",
                        help="text for a language model to continue")
    parser.add_argument("--length", type=int, default=200,
                        help="how many tokens to generate after --prompt")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="0 is the most likely token every time; "
                             "higher is more varied")
    args = parser.parse_args()

    import torch

    model, manifest = load_model(args.model)
    print(f"\nLoaded {args.model}")
    describe(manifest)

    total = sum(p.numel() for p in model.parameters())
    print(f"  parameters   {total:,}")

    architecture = manifest.get("architecture", "mlp")
    is_classifier = architecture in ("mlp", "feedforward")

    if args.prompt is not None:
        if is_classifier:
            print("\n--prompt only makes sense for a language model; "
                  "this one classifies rows of numbers.")
            return 1
        print()
        print(generate(model, manifest, args.prompt,
                       length=args.length, temperature=args.temperature))
        return 0

    if args.input is None:
        print("\nPass --input <numbers> to run a prediction."
              if is_classifier else
              "\nPass --prompt \"some text\" to continue it.")
        return 0

    if not is_classifier:
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

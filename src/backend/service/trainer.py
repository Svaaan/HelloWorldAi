"""Real GPU training across a pool of heterogeneous local GPUs.

Why this is not just DistributedDataParallel
--------------------------------------------
DDP and DataParallel split a batch evenly. On a mixed pool every device then
runs at the pace of the slowest card, so an N-GPU pool delivers `N * slowest`.

This trainer gives each device a share of the batch sized to its *measured*
throughput (see poolPlanner), so every device finishes its share at roughly the
same moment and the pool delivers `sum(throughput)`.

Uneven shards stay mathematically correct because gradients are averaged
weighted by sample count:

    g_global = sum(n_i * g_i) / sum(n_i)

where g_i is the mean-reduced gradient over device i's n_i samples. That is
exactly the gradient of the mean loss over the whole batch, so a proportional
split trains identically to a single big batch -- only faster. This property is
covered directly by tests.

The equivalence is exact for a deterministic model. With stochastic layers such
as dropout it holds in expectation rather than step for step, because each
replica draws its own masks.
"""

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

# Pure data, no torch, so importing it here costs nothing.
from backend.service.modelManifest import build_manifest

logger = logging.getLogger(__name__)

# How often to look at GPU temperature while training. A read costs about a
# millisecond and a step costs far more, so this is effectively free, but there
# is no point checking more often than the card can heat up.
THERMAL_CHECK_EVERY = int(os.getenv("THERMAL_CHECK_EVERY", 5))


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


# --- workloads -----------------------------------------------------------

def _build_mlp(spec: Dict[str, Any]):
    torch = _torch()
    nn = torch.nn

    input_dim = int(spec.get("input_dim", 256))
    hidden_dim = int(spec.get("hidden_dim", 1024))
    depth = max(1, int(spec.get("depth", 4)))
    output_dim = int(spec.get("output_dim", 10))

    layers: List[Any] = []
    dim = input_dim
    for _ in range(depth):
        layers += [nn.Linear(dim, hidden_dim), nn.ReLU()]
        dim = hidden_dim
    layers.append(nn.Linear(dim, output_dim))
    return nn.Sequential(*layers)


def _build_tiny_lm(spec: Dict[str, Any]):
    """A small causal transformer -- the honest shape for `llm_training`."""
    torch = _torch()
    nn = torch.nn

    vocab_size = int(spec.get("vocab_size", 1024))
    d_model = int(spec.get("d_model", 256))
    n_head = int(spec.get("n_head", 4))
    n_layer = max(1, int(spec.get("n_layer", 2)))
    seq_len = int(spec.get("seq_len", 64))
    # Set dropout to 0 for a deterministic, exactly reproducible run.
    dropout = float(spec.get("dropout", 0.1))

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding = nn.Embedding(vocab_size, d_model)
            self.position_embedding = nn.Embedding(seq_len, d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dim_feedforward=4 * d_model,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layer)
            self.head = nn.Linear(d_model, vocab_size)
            self.seq_len = seq_len

        def forward(self, tokens):
            _batch, length = tokens.shape
            positions = torch.arange(length, device=tokens.device)
            hidden = self.token_embedding(tokens) + self.position_embedding(positions)
            # Causal mask: position t may only attend to <= t.
            mask = torch.triu(
                torch.full((length, length), float("-inf"), device=tokens.device),
                diagonal=1,
            )
            hidden = self.encoder(hidden, mask=mask)
            return self.head(hidden)

    return TinyLM()


def build_workload(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Return the model factory, sample generator, loss and cost model.

    `flops_per_sample` is the standard 6*params approximation for a training
    step (forward + backward), used to report achieved TFLOPS.
    """
    torch = _torch()
    if torch is None:
        raise RuntimeError("torch is not installed; cannot train.")

    architecture = str(spec.get("architecture", "mlp")).lower()

    if architecture in ("mlp", "feedforward"):
        input_dim = int(spec.get("input_dim", 256))
        output_dim = int(spec.get("output_dim", 10))

        def factory():
            return _build_mlp(spec)

        def make_batch(n, generator, device):
            x = torch.randn(n, input_dim, generator=generator).to(device)
            y = torch.randint(0, output_dim, (n,), generator=generator).to(device)
            return x, y

        loss_fn = torch.nn.functional.cross_entropy
        tokens_per_sample = 1

    elif architecture in ("transformer", "tiny_lm", "tinylm", "lm"):
        vocab_size = int(spec.get("vocab_size", 1024))
        seq_len = int(spec.get("seq_len", 64))

        def factory():
            return _build_tiny_lm(spec)

        def make_batch(n, generator, device):
            tokens = torch.randint(0, vocab_size, (n, seq_len + 1), generator=generator)
            return tokens[:, :-1].to(device), tokens[:, 1:].to(device)

        def loss_fn(logits, targets):
            return torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

        tokens_per_sample = seq_len

    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    probe = factory()
    param_count = sum(p.numel() for p in probe.parameters())
    del probe

    return {
        "factory": factory,
        "make_batch": make_batch,
        "loss_fn": loss_fn,
        "param_count": param_count,
        "flops_per_sample": 6.0 * param_count * tokens_per_sample,
    }


def dataset_sampler(features, labels, spec: Dict[str, Any]):
    """Return a make_batch that draws real rows instead of random noise.

    Arrays arrive as numpy from artifacts.unpack_dataset, so they are plain data
    -- nothing here can execute code from a submitted payload.
    """
    torch = _torch()

    x = torch.as_tensor(features)
    y = torch.as_tensor(labels)

    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"Dataset mismatch: {x.shape[0]} feature rows vs {y.shape[0]} labels"
        )
    if x.shape[0] == 0:
        raise ValueError("Dataset is empty.")

    architecture = str(spec.get("architecture", "mlp")).lower()
    if architecture in ("mlp", "feedforward"):
        # One label per row: a class index.
        feature_dtype = torch.float32
    else:
        # One label per *position*: the token that follows it. This used to
        # cast a 1-D label array to long and hand it over, which meant the
        # model produced seq_len predictions per row against a single target
        # and cross_entropy refused the batch -- so the transformer could only
        # ever run on the synthetic data it generated for itself.
        if x.dim() != 2 or y.shape != x.shape:
            raise ValueError(
                f"{architecture} training expects x and y to both be "
                f"(rows, sequence length) token ids; got x{tuple(x.shape)} "
                f"and y{tuple(y.shape)}. Upload a .txt file and the targets "
                f"are built for you."
            )
        feature_dtype = torch.long

    total = x.shape[0]

    # Widened per batch rather than over the whole dataset. Casting up front
    # turned a corpus of byte ids into an int64 array eight times its size and
    # held it there for the length of the job -- on someone else's machine, to
    # feed a few hundred rows at a time.
    def make_batch(n, generator, device):
        # Sample with replacement so a batch larger than the dataset still works.
        index = torch.randint(0, total, (n,), generator=generator)
        return (x[index].to(device=device, dtype=feature_dtype),
                y[index].to(device=device, dtype=torch.long))

    return make_batch, total


def infer_spec_from_dataset(features, labels, spec: Dict[str, Any]) -> Dict[str, Any]:
    """Fill in model dimensions the submitter did not state, from the data."""
    resolved = dict(spec)
    architecture = str(resolved.get("architecture", "mlp")).lower()

    if architecture in ("mlp", "feedforward"):
        if features.ndim != 2:
            raise ValueError(
                f"MLP training expects 2-D features (rows, columns); got shape {features.shape}"
            )
        resolved.setdefault("input_dim", int(features.shape[1]))
        if int(resolved["input_dim"]) != int(features.shape[1]):
            raise ValueError(
                f"model input_dim {resolved['input_dim']} does not match the "
                f"dataset's {features.shape[1]} columns"
            )
        classes = int(labels.max()) + 1 if labels.size else 1
        resolved["output_dim"] = max(int(resolved.get("output_dim", classes)), classes)

    else:
        # A language model's shape is a property of the data, not a choice:
        # the position embedding has to be exactly as long as the sequences,
        # and the token embedding has to cover every id that appears. Getting
        # either wrong is not a bad result, it is an index error part-way
        # through somebody else's GPU job -- or, on CUDA, a device-side assert
        # that takes the whole process with it.
        if features.ndim != 2 or labels.shape != features.shape:
            raise ValueError(
                f"{architecture} training expects x and y to both be 2-D "
                f"(rows, sequence length) token ids; got x{features.shape} "
                f"and y{labels.shape}."
            )
        if features.dtype.kind not in ("i", "u") or labels.dtype.kind not in ("i", "u"):
            raise ValueError(
                f"{architecture} training expects whole-number token ids; got "
                f"x as {features.dtype} and y as {labels.dtype}."
            )

        seq_len = int(features.shape[1])
        resolved.setdefault("seq_len", seq_len)
        if int(resolved["seq_len"]) != seq_len:
            raise ValueError(
                f"model seq_len {resolved['seq_len']} does not match the "
                f"dataset's sequence length of {seq_len}"
            )

        lowest = min(int(features.min()), int(labels.min())) if features.size else 0
        if lowest < 0:
            raise ValueError(f"Token ids cannot be negative; found {lowest}.")

        highest = max(int(features.max()), int(labels.max())) if features.size else 0
        # A vocabulary larger than what appears in the data is legitimate -- a
        # tokeniser has ids this sample happens not to use -- so take whichever
        # is bigger rather than overwriting the submitter's value.
        resolved["vocab_size"] = max(
            int(resolved.get("vocab_size", 0) or 0), highest + 1
        )

    return resolved


# How much generated text to send back with a finished job, and how much of
# the submitter's own data to prime it with. Small on purpose: this travels
# inside the task document and is read on a summary card, not a page.
SAMPLE_COUNT = 3
SAMPLE_PROMPT_TOKENS = 24
SAMPLE_NEW_TOKENS = 140

# Sampled rather than argmax. Always taking the most likely token makes a small
# model repeat one phrase, which looks like a bug rather than a weak model.
SAMPLE_TEMPERATURE = 0.8


def render_bytes(ids) -> str:
    """Token ids back to text. 'replace' because the model works a byte at a
    time and can stop part-way through a multi-byte character."""
    return bytes(int(i) & 0xFF for i in ids).decode("utf-8", "replace")


def continue_tokens(model, spec: Dict[str, Any], ids: List[int],
                    length: int = SAMPLE_NEW_TOKENS,
                    temperature: float = SAMPLE_TEMPERATURE) -> List[int]:
    """Extend a sequence of token ids, one token at a time.

    Sampled rather than argmax: always taking the most likely token makes a
    small model repeat one phrase, which reads like a bug rather than a weak
    model.
    """
    torch = _torch()
    seq_len = int(spec.get("seq_len", 64))
    ids = list(ids)

    with torch.no_grad():
        device = next(model.parameters()).device
        for _ in range(length):
            # The position embedding only reaches seq_len, so that is as far
            # back as the model can look -- over its own output as well.
            window = torch.tensor([ids[-seq_len:]], dtype=torch.long, device=device)
            logits = model(window)[0, -1]

            if temperature <= 0:
                ids.append(int(torch.argmax(logits)))
            else:
                probabilities = torch.softmax(logits / temperature, dim=-1)
                ids.append(int(torch.multinomial(probabilities, 1)))

    return ids


def sample_text(model, spec: Dict[str, Any], features, count: int = SAMPLE_COUNT
                ) -> List[Dict[str, str]]:
    """Continue a few real snippets, so the submitter can read the result.

    A finished language model used to arrive as a number and a download. What
    it actually writes is the only question anyone has about it, and answering
    it here costs a fraction of a second on a card that has just spent minutes
    training -- against a download, a Python install and a script for the
    person who asked.

    Priming with the submitter's own text rather than a fixed prompt keeps the
    comparison fair: it is continuing the kind of thing it was trained on.
    """
    torch = _torch()
    if torch is None or features is None or len(features) == 0:
        return []

    seq_len = int(spec.get("seq_len", 64))
    prompt_len = max(1, min(SAMPLE_PROMPT_TOKENS, seq_len - 1))

    # Spread across the data rather than taking the first rows, which on a
    # sorted or structured file would all look the same.
    total = int(features.shape[0])
    picks = [int(i * total / max(1, count)) % total for i in range(count)]

    was_training = model.training
    model.eval()

    samples: List[Dict[str, str]] = []
    try:
        for index in picks:
            ids = [int(v) for v in features[index][:prompt_len]]
            grown = continue_tokens(model, spec, ids)
            samples.append({
                "prompt": render_bytes(ids),
                "continuation": render_bytes(grown[prompt_len:]),
            })
    except Exception as e:
        # A sample is a courtesy. It must never be the reason a finished job
        # fails after the training has already been paid for.
        logger.warning(f"Could not generate samples: {e}")
        return []
    finally:
        if was_training:
            model.train()

    return samples


# --- proportional data-parallel trainer ----------------------------------

class PooledTrainer:
    """Replicates a model across devices and trains with uneven shards."""

    def __init__(self, factory: Callable[[], Any], devices: List[Any],
                 initial_state: Optional[Dict[str, Any]] = None):
        torch = _torch()
        self.torch = torch
        self.devices = devices

        self.replicas = []
        for device in devices:
            model = factory()
            # Carrying on from a previous run rather than from random noise.
            # strict=True: weights that do not fit this model are a mistake to
            # report, not something to paper over by loading half of them.
            if initial_state is not None:
                model.load_state_dict(
                    {name: torch.as_tensor(value)
                     for name, value in initial_state.items()},
                    strict=True,
                )
            self.replicas.append(model.to(device))

        self.master = self.replicas[0]
        self.primary = devices[0]

        # Every replica must start from identical weights.
        master_state = self.master.state_dict()
        for replica in self.replicas[1:]:
            replica.load_state_dict(master_state)

    def parameters(self):
        return self.master.parameters()

    def _sync_replicas(self):
        if len(self.replicas) == 1:
            return
        state = self.master.state_dict()
        for replica in self.replicas[1:]:
            replica.load_state_dict(state)

    def train_step(self, batch_x, batch_y, shard_sizes: List[int],
                   loss_fn, optimizer) -> float:
        """One optimiser step over a batch split unevenly across devices."""
        torch = self.torch
        total = sum(shard_sizes)
        if total <= 0:
            raise ValueError("Cannot train on an empty batch.")

        self._sync_replicas()

        # Carve the batch into per-device shards.
        chunks: List[Optional[Tuple[Any, Any]]] = []
        offset = 0
        for size in shard_sizes:
            if size <= 0:
                chunks.append(None)
                continue
            chunks.append((batch_x[offset:offset + size], batch_y[offset:offset + size]))
            offset += size

        weighted_losses: List[float] = [0.0] * len(self.replicas)

        def run_shard(i: int):
            chunk = chunks[i]
            if chunk is None:
                return
            model = self.replicas[i]
            device = self.devices[i]
            x, y = chunk

            model.zero_grad(set_to_none=True)
            output = model(x.to(device))
            loss = loss_fn(output, y.to(device))
            loss.backward()
            # Track the sum so the reported loss is the true batch mean.
            weighted_losses[i] = float(loss.detach().item()) * shard_sizes[i]

        # CUDA kernels release the GIL and run asynchronously, so threads give
        # real overlap here -- this is how DataParallel works internally too.
        if len(self.replicas) == 1:
            run_shard(0)
        else:
            with ThreadPoolExecutor(max_workers=len(self.replicas)) as pool:
                list(pool.map(run_shard, range(len(self.replicas))))

        # Weighted gradient reduction: sum(n_i * g_i) / sum(n_i).
        # Accumulated into fresh tensors because replicas[0] IS master, so
        # adding in place would double-count the primary device's gradient.
        accumulated = [
            torch.zeros_like(p, device=self.primary) for p in self.master.parameters()
        ]

        for i, model in enumerate(self.replicas):
            if shard_sizes[i] <= 0:
                continue
            weight = shard_sizes[i] / total
            for slot, param in zip(accumulated, model.parameters()):
                if param.grad is not None:
                    slot.add_(param.grad.detach().to(self.primary), alpha=weight)

        for param, grad in zip(self.master.parameters(), accumulated):
            param.grad = grad

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        return sum(weighted_losses) / total


# --- entry point ---------------------------------------------------------

def resolve_devices(plan: List[Dict[str, Any]], log: Callable[[str], None]):
    """Torch devices for a pool plan, falling back to CPU when there is no GPU."""
    torch = _torch()
    if torch is None:
        raise RuntimeError("torch is not installed; cannot train.")

    if plan and torch.cuda.is_available():
        return [torch.device(f"cuda:{part['device_index']}") for part in plan]

    log("No usable GPU found - training on CPU. This will be slow.")
    return [torch.device("cpu")]


def train(task_data: Dict[str, Any], log: Callable[[str], None],
          plan: Optional[List[Dict[str, Any]]] = None,
          batch_size: Optional[int] = None,
          dataset: Optional[Tuple[Any, Any]] = None,
          on_progress: Optional[Callable[[Dict[str, Any]], None]] = None,
          initial_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run a real training job. Returns metrics for the coordinator.

    `batch_size` overrides the hyperparameter when the caller has already
    sanitised it, so a malformed value cannot be parsed differently here.

    `dataset` is an (x, y) pair of numpy arrays from the submitter. Without one
    the run falls back to synthetic data, which is only useful for benchmarking.

    `on_progress` is called after every step with structured numbers, so the
    dashboard can draw a progress bar instead of scraping the log text.

    `initial_state` starts training from a previous run's weights instead of
    from random initialisation. Each run in a series used to begin again from
    nothing, so improving a model meant paying for everything it had already
    learned a second time.

    Returns {"metrics": {...}, "state_dict": {...}} where state_dict holds the
    trained weights as numpy arrays, ready to hand back to the submitter.
    """
    torch = _torch()
    if torch is None:
        raise RuntimeError("torch is not installed; cannot train.")

    hyperparameters = task_data.get("hyperparameters") or {}
    spec = task_data.get("model_spec") or hyperparameters.get("model_spec") or {}

    def _positive_int(value, fallback):
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return fallback
        return parsed if parsed > 0 else fallback

    steps = _positive_int(hyperparameters.get("steps", 20), 20)
    if batch_size is None:
        batch_size = _positive_int(hyperparameters.get("batch_size", 64), 64)
    batch_size = max(1, int(batch_size))

    try:
        learning_rate = float(hyperparameters.get("learning_rate", 1e-3))
    except (TypeError, ValueError):
        learning_rate = 1e-3
    seed = _positive_int(hyperparameters.get("seed", 0), 0) if hyperparameters.get("seed") else 0

    sampler = None
    dataset_rows = None
    if dataset is not None:
        features, labels = dataset
        spec = infer_spec_from_dataset(features, labels, spec)

    workload = build_workload(spec)

    if dataset is not None:
        features, labels = dataset
        sampler, dataset_rows = dataset_sampler(features, labels, spec)
        if str(spec.get("architecture", "mlp")).lower() in ("mlp", "feedforward"):
            log(f"Training on the submitted dataset: {dataset_rows:,} rows")
        else:
            log(f"Training on the submitted dataset: {dataset_rows:,} sequences "
                f"of {int(spec.get('seq_len', 0))} tokens "
                f"({dataset_rows * int(spec.get('seq_len', 0)):,} tokens), "
                f"vocabulary {int(spec.get('vocab_size', 0))}")
    else:
        log("No dataset supplied - training on synthetic data (benchmark only).")

    devices = resolve_devices(plan, log)

    # Shard sizes come from the pool plan; a single device takes everything.
    if plan and len(devices) == len(plan):
        shard_sizes = [part["batch_size"] for part in plan]
    else:
        shard_sizes = [batch_size]

    effective_batch = sum(shard_sizes)

    log(f"Model: {spec.get('architecture', 'mlp')} "
        f"({workload['param_count']:,} parameters)")
    log(f"Training {steps} steps at batch {effective_batch} "
        f"across {len(devices)} device(s)")

    if initial_state is not None:
        log(f"Continuing from a model that was already trained "
            f"({len(initial_state)} tensors).")

    try:
        trainer = PooledTrainer(workload["factory"], devices, initial_state)
    except (RuntimeError, KeyError) as e:
        # A shape that does not fit is worth naming: the submitter changed the
        # model between runs and the old weights cannot be carried over.
        raise ValueError(
            f"The weights being continued from do not fit this model: {e}"
        )

    optimizer = torch.optim.Adam(trainer.parameters(), lr=learning_rate)

    generator = torch.Generator().manual_seed(seed)
    first_loss = None
    last_loss = None

    from backend.service.thermalPolicy import (
        STATE_STOP, STATE_WARN, ThermalAbort, thermal_status,
    )

    make_batch = sampler or workload["make_batch"]
    warned_hot = False
    stopped_early = None

    start = time.perf_counter()
    for step in range(1, steps + 1):
        # Stop of our own accord before the card has to throttle itself. This
        # is somebody else's hardware.
        if step % THERMAL_CHECK_EVERY == 0:
            status = thermal_status()
            if status["state"] == STATE_STOP:
                stopped_early = status["reason"]
                log(f"Stopping early — {status['reason']}")
                raise ThermalAbort(status["reason"])
            if status["state"] == STATE_WARN and not warned_hot:
                log(f"Running hot — {status['reason']}")
                warned_hot = True

        batch_x, batch_y = make_batch(
            effective_batch, generator, torch.device("cpu")
        )
        last_loss = trainer.train_step(
            batch_x, batch_y, shard_sizes, workload["loss_fn"], optimizer
        )
        if first_loss is None:
            first_loss = last_loss
        if on_progress:
            on_progress({
                "step": step,
                "steps": steps,
                "loss": round(last_loss, 5),
                "initial_loss": round(first_loss, 5) if first_loss is not None else None,
                "elapsed_s": round(time.perf_counter() - start, 1),
            })

        if step == 1 or step % max(1, steps // 4) == 0:
            log(f"  step {step}/{steps}  loss {last_loss:.4f}")

    if devices[0].type == "cuda":
        torch.cuda.synchronize()
    elapsed = max(time.perf_counter() - start, 1e-9)

    samples = steps * effective_batch
    achieved_tflops = (workload["flops_per_sample"] * samples) / elapsed / 1e12

    log(f"Done in {elapsed:.1f}s - {samples / elapsed:.0f} samples/s, "
        f"{achieved_tflops:.2f} TFLOPS achieved")
    log(f"Loss {first_loss:.4f} -> {last_loss:.4f}")

    # Before the weights are copied out, while the model is still on the card.
    # Not `samples` -- that name is already the throughput count a few lines
    # down, and shadowing it turned a finished job into a TypeError after the
    # training had been paid for.
    writing = []
    if dataset is not None and str(spec.get("architecture", "mlp")).lower() \
            not in ("mlp", "feedforward"):
        writing = sample_text(trainer.master, spec, dataset[0])
        if writing:
            log("Sample of what it writes now:")
            log(f"  {writing[0]['prompt']}{writing[0]['continuation'][:120]}")

    state_dict = {
        name: tensor.detach().cpu().numpy()
        for name, tensor in trainer.master.state_dict().items()
    }

    metrics = {
        "steps": steps,
        "batch_size": effective_batch,
        "parameters": workload["param_count"],
        "devices": [str(d) for d in devices],
        "shard_sizes": shard_sizes,
        "seconds": round(elapsed, 3),
        "samples_per_second": round(samples / elapsed, 1),
        "achieved_tflops": round(achieved_tflops, 6),
        "achieved_gflops": round(achieved_tflops * 1000, 3),
        "initial_loss": round(first_loss, 5) if first_loss is not None else None,
        "final_loss": round(last_loss, 5) if last_loss is not None else None,
        "dataset_rows": dataset_rows,
        "synthetic_data": dataset is None,
        "ran_hot": warned_hot,
        "stopped_early": stopped_early,
        "warm_started": initial_state is not None,
    }

    # Kept out of the manifest: build_manifest picks named keys, and a model
    # file should carry a description of itself rather than a writing sample.
    if writing:
        metrics["samples"] = writing

    # Packed with the weights so the submitter can rebuild and run the model
    # without knowing anything about this codebase.
    # What the numbers in the dataset stood for -- class names for a
    # classifier, the tokeniser for a language model. Without it the download
    # answers "2" and its owner has to remember what 2 was, or hands back token
    # ids with no way to turn them into text.
    dataset_info = task_data.get("dataset_info") or {}

    manifest = build_manifest(
        spec,
        state_dict,
        model_name=task_data.get("model_name"),
        class_names=dataset_info.get("class_names"),
        tokenizer=dataset_info.get("tokenizer"),
        metrics=metrics,
    )

    return {"metrics": metrics, "state_dict": state_dict, "manifest": manifest}

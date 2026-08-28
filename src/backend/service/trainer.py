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
        x = x.float()
        y = y.long()
    else:
        x = x.long()
        y = y.long()

    total = x.shape[0]

    def make_batch(n, generator, device):
        # Sample with replacement so a batch larger than the dataset still works.
        index = torch.randint(0, total, (n,), generator=generator)
        return x[index].to(device), y[index].to(device)

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

    return resolved


# --- proportional data-parallel trainer ----------------------------------

class PooledTrainer:
    """Replicates a model across devices and trains with uneven shards."""

    def __init__(self, factory: Callable[[], Any], devices: List[Any]):
        torch = _torch()
        self.torch = torch
        self.devices = devices

        self.replicas = []
        for device in devices:
            self.replicas.append(factory().to(device))

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
          on_progress: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Run a real training job. Returns metrics for the coordinator.

    `batch_size` overrides the hyperparameter when the caller has already
    sanitised it, so a malformed value cannot be parsed differently here.

    `dataset` is an (x, y) pair of numpy arrays from the submitter. Without one
    the run falls back to synthetic data, which is only useful for benchmarking.

    `on_progress` is called after every step with structured numbers, so the
    dashboard can draw a progress bar instead of scraping the log text.

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
        log(f"Training on the submitted dataset: {dataset_rows:,} rows")
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

    trainer = PooledTrainer(workload["factory"], devices)
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
    }

    return {"metrics": metrics, "state_dict": state_dict}

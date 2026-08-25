"""Check whether returned weights are the product of real training.

The problem
-----------
A contributor is paid (in reputation, or eventually money) for compute nobody
watched. Nothing stops them from claiming a job, sleeping, and returning random
numbers with an invented loss curve.

What is actually achievable
---------------------------
Proving that a specific number of FLOPs were executed is hard: exact replay
across different GPUs is not reproducible (different architectures and kernel
choices give different floating point results), so any bit-exact check would
reject honest nodes.

What *is* cheap and decisive is checking the deliverable rather than the
process:

  1. The coordinator holds back a slice of the dataset before sending the rest.
     The node never sees it, so it cannot tune against it. Scoring the returned
     weights on that slice is a single forward pass -- orders of magnitude
     cheaper than the training it verifies.

  2. The floor for "did it learn" is an *untrained* model of the same
     architecture, not blind guessing. This matters: on well-separated data a
     randomly initialised network scores far above the majority class (72% vs
     44% in one measured case), so comparing against guessing would wave a
     fabricated result straight through.

  3. Structural checks catch the lazy frauds outright: NaN weights, weights
     identical to their initialisation, wrong shapes.

  4. The claimed training loss is recomputed on data the node *did* see. A
     mismatch there cannot be explained by overfitting, because it is the same
     data.

Honest limits, stated plainly:

  * This verifies that a useful model came back, not that a particular amount of
    compute happened. A node that trains well using less compute than expected
    passes -- which is arguably correct.
  * A determined attacker who actually trains a small model cheaply and returns
    it will pass. That raises the cost of fraud above the cost of honest work,
    which is the practical goal, not an absolute guarantee.
  * Nothing here defends against a node that returns a *good* model trained on
    the wrong objective. Only the submitter can judge that.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# A training loss is a running average over batches, so allow real slack before
# calling a claim dishonest.
LOSS_CLAIM_TOLERANCE = 0.5

# How much better than blind guessing a model must be before we stop being
# suspicious of it.
MIN_ACCURACY_MARGIN = 0.05

VERDICT_ACCEPTED = "accepted"
VERDICT_SUSPICIOUS = "suspicious"
VERDICT_REJECTED = "rejected"


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def split_holdout(features, labels, holdout_fraction: float = 0.2,
                  seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a dataset into (train_x, train_y, holdout_x, holdout_y).

    Only the training half is ever sent to a node.
    """
    x = np.asarray(features)
    y = np.asarray(labels)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"Length mismatch: {x.shape[0]} rows vs {y.shape[0]} labels")

    rows = x.shape[0]
    if rows < 4:
        raise ValueError("Dataset is too small to hold anything back.")

    if not 0.0 < holdout_fraction < 1.0:
        raise ValueError(f"holdout_fraction must be between 0 and 1, got {holdout_fraction}")

    holdout_rows = max(1, int(round(rows * holdout_fraction)))
    holdout_rows = min(holdout_rows, rows - 1)   # always leave something to train on

    order = np.random.default_rng(seed).permutation(rows)
    holdout_index, train_index = order[:holdout_rows], order[holdout_rows:]

    return x[train_index], y[train_index], x[holdout_index], y[holdout_index]


def baseline_accuracy(labels) -> float:
    """Accuracy of always guessing the most common class."""
    y = np.asarray(labels)
    if y.size == 0:
        return 0.0
    _values, counts = np.unique(y, return_counts=True)
    return float(counts.max() / y.size)


def untrained_reference(spec: Dict[str, Any], features, labels,
                        samples: int = 3) -> Dict[str, float]:
    """Score freshly initialised models, averaged over a few seeds.

    This is the honest floor for "did this model learn anything". A random
    projection of well-separated data already classifies it surprisingly well,
    so a majority-class baseline is far too easy to clear.
    """
    torch = _torch()
    if torch is None:
        raise RuntimeError("torch is not installed; cannot evaluate.")

    from backend.service.trainer import build_workload, infer_spec_from_dataset

    x = np.asarray(features)
    y = np.asarray(labels)
    resolved = infer_spec_from_dataset(x, y, spec)

    losses, accuracies = [], []
    for seed in range(samples):
        torch.manual_seed(1000 + seed)
        model = build_workload(resolved)["factory"]()
        scored = _score_model(model, resolved, x, y)
        losses.append(scored["loss"])
        accuracies.append(scored["accuracy"])

    return {"loss": float(np.mean(losses)), "accuracy": float(np.mean(accuracies))}


def _score_model(model, resolved: Dict[str, Any], x, y) -> Dict[str, float]:
    """Forward pass only: loss and accuracy for an already-built model."""
    torch = _torch()
    from backend.service.trainer import build_workload

    workload = build_workload(resolved)
    model.eval()

    architecture = str(resolved.get("architecture", "mlp")).lower()
    inputs = torch.as_tensor(np.asarray(x))
    inputs = inputs.float() if architecture in ("mlp", "feedforward") else inputs.long()
    targets = torch.as_tensor(np.asarray(y)).long()

    with torch.no_grad():
        outputs = model(inputs)
        loss = float(workload["loss_fn"](outputs, targets).item())
        predictions = outputs.reshape(-1, outputs.shape[-1]).argmax(dim=-1)
        accuracy = float((predictions == targets.reshape(-1)).float().mean().item())

    return {"loss": loss, "accuracy": accuracy}


def evaluate(state_dict: Dict[str, Any], spec: Dict[str, Any],
             features, labels) -> Dict[str, float]:
    """Score weights on a dataset. One forward pass, no training."""
    torch = _torch()
    if torch is None:
        raise RuntimeError("torch is not installed; cannot evaluate.")

    from backend.service.trainer import build_workload, infer_spec_from_dataset

    x = np.asarray(features)
    y = np.asarray(labels)

    resolved = infer_spec_from_dataset(x, y, spec)
    workload = build_workload(resolved)
    model = workload["factory"]()

    tensors = {name: torch.as_tensor(np.asarray(value))
               for name, value in state_dict.items()}
    # strict=True: a fabricated or mismatched state dict fails here rather than
    # silently evaluating a randomly initialised model.
    model.load_state_dict(tensors, strict=True)
    model.eval()

    architecture = str(resolved.get("architecture", "mlp")).lower()
    if architecture in ("mlp", "feedforward"):
        inputs = torch.as_tensor(x).float()
    else:
        inputs = torch.as_tensor(x).long()
    targets = torch.as_tensor(y).long()

    with torch.no_grad():
        outputs = model(inputs)
        loss = float(workload["loss_fn"](outputs, targets).item())
        predictions = outputs.reshape(-1, outputs.shape[-1]).argmax(dim=-1)
        accuracy = float((predictions == targets.reshape(-1)).float().mean().item())

    return {"loss": loss, "accuracy": accuracy}


def _check(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def verify_training_result(
    state_dict: Dict[str, Any],
    spec: Dict[str, Any],
    holdout_x,
    holdout_y,
    claimed_metrics: Optional[Dict[str, Any]] = None,
    train_sample: Optional[Tuple[Any, Any]] = None,
    initial_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Decide whether a returned model is plausibly the product of training.

    Returns a report with a verdict and the individual checks behind it, so a
    node operator can see exactly why their result was refused.
    """
    checks: List[Dict[str, Any]] = []
    claimed_metrics = claimed_metrics or {}

    # --- structural: the cheapest and most decisive ---
    if not state_dict:
        return {
            "verdict": VERDICT_REJECTED,
            "checks": [_check("weights_present", False, "No weights were returned.")],
            "measured": {},
        }

    arrays = {name: np.asarray(value) for name, value in state_dict.items()}

    finite = all(np.isfinite(a).all() for a in arrays.values() if a.dtype.kind == "f")
    checks.append(_check(
        "weights_finite", finite,
        "All weights are finite." if finite else "Weights contain NaN or infinity.",
    ))

    if initial_state is not None:
        unchanged = all(
            np.array_equal(arrays[name], np.asarray(initial))
            for name, initial in initial_state.items()
            if name in arrays
        )
        checks.append(_check(
            "weights_changed", not unchanged,
            "Weights differ from initialisation."
            if not unchanged else "Weights are identical to their initialisation - nothing was trained.",
        ))

    # A structural failure is unambiguous; stop before spending a forward pass.
    if not all(c["passed"] for c in checks):
        return {"verdict": VERDICT_REJECTED, "checks": checks, "measured": {}}

    # --- performance on data the node never saw ---
    measured: Dict[str, Any] = {}
    try:
        holdout = evaluate(state_dict, spec, holdout_x, holdout_y)
        measured["holdout_loss"] = round(holdout["loss"], 5)
        measured["holdout_accuracy"] = round(holdout["accuracy"], 5)
        checks.append(_check("weights_loadable", True, "Weights matched the model architecture."))
    except Exception as e:
        # Wrong shapes or junk keys land here: the model could not be rebuilt.
        checks.append(_check("weights_loadable", False, f"Weights could not be loaded: {e}"))
        return {"verdict": VERDICT_REJECTED, "checks": checks, "measured": measured}

    baseline = baseline_accuracy(holdout_y)
    measured["baseline_accuracy"] = round(baseline, 5)

    try:
        untrained = untrained_reference(spec, holdout_x, holdout_y)
        measured["untrained_accuracy"] = round(untrained["accuracy"], 5)
        measured["untrained_loss"] = round(untrained["loss"], 5)
    except Exception as e:
        logger.warning(f"Could not score an untrained reference model: {e}")
        untrained = None

    # The bar is whichever floor is higher: guessing, or an untrained network.
    floor = baseline
    floor_name = "guessing the most common class"
    if untrained is not None and untrained["accuracy"] > floor:
        floor = untrained["accuracy"]
        floor_name = "an untrained model of the same architecture"

    beats_baseline = holdout["accuracy"] >= floor + MIN_ACCURACY_MARGIN
    checks.append(_check(
        "beats_baseline", beats_baseline,
        f"Holdout accuracy {holdout['accuracy']:.3f} vs {floor:.3f} for {floor_name}.",
    ))

    # Loss separates trained from untrained far more sharply than accuracy does.
    if untrained is not None:
        better_loss = holdout["loss"] < untrained["loss"]
        checks.append(_check(
            "beats_untrained_loss", better_loss,
            f"Holdout loss {holdout['loss']:.4f} vs {untrained['loss']:.4f} untrained.",
        ))

    # --- claimed loss, recomputed on data the node did see ---
    claimed_loss = claimed_metrics.get("final_loss")
    if train_sample is not None and claimed_loss is not None:
        try:
            seen = evaluate(state_dict, spec, train_sample[0], train_sample[1])
            measured["recomputed_train_loss"] = round(seen["loss"], 5)
            gap = abs(seen["loss"] - float(claimed_loss))
            honest = gap <= LOSS_CLAIM_TOLERANCE
            checks.append(_check(
                "loss_claim_matches", honest,
                f"Claimed final loss {claimed_loss}, recomputed {seen['loss']:.4f} "
                f"on the same data (gap {gap:.4f}).",
            ))
        except Exception as e:
            logger.warning(f"Could not recompute the training loss: {e}")

    # --- verdict ---
    soft_checks = {"beats_baseline", "beats_untrained_loss"}
    if not all(c["passed"] for c in checks if c["name"] not in soft_checks):
        verdict = VERDICT_REJECTED
    elif not all(c["passed"] for c in checks if c["name"] in soft_checks):
        # Could be fraud, could be a genuinely hard problem. Flag, do not condemn.
        verdict = VERDICT_SUSPICIOUS
    else:
        verdict = VERDICT_ACCEPTED

    return {"verdict": verdict, "checks": checks, "measured": measured}


def summarise(report: Dict[str, Any]) -> str:
    """One-line explanation of a verdict, for logs and the dashboard."""
    failed = [c["name"] for c in report.get("checks", []) if not c["passed"]]
    if report.get("verdict") == VERDICT_ACCEPTED:
        measured = report.get("measured", {})
        return (f"accepted (holdout accuracy {measured.get('holdout_accuracy')} "
                f"vs baseline {measured.get('baseline_accuracy')})")
    return f"{report.get('verdict')} - failed: {', '.join(failed) or 'none'}"

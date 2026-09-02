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

# A verdict answers "is this a real trained model, or fabricated". It says
# nothing about whether the model is any good, and people read it as though it
# does: a text model that produced word-shaped nonsense came back marked
# accepted, which was true and deeply misleading.
#
# So alongside the verdict, how far above guessing it got. Not raw accuracy --
# 40% is poor over 3 classes and remarkable over 256 -- but the share of the
# available headroom it captured:
#
#     captured = (accuracy - floor) / (1 - floor)
#
# where floor is whatever a model that learned nothing would score. That is
# comparable across problems of different difficulty, which raw accuracy is
# not.
STRENGTH_WEAK = "weak"
STRENGTH_CLEAR = "clear"
STRENGTH_STRONG = "strong"

# Boundaries drawn from measured runs rather than taste. On this service a
# byte-level text model that captured 0.30 produced unreadable output; one that
# captured 0.51 produced plausible lines; a three-class table classifier
# captured 0.96 and was correct on essentially everything.
WEAK_BELOW = 0.35
STRONG_FROM = 0.75


def learned_fraction(accuracy: float, floor: float) -> float:
    """The share of the possible improvement over guessing that was achieved."""
    headroom = 1.0 - floor
    if headroom <= 0:
        return 0.0
    return max(0.0, min(1.0, (accuracy - floor) / headroom))


def strength_of(captured: float) -> str:
    if captured < WEAK_BELOW:
        return STRENGTH_WEAK
    if captured < STRONG_FROM:
        return STRENGTH_CLEAR
    return STRENGTH_STRONG


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


# The most rows worth holding back, however large the dataset.
#
# A holdout exists to measure one number, and the precision of that number
# depends on how many predictions it covers, not on what share of the data it
# is. A flat 20% of a 200,000 sequence corpus is 40,000 sequences, which is
# five million predictions -- and scoring it means four forward passes on the
# coordinator's CPU, which took longer than the training did on a GPU and
# pinned every core while it ran.
#
# A thousand rows is a standard error well under a percentage point, and it
# costs the same whether the corpus is one megabyte or a hundred. Everything
# above it goes to the node instead, so a bigger dataset now buys more
# training rather than a slower check.
MAX_HOLDOUT_ROWS = 1024


def split_holdout(features, labels, holdout_fraction: float = 0.2,
                  seed: int = 0,
                  max_rows: Optional[int] = MAX_HOLDOUT_ROWS,
                  ordered: bool = False
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a dataset into (train_x, train_y, holdout_x, holdout_y).

    Only the training half is ever sent to a node. `max_rows` bounds the
    holdout so verification costs the same on any size of dataset; pass None
    for a strict fraction.

    `ordered` changes which rows are held back, and it matters more than it
    looks. The default takes a random slice, which is right for rows that have
    no order -- photographs, customers, measurements of separate things.

    It is wrong for anything recorded over time. A random slice of a price
    series, a sensor log or a sales history means training on Tuesday and
    Thursday and being graded on Wednesday, with both neighbours already
    memorised. Series like that are autocorrelated enough for that alone to be
    worth several points of accuracy, and every one of them is fictional.

    Measured on this service, on a model trained from ten years of daily closes
    for ten companies: a random holdout scored 54.2% against an untrained floor
    of 50.7%, which reads as a real edge. The same weights, graded on the last
    two years instead, scored 51.7% against a 51.6% baseline -- no edge at all.
    The submitter was shown the first number.

    So when the data is declared ordered, the holdout is the *end* of it: the
    model is trained on the past and graded on the future, which is the question
    anybody with a time series is actually asking.
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
    if max_rows:
        holdout_rows = min(holdout_rows, int(max_rows))
    holdout_rows = min(holdout_rows, rows - 1)   # always leave something to train on

    if ordered:
        # The last rows, in order. Nothing is shuffled: the point is that the
        # training half happened before the held-back half.
        train_index = np.arange(rows - holdout_rows)
        holdout_index = np.arange(rows - holdout_rows, rows)
    else:
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


# How many rows to push through the model at once when scoring.
#
# This used to be "all of them". On a small table that is the same thing and
# reads better; on a language model trained over a large corpus the holdout is
# hundreds of thousands of sequences, and one forward pass over all of it asks
# for more memory than the machine has. Scoring must not be the thing that
# fails on a job that trained fine.
EVAL_BATCH_ROWS = 512


def _score_batched(model, workload, x, y, architecture: str) -> Dict[str, float]:
    """Loss and accuracy over data too large for a single forward pass.

    The mean over chunks is weighted by how many predictions each contained,
    which makes it exactly the mean over the whole set rather than an average
    of averages -- they differ whenever the last chunk is short.
    """
    torch = _torch()

    x = np.asarray(x)
    y = np.asarray(y)
    is_classifier = architecture in ("mlp", "feedforward")

    total_loss = 0.0
    correct = 0
    predicted = 0

    with torch.no_grad():
        for start in range(0, x.shape[0], EVAL_BATCH_ROWS):
            chunk_x = torch.as_tensor(x[start:start + EVAL_BATCH_ROWS])
            chunk_y = torch.as_tensor(y[start:start + EVAL_BATCH_ROWS]).long()
            # Widened here rather than over the whole array, for the same
            # reason the trainer widens per batch.
            chunk_x = chunk_x.float() if is_classifier else chunk_x.long()

            outputs = model(chunk_x)
            flat = chunk_y.reshape(-1)
            count = flat.numel()
            if not count:
                continue

            total_loss += float(workload["loss_fn"](outputs, chunk_y).item()) * count
            predictions = outputs.reshape(-1, outputs.shape[-1]).argmax(dim=-1)
            correct += int((predictions == flat).sum().item())
            predicted += count

    if not predicted:
        return {"loss": 0.0, "accuracy": 0.0}

    return {"loss": total_loss / predicted, "accuracy": correct / predicted}


def _score_model(model, resolved: Dict[str, Any], x, y) -> Dict[str, float]:
    """Forward pass only: loss and accuracy for an already-built model."""
    from backend.service.trainer import build_workload

    workload = build_workload(resolved)
    model.eval()

    architecture = str(resolved.get("architecture", "mlp")).lower()
    return _score_batched(model, workload, x, y, architecture)


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
    return _score_batched(model, workload, x, y, architecture)


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

    captured = learned_fraction(holdout["accuracy"], floor)
    measured["learned_fraction"] = round(captured, 5)
    measured["floor_accuracy"] = round(floor, 5)

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

    report = {"verdict": verdict, "checks": checks, "measured": measured}

    # Only meaningful for a result we believe is genuine. Grading something we
    # already suspect is fabricated would dress up the wrong question.
    if verdict == VERDICT_ACCEPTED:
        report["strength"] = strength_of(captured)

    return report


def summarise(report: Dict[str, Any]) -> str:
    """One-line explanation of a verdict, for logs and the dashboard."""
    failed = [c["name"] for c in report.get("checks", []) if not c["passed"]]
    if report.get("verdict") == VERDICT_ACCEPTED:
        measured = report.get("measured", {})
        return (f"accepted, {report.get('strength', 'unknown')} "
                f"(holdout accuracy {measured.get('holdout_accuracy')} "
                f"vs {measured.get('floor_accuracy')} for a model that learned "
                f"nothing)")
    return f"{report.get('verdict')} - failed: {', '.join(failed) or 'none'}"

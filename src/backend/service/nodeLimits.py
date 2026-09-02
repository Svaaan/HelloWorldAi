"""What this machine is willing to be asked to do.

Until now a job was checked against a table of constants and never against the
card that would run it. The form accepts hidden_dim up to 16384 and depth up to
64, which is about seventeen billion parameters -- sixty-seven gigabytes of
weights before gradients or optimiser state, on a card that might have eight.
Nothing rejected it. The contributor's machine took the job, downloaded the
data, spun the GPU up and died of an out-of-memory error, having spent their
time and their electricity on a job that could never have finished.

That is worth fixing from the machine's side rather than by lowering the global
caps, because the right answer differs per card: a 4090 can take work a 3070
cannot, and a shared table has to be set for the smallest.

So a node says what it will accept, and the coordinator refuses anything larger
before it is queued. Two ways it is decided:

  * the owner sets it, in the environment, and that is final
  * otherwise it is derived from the card's memory, conservatively

The derived numbers are deliberately not the largest thing that would fit. A job
that exactly fills a card leaves nothing for the desktop the owner is also using,
and "it worked in a benchmark" is not the bar -- the bar is that somebody can
lend a machine they are still sitting at.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Every parameter is stored four times over during training: the weight, its
# gradient, and two Adam moments. At four bytes each that is sixteen bytes per
# parameter before a single activation.
BYTES_PER_PARAMETER = 16

# How much of the card a job may plan to use. The rest is the display, the
# desktop, and the difference between an estimate and the truth.
MEMORY_HEADROOM = 0.55

# Floors, so a machine that reports its memory strangely still accepts
# something reasonable rather than nothing.
MIN_PARAMETERS = 1_000_000
MIN_BATCH = 32

# Ceilings, so a very large card does not advertise limits above what the
# service itself will validate.
MAX_PARAMETERS = 2_000_000_000
MAX_BATCH = 8192
MAX_STEPS = 1_000_000


def _from_env(name: str) -> Optional[int]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        logger.warning("%s is not a number (%r); ignoring it", name, raw)
        return None
    if value <= 0:
        logger.warning("%s must be positive (got %d); ignoring it", name, value)
        return None
    return value


def _total_memory_mb(capabilities: Dict[str, Any]) -> Optional[int]:
    """The smallest card in the machine, because a job runs across all of them.

    Taking the largest would let a job through that the weakest card cannot
    hold, and the pool splits a batch across every device it has.
    """
    gpus = (capabilities or {}).get("gpu")
    if isinstance(gpus, dict):
        gpus = [gpus]
    if not isinstance(gpus, list):
        return None

    sizes = [int(g.get("total_memory") or 0) for g in gpus if isinstance(g, dict)]
    sizes = [s for s in sizes if s > 0]
    return min(sizes) if sizes else None


def derive(capabilities: Dict[str, Any]) -> Dict[str, int]:
    """What this machine will accept, from its hardware and its owner.

    Returned in the shape the coordinator stores and the send-work form reads,
    so a submitter sees the same numbers that will be enforced.
    """
    memory_mb = _total_memory_mb(capabilities)

    if memory_mb:
        usable_bytes = memory_mb * 1024 * 1024 * MEMORY_HEADROOM
        derived_parameters = int(usable_bytes / BYTES_PER_PARAMETER)
    else:
        # No card, or it did not say. A CPU-only node should still be able to
        # take small work rather than advertise nothing.
        derived_parameters = MIN_PARAMETERS * 10

    parameters = _from_env("MAX_MODEL_PARAMETERS") or derived_parameters
    parameters = max(MIN_PARAMETERS, min(parameters, MAX_PARAMETERS))

    # Activations scale with the batch, and far more gently than parameters do.
    # This is a guard against the absurd rather than a tuned figure.
    batch = _from_env("MAX_BATCH_SIZE") or MAX_BATCH
    batch = max(MIN_BATCH, min(batch, MAX_BATCH))

    steps = _from_env("MAX_STEPS") or MAX_STEPS
    steps = max(1, min(steps, MAX_STEPS))

    limits = {
        "max_model_parameters": parameters,
        "max_batch_size": batch,
        "max_steps": steps,
        # Reported so the form can explain where the number came from rather
        # than presenting it as a rule from nowhere.
        "gpu_memory_mb": memory_mb or 0,
        "owner_set": bool(_from_env("MAX_MODEL_PARAMETERS")
                          or _from_env("MAX_BATCH_SIZE")
                          or _from_env("MAX_STEPS")),
    }

    logger.info("This machine will accept up to %s parameters, batch %s, %s steps%s",
                f"{limits['max_model_parameters']:,}",
                limits["max_batch_size"], f"{limits['max_steps']:,}",
                " (set by the owner)" if limits["owner_set"] else "")
    return limits


def parameters_for(spec: Dict[str, Any]) -> int:
    """How many weights a model description implies.

    Counted rather than estimated, because the shape is fully determined by the
    spec and a submitter deserves the real number in the refusal.
    """
    architecture = str(spec.get("architecture", "mlp")).lower()

    if architecture == "mlp":
        input_dim = int(spec.get("input_dim") or 0)
        hidden = int(spec.get("hidden_dim") or 0)
        depth = max(1, int(spec.get("depth") or 1))
        output_dim = int(spec.get("output_dim") or 0)

        # Inputs are read from the dataset, so before that is known the first
        # layer cannot be counted. Assume it matches the hidden width, which
        # over-counts slightly and never under-counts.
        if input_dim <= 0:
            input_dim = hidden

        total = input_dim * hidden + hidden                 # first layer
        total += (depth - 1) * (hidden * hidden + hidden)   # the stack
        total += hidden * max(output_dim, 1) + max(output_dim, 1)
        return int(total)

    if architecture == "transformer":
        width = int(spec.get("d_model") or 0)
        layers = max(1, int(spec.get("n_layer") or 1))
        vocab = int(spec.get("vocab_size") or 0) or width * 4

        # Attention and the feedforward block, per layer, plus embeddings.
        per_layer = 4 * width * width + 8 * width * width
        return int(layers * per_layer + vocab * width)

    return 0


def check(spec: Dict[str, Any], hyperparameters: Dict[str, Any],
          limits: Optional[Dict[str, Any]]) -> Optional[str]:
    """Why this machine will not take this job, or None if it will.

    The message is written for the person who has to change something, so it
    names the number they asked for, the number allowed, and whose machine set
    it.
    """
    if not limits:
        return None                     # a node that has not said; nothing to check

    whose = ("the owner of this machine has set"
             if limits.get("owner_set") else "this machine can take")

    parameters = parameters_for(spec)
    allowed = int(limits.get("max_model_parameters") or 0)
    if allowed and parameters > allowed:
        memory = limits.get("gpu_memory_mb")
        detail = f" ({memory:,} MB of graphics memory)" if memory else ""
        return (
            f"That model has about {parameters:,} parameters and {whose} "
            f"{allowed:,}{detail}. Reduce the width or the number of layers, "
            f"or send it to a larger machine."
        )

    batch = int(hyperparameters.get("batch_size") or 0)
    allowed = int(limits.get("max_batch_size") or 0)
    if allowed and batch > allowed:
        return (f"A batch of {batch:,} is larger than the {allowed:,} {whose}.")

    steps = int(hyperparameters.get("steps") or 0)
    allowed = int(limits.get("max_steps") or 0)
    if allowed and steps > allowed:
        return (f"{steps:,} steps is more than the {allowed:,} {whose}. "
                f"A long job on somebody else's machine is a lot to ask.")

    return None

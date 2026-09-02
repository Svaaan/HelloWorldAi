"""A machine gets to say what it will be asked to do.

The form accepted hidden_dim up to 16384 at depth up to 64. That is about
seventeen billion parameters -- sixty-seven gigabytes of weights before
gradients or optimiser state -- and nothing checked it against the card that
would run it, because the caps were a table of constants and the table has to be
set for the smallest machine on the network.

So a contributor's 8GB card could be handed a job that could never fit. It would
accept it, download the data, spin up, and die of an out-of-memory error, having
spent its owner's time and electricity on arithmetic that was never going to
finish. Nothing in the service considered that a problem.

Now the node advertises limits with its capabilities and the coordinator refuses
anything larger before queueing it. These check both halves: that the numbers a
machine advertises are sane, and that the check actually stops the job.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service import nodeLimits          # noqa: E402


def card(memory_mb):
    return {"gpu": [{"name": "Test GPU", "total_memory": memory_mb}]}


# --- what a machine advertises --------------------------------------------

def test_a_bigger_card_offers_more():
    small = nodeLimits.derive(card(8192))
    large = nodeLimits.derive(card(24564))

    assert large["max_model_parameters"] > small["max_model_parameters"], (
        "limits should follow the hardware; a shared constant has to be set for "
        "the smallest machine and wastes the largest")


def test_the_limit_leaves_the_owner_their_machine():
    """A job that exactly fills the card leaves nothing for the desktop.

    The person lending it is usually still sitting at it.
    """
    limits = nodeLimits.derive(card(8192))

    # Every parameter costs 16 bytes in training: weight, gradient, two Adam
    # moments. The advertised limit must sit well inside the card.
    bytes_needed = limits["max_model_parameters"] * nodeLimits.BYTES_PER_PARAMETER
    card_bytes = 8192 * 1024 * 1024

    assert bytes_needed < card_bytes * 0.7, (
        "the advertised limit would fill the card, leaving nothing for the "
        "display or for the gap between an estimate and the truth")


def test_a_machine_with_no_gpu_still_offers_something():
    """A CPU-only node should take small work, not advertise nothing."""
    limits = nodeLimits.derive({"gpu": [{"name": "No GPU Detected"}]})
    assert limits["max_model_parameters"] >= nodeLimits.MIN_PARAMETERS


def test_the_smallest_card_decides():
    """A batch is split across every device, so the weakest one is the limit."""
    mixed = {"gpu": [{"name": "big", "total_memory": 24564},
                     {"name": "small", "total_memory": 8192}]}

    assert (nodeLimits.derive(mixed)["max_model_parameters"]
            == nodeLimits.derive(card(8192))["max_model_parameters"]), (
        "taking the largest card would admit a job the smallest cannot hold")


def test_an_owner_can_set_it_themselves(monkeypatch):
    monkeypatch.setenv("MAX_MODEL_PARAMETERS", "5000000")
    limits = nodeLimits.derive(card(24564))

    assert limits["max_model_parameters"] == 5_000_000
    assert limits["owner_set"] is True, (
        "the page should be able to say the owner chose this, not the hardware")


def test_nonsense_from_the_environment_is_ignored(monkeypatch):
    """A typo in a compose file should not disable the protection."""
    monkeypatch.setenv("MAX_MODEL_PARAMETERS", "lots")
    limits = nodeLimits.derive(card(8192))

    assert limits["max_model_parameters"] > 0
    assert limits["owner_set"] is False


# --- counting a model ------------------------------------------------------

def test_the_job_that_started_this_is_counted_honestly():
    spec = {"architecture": "mlp", "hidden_dim": 16384, "depth": 64,
            "input_dim": 9, "output_dim": 2}

    parameters = nodeLimits.parameters_for(spec)

    # Sixteen thousand wide, sixty-four deep. Roughly 16384^2 x 63.
    assert parameters > 16_000_000_000, parameters
    assert parameters * nodeLimits.BYTES_PER_PARAMETER > 200 * 1024 ** 3, (
        "this is hundreds of gigabytes of training state")


def test_a_small_model_is_small():
    spec = {"architecture": "mlp", "hidden_dim": 64, "depth": 2,
            "input_dim": 9, "output_dim": 2}
    assert nodeLimits.parameters_for(spec) < 10_000


# --- the refusal -----------------------------------------------------------

def test_a_model_too_large_for_the_card_is_refused():
    limits = nodeLimits.derive(card(8192))
    spec = {"architecture": "mlp", "hidden_dim": 16384, "depth": 64,
            "input_dim": 9, "output_dim": 2}

    refusal = nodeLimits.check(spec, {"batch_size": 32, "steps": 200}, limits)

    assert refusal is not None
    # The message has to be actionable: both numbers, and what to do.
    assert "16,912,662,530" in refusal or "parameters" in refusal
    assert "Reduce the width" in refusal


def test_a_sensible_job_is_accepted():
    limits = nodeLimits.derive(card(8192))
    spec = {"architecture": "mlp", "hidden_dim": 64, "depth": 2,
            "input_dim": 9, "output_dim": 2}

    assert nodeLimits.check(spec, {"batch_size": 32, "steps": 4000}, limits) is None


def test_an_oversized_batch_is_refused(monkeypatch):
    monkeypatch.setenv("MAX_BATCH_SIZE", "128")
    limits = nodeLimits.derive(card(8192))
    spec = {"architecture": "mlp", "hidden_dim": 64, "depth": 2}

    refusal = nodeLimits.check(spec, {"batch_size": 4096, "steps": 100}, limits)
    assert refusal and "batch" in refusal.lower()


def test_too_many_steps_is_refused(monkeypatch):
    """A long job on somebody else's machine is a lot to ask."""
    monkeypatch.setenv("MAX_STEPS", "1000")
    limits = nodeLimits.derive(card(8192))
    spec = {"architecture": "mlp", "hidden_dim": 64, "depth": 2}

    refusal = nodeLimits.check(spec, {"batch_size": 32, "steps": 500000}, limits)
    assert refusal and "steps" in refusal


def test_the_refusal_says_whose_rule_it_is(monkeypatch):
    """An owner's choice and a hardware ceiling read differently to a submitter."""
    spec = {"architecture": "mlp", "hidden_dim": 4096, "depth": 8}

    hardware = nodeLimits.check(spec, {}, nodeLimits.derive(card(2048)))
    assert hardware and "this machine can take" in hardware

    monkeypatch.setenv("MAX_MODEL_PARAMETERS", "1000000")
    owner = nodeLimits.check(spec, {}, nodeLimits.derive(card(24564)))
    assert owner and "the owner of this machine has set" in owner


def test_a_node_that_has_not_said_is_not_second_guessed():
    """An older node advertises no limits; it is left alone rather than assumed."""
    spec = {"architecture": "mlp", "hidden_dim": 16384, "depth": 64}
    assert nodeLimits.check(spec, {"batch_size": 8192, "steps": 999999}, None) is None
    assert nodeLimits.check(spec, {}, {}) is None


# --- the wiring ------------------------------------------------------------

def test_the_node_advertises_limits_with_its_capabilities():
    import inspect
    from backend.service import systemInfoService

    source = inspect.getsource(systemInfoService)
    assert 'capabilities["limits"]' in source, (
        "the node does not send its limits, so the coordinator has nothing to "
        "check a job against")


def test_the_coordinator_checks_before_queueing():
    import inspect
    from backend.routes import tasks

    source = inspect.getsource(tasks._queue_task)
    assert "nodeLimits.check" in source, (
        "a job is queued without being compared to the machine that will run it")

    # And the auto path filters candidates rather than assigning then refusing.
    anywhere = inspect.getsource(tasks.submit_task_anywhere)
    assert "nodeLimits.check" in anywhere, (
        "auto-placement can still hand a large job to a small card")

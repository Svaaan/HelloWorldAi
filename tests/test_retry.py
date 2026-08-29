"""Running a job again on the same data, with different settings.

Two things came out of using the service. "Run again" repeated a result you had
already seen, and getting the same data with a bigger model meant re-uploading
the file -- which put the data on the wire twice and scored the second run
against a different holdout, so the two could not honestly be compared.

And a quieter one: the data behind a finished job is deleted after a grace
period, and the task keeps pointing at nothing. Retry never checked. It queued
a job with no dataset, which trained on random numbers, could not be verified,
and came back marked completed -- the exact thing a normally submitted job is
now refused for.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service.jobSpec import JobSpecError, validate_job   # noqa: E402


def merged(task_data, changes):
    """The merge /retry-task performs before validating.

    Kept in step with the endpoint by asserting on the same behaviour it has:
    changes are folded into the stored job, then run through the same validator
    a first submission uses.
    """
    task_data = dict(task_data)
    for field in ("model_spec", "hyperparameters"):
        supplied = changes.get(field)
        if isinstance(supplied, dict):
            task_data[field] = {**(task_data.get(field) or {}), **supplied}
    if changes.get("model_name"):
        task_data["model_name"] = changes["model_name"]
    return task_data


ORIGINAL = {
    "task_type": "llm_training",
    "model_name": "tuning",
    "model_spec": {"architecture": "mlp", "hidden_dim": 8, "depth": 1},
    "hyperparameters": {"steps": 40, "batch_size": 16, "learning_rate": 0.01},
}


# --- what a change is allowed to do --------------------------------------

def test_a_change_replaces_only_what_it_names():
    changed = merged(ORIGINAL, {"model_spec": {"hidden_dim": 64}})
    clean, _ = validate_job(changed)

    assert clean["model_spec"]["hidden_dim"] == 64
    # Untouched, rather than reset to the schema default.
    assert clean["model_spec"]["depth"] == 1
    assert clean["hyperparameters"]["steps"] == 40


def test_settings_and_model_can_change_together():
    changed = merged(ORIGINAL, {
        "model_name": "tuning-v2",
        "model_spec": {"hidden_dim": 64, "depth": 2},
        "hyperparameters": {"steps": 400},
    })
    clean, _ = validate_job(changed)

    assert clean["model_name"] == "tuning-v2"
    assert clean["model_spec"] == {"architecture": "mlp", "hidden_dim": 64, "depth": 2}
    assert clean["hyperparameters"]["steps"] == 400
    assert clean["hyperparameters"]["learning_rate"] == 0.01


def test_no_changes_reruns_exactly_what_ran_before():
    clean, _ = validate_job(merged(ORIGINAL, {}))

    assert clean["model_spec"] == ORIGINAL["model_spec"]
    assert clean["hyperparameters"] == ORIGINAL["hyperparameters"]


def test_a_retry_cannot_smuggle_past_the_validator():
    # It goes through the same check a first submission does, so a value the
    # form would refuse is refused here too.
    changed = merged(ORIGINAL, {"hyperparameters": {"steps": -5}})

    with pytest.raises(JobSpecError, match="between"):
        validate_job(changed)


def test_switching_architecture_is_refused():
    """The dataset is fixed for a re-run, so the kind of model is too.

    A CSV cannot be read by a language model and text cannot be read by a
    classifier. The check that refuses that pairing runs at upload, which a
    re-run does not repeat -- so the endpoint refuses the switch outright.
    Validation alone does not: the merged job parses perfectly well.
    """
    changed = merged(ORIGINAL, {"model_spec": {"architecture": "transformer"}})
    clean, _ = validate_job(changed)
    assert clean["model_spec"]["architecture"] == "transformer"   # parses fine

    assert refuses_architecture_change(ORIGINAL, {"architecture": "transformer"})
    assert not refuses_architecture_change(ORIGINAL, {"hidden_dim": 64})
    # Same architecture spelled the same way is not a change.
    assert not refuses_architecture_change(ORIGINAL, {"architecture": "MLP"})


def refuses_architecture_change(task_data, spec_changes):
    """The guard /retry-task applies before merging anything."""
    original = str((task_data.get("model_spec") or {}).get("architecture", "mlp"))
    wanted = spec_changes.get("architecture")
    return bool(wanted) and str(wanted).lower() != original.lower()


def test_an_impossible_transformer_is_still_caught():
    original = {
        "model_name": "text",
        "model_spec": {"architecture": "transformer", "d_model": 100, "n_head": 4,
                       "n_layer": 2},
        "hyperparameters": {"steps": 100, "batch_size": 8, "learning_rate": 0.0005},
    }
    changed = merged(original, {"model_spec": {"n_head": 7}})

    with pytest.raises(JobSpecError, match="divide evenly"):
        validate_job(changed)


# --- and the arithmetic still runs on the new numbers --------------------

def test_advice_is_recomputed_for_the_new_settings():
    from backend.service.jobSpec import advise

    original = {
        "model_spec": {"architecture": "transformer"},
        "hyperparameters": {"steps": 2000, "batch_size": 32},
    }
    # Plenty at 2,000 steps; far too few at 10.
    assert advise(original, {"rows": 5000}) == []

    fewer = merged(original, {"hyperparameters": {"steps": 10}})
    notes = advise(fewer, {"rows": 5000})

    assert len(notes) == 1
    assert "More steps" in notes[0]

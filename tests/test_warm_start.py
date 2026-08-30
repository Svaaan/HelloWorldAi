"""Carrying on from a trained model instead of starting over.

The runs on one dataset are a sequence -- each one an adjustment of the last --
and every one of them began from random noise. Improving a model meant paying
for everything it had already learned a second time.

The reason this needs care rather than a flag: a job that continues another is
*handed* a model that already works. Nothing stops a node from claiming the
job, sleeping, and returning the same weights, and they would score well
because they were good when they arrived. The check for that already existed
and had nothing to compare against.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

SPEC = {"architecture": "mlp", "hidden_dim": 16, "depth": 1}


def clusters(seed=4, count=300):
    """Three well-separated blobs: an easy problem, honestly solvable."""
    rng = np.random.default_rng(seed)
    centres = np.array([[0.0, 0.0], [6.0, 6.0], [0.0, 6.0]], dtype=np.float32)
    labels = np.repeat(np.arange(3), count // 3).astype(np.int64)
    features = (centres[labels] + rng.normal(scale=0.4, size=(len(labels), 2)))
    return features.astype(np.float32), labels


def trained(dataset, steps=200, initial_state=None):
    from backend.service.trainer import train
    return train(
        {"model_name": "m", "model_spec": SPEC,
         "hyperparameters": {"steps": steps, "batch_size": 32,
                             "learning_rate": 0.01}},
        log=lambda _m: None, dataset=dataset, initial_state=initial_state,
    )


# --- picking up where the last run stopped -------------------------------

def test_a_continued_run_does_not_start_from_noise():
    pytest.importorskip("torch")
    x, y = clusters()

    first = trained((x, y), steps=200)
    second = trained((x, y), steps=20, initial_state=first["state_dict"])

    # A fresh model starts near ln(3) = 1.10 on three classes. One that
    # carries on starts wherever the last one finished.
    assert second["metrics"]["initial_loss"] < first["metrics"]["initial_loss"]
    assert second["metrics"]["initial_loss"] <= first["metrics"]["final_loss"] * 2


def test_it_says_so_in_the_metrics():
    pytest.importorskip("torch")
    x, y = clusters()

    cold = trained((x, y), steps=10)
    warm = trained((x, y), steps=10, initial_state=cold["state_dict"])

    assert cold["metrics"]["warm_started"] is False
    assert warm["metrics"]["warm_started"] is True


def test_continuing_reaches_further_than_starting_again():
    """The whole point: the second run keeps the first one's progress."""
    pytest.importorskip("torch")
    x, y = clusters()

    first = trained((x, y), steps=200)
    carried = trained((x, y), steps=200, initial_state=first["state_dict"])
    from_scratch = trained((x, y), steps=200)

    assert carried["metrics"]["final_loss"] < from_scratch["metrics"]["final_loss"]


def test_weights_that_do_not_fit_are_refused_by_name():
    pytest.importorskip("torch")
    x, y = clusters()

    wide = trained((x, y), steps=5)
    narrow = {"architecture": "mlp", "hidden_dim": 4, "depth": 1}

    from backend.service.trainer import train
    with pytest.raises(ValueError, match="do not fit"):
        train({"model_name": "m", "model_spec": narrow,
               "hyperparameters": {"steps": 5, "batch_size": 8, "learning_rate": 0.01}},
              log=lambda _m: None, dataset=(x, y),
              initial_state=wide["state_dict"])


def test_a_partial_state_is_refused_rather_than_half_loaded():
    pytest.importorskip("torch")
    x, y = clusters()

    full = trained((x, y), steps=5)["state_dict"]
    partial = {name: value for i, (name, value) in enumerate(full.items()) if i}

    from backend.service.trainer import train
    with pytest.raises(ValueError, match="do not fit"):
        train({"model_name": "m", "model_spec": SPEC,
               "hyperparameters": {"steps": 5, "batch_size": 8, "learning_rate": 0.01}},
              log=lambda _m: None, dataset=(x, y), initial_state=partial)


# --- and the fraud it would otherwise invite -----------------------------

def test_handing_the_starting_weights_straight_back_is_rejected():
    pytest.importorskip("torch")
    from backend.service.verification import split_holdout, verify_training_result

    x, y = clusters()
    train_x, train_y, holdout_x, holdout_y = split_holdout(x, y)
    good = trained((train_x, train_y), steps=200)["state_dict"]

    report = verify_training_result(good, SPEC, holdout_x, holdout_y,
                                    {}, None, good)

    assert report["verdict"] == "rejected"
    assert any(c["name"] == "weights_changed" and not c["passed"]
               for c in report["checks"])


def test_the_same_fraud_passes_when_the_starting_point_is_withheld():
    """Why the coordinator has to pass it, not merely be able to.

    Without the starting point there is nothing to compare against, and
    returning a good model somebody else trained scores exactly as well as
    training one.
    """
    pytest.importorskip("torch")
    from backend.service.verification import split_holdout, verify_training_result

    x, y = clusters()
    train_x, train_y, holdout_x, holdout_y = split_holdout(x, y)
    good = trained((train_x, train_y), steps=200)["state_dict"]

    blind = verify_training_result(good, SPEC, holdout_x, holdout_y, {})

    assert blind["verdict"] == "accepted"


def test_an_honest_continuation_still_passes():
    pytest.importorskip("torch")
    from backend.service.verification import split_holdout, verify_training_result

    x, y = clusters()
    train_x, train_y, holdout_x, holdout_y = split_holdout(x, y)

    first = trained((train_x, train_y), steps=200)["state_dict"]
    second = trained((train_x, train_y), steps=100, initial_state=first)

    report = verify_training_result(second["state_dict"], SPEC,
                                    holdout_x, holdout_y, {}, None, first)

    assert report["verdict"] == "accepted"
    assert all(c["passed"] for c in report["checks"])


# --- what a continuing job may change ------------------------------------

def refuses_shape_change(task_data, spec_changes):
    """The guard /retry-task applies when continue_from is set."""
    current = task_data.get("model_spec") or {}
    differing = [k for k, v in spec_changes.items()
                 if k != "architecture" and current.get(k) != v]
    return sorted(differing)


JOB = {"model_spec": {"architecture": "mlp", "hidden_dim": 16, "depth": 1},
       "hyperparameters": {"steps": 200, "batch_size": 32, "learning_rate": 0.01}}


def test_the_shape_cannot_change_while_continuing():
    # Refused where it is a sentence, rather than on a contributor's machine
    # after the job has been claimed.
    assert refuses_shape_change(JOB, {"hidden_dim": 64}) == ["hidden_dim"]
    assert refuses_shape_change(JOB, {"hidden_dim": 64, "depth": 3}) == ["depth", "hidden_dim"]


def test_the_same_shape_restated_is_not_a_change():
    assert refuses_shape_change(JOB, {"hidden_dim": 16, "depth": 1}) == []


def test_training_settings_are_free_to_change():
    # They are the reason to continue at all: train longer, or slower.
    assert refuses_shape_change(JOB, {}) == []

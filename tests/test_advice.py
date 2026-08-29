"""Telling people what a run will be worth, before it costs anyone anything.

Both of these came out of running the service as a first-time user. 71 KB of
text trained cleanly, passed every verification check, was labelled Verified,
and produced word-shaped nonsense. Nothing was broken. The system simply
answered a question ("is this model genuine") that reads like a different
question ("is this model good"), and never mentioned that the corpus was far
too small until after a contributor's GPU had spent the time.

So: say it at upload, and grade the result separately from trusting it.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service.artifacts import (          # noqa: E402
    TEXT_COMFORTABLE_BYTES, TEXT_THIN_BYTES, text_size_advice,
)
from backend.service.jobSpec import (            # noqa: E402
    MAX_COMFORTABLE_PASSES, advise,
)
from backend.service.verification import (       # noqa: E402
    STRENGTH_CLEAR, STRENGTH_STRONG, STRENGTH_WEAK,
    learned_fraction, strength_of, verify_training_result,
)


# --- how much text is enough ---------------------------------------------

def test_a_tiny_corpus_is_called_out():
    advice = text_size_advice(70 * 1024)

    assert advice
    # The specific failure a first-time user hits, named rather than hinted at.
    assert "words without meaning" in advice


def test_a_thin_corpus_gets_a_softer_note():
    small = text_size_advice(70 * 1024)
    thin = text_size_advice(TEXT_THIN_BYTES + 1024)

    assert thin
    assert thin != small
    assert "thin" in thin


def test_enough_text_says_nothing_at_all():
    # Silence is the right output for a file that is big enough; a warning on
    # every upload is a warning on none of them.
    assert text_size_advice(TEXT_COMFORTABLE_BYTES) is None
    assert text_size_advice(50 * 1024 * 1024) is None


def test_the_advice_is_readable_rather_than_technically_correct():
    advice = text_size_advice(40 * 1024)

    assert "1 MB" in advice          # not "1024 KB"
    assert "40 KB" in advice


# --- how many times a run reads the data ---------------------------------

TEXT_JOB = {"model_spec": {"architecture": "transformer"},
            "hyperparameters": {"steps": 1000, "batch_size": 32}}


def test_too_many_passes_over_too_little_data_is_flagged():
    notes = advise(TEXT_JOB, {"rows": 897})

    assert len(notes) == 1
    assert "36 times" in notes[0]
    assert str(MAX_COMFORTABLE_PASSES) in notes[0]


def test_a_run_that_barely_reads_the_data_is_flagged_too():
    notes = advise({"model_spec": {"architecture": "transformer"},
                    "hyperparameters": {"steps": 20, "batch_size": 8}},
                   {"rows": 5000})

    assert len(notes) == 1
    assert "3%" in notes[0]


def test_a_well_proportioned_run_is_left_alone():
    # 1000 steps x batch 32 = 32,000 samples; over 4,000 rows that is 8 passes.
    assert advise(TEXT_JOB, {"rows": 4000}) == []


def test_a_classifier_is_not_nagged_about_epochs():
    # Measured, not assumed: 240 rows over 53 passes scored 100% on the
    # holdout. Many epochs over a small table is normal practice.
    notes = advise({"model_spec": {"architecture": "mlp"},
                    "hyperparameters": {"steps": 400, "batch_size": 32}},
                   {"rows": 240})

    assert notes == []


def test_advice_needs_a_dataset_to_say_anything():
    assert advise(TEXT_JOB, None) == []
    assert advise(TEXT_JOB, {}) == []


def test_advice_never_raises_on_a_malformed_job():
    # It runs after validation, but it must not be the thing that breaks a
    # submission -- none of this makes a job wrong.
    assert advise({}, {"rows": 100}) == []
    assert advise({"hyperparameters": {"steps": 0, "batch_size": 0}},
                  {"rows": 100}) == []


# --- how well it learned, as distinct from whether it is genuine ---------

def test_the_fraction_is_the_share_of_headroom_captured():
    # Halfway from the floor to perfect.
    assert learned_fraction(0.5, 0.0) == pytest.approx(0.5)
    assert learned_fraction(0.6, 0.2) == pytest.approx(0.5)
    # At the floor, nothing was captured.
    assert learned_fraction(0.2, 0.2) == pytest.approx(0.0)
    # Perfect captures everything.
    assert learned_fraction(1.0, 0.3) == pytest.approx(1.0)


def test_below_the_floor_is_zero_rather_than_negative():
    assert learned_fraction(0.1, 0.5) == 0.0


def test_an_impossible_floor_does_not_divide_by_zero():
    assert learned_fraction(1.0, 1.0) == 0.0


def test_the_measure_compares_across_problems_of_different_difficulty():
    # 40% over 256 byte values and 40% over three classes are not the same
    # achievement, and raw accuracy cannot tell them apart.
    text_like = learned_fraction(0.405, 0.159)
    table_like = learned_fraction(0.405, 0.400)

    assert text_like > table_like
    assert strength_of(text_like) == STRENGTH_WEAK
    assert strength_of(table_like) == STRENGTH_WEAK


def test_the_grades_run_in_the_right_order():
    assert strength_of(0.0) == STRENGTH_WEAK
    assert strength_of(0.293) == STRENGTH_WEAK      # the 71 KB text model
    assert strength_of(0.509) == STRENGTH_CLEAR     # the 310 KB text model
    assert strength_of(1.0) == STRENGTH_STRONG      # the table classifier


def test_a_genuine_but_poor_model_is_accepted_and_graded_weak():
    pytest.importorskip("torch")
    from backend.service.trainer import train

    # Barely trained on purpose: real weights, real training, poor result.
    rng = np.random.default_rng(0)
    x = rng.normal(size=(400, 6)).astype(np.float32)
    y = rng.integers(0, 4, size=400).astype(np.int64)   # labels carry no signal

    spec = {"architecture": "mlp", "hidden_dim": 8, "depth": 1}
    result = train(
        {"model_name": "noise", "model_spec": spec,
         "hyperparameters": {"steps": 2, "batch_size": 8, "learning_rate": 1e-6}},
        log=lambda _m: None, dataset=(x[:300], y[:300]),
    )

    report = verify_training_result(result["state_dict"], spec, x[300:], y[300:])

    # It may or may not clear the soft checks -- unlearnable data is exactly
    # the ambiguous case. What matters is that when it is accepted, it is not
    # accepted silently.
    if report["verdict"] == "accepted":
        assert report["strength"] == STRENGTH_WEAK
    assert "learned_fraction" in report["measured"]


def test_a_good_model_is_graded_strong():
    pytest.importorskip("torch")
    from backend.service.trainer import train
    from backend.service.verification import split_holdout

    # Three clearly separated clusters: an easy problem, honestly solved.
    rng = np.random.default_rng(3)
    centres = np.array([[0.0, 0.0], [8.0, 8.0], [0.0, 8.0]], dtype=np.float32)
    labels = np.repeat(np.arange(3), 120).astype(np.int64)
    features = (centres[labels] + rng.normal(scale=0.4, size=(360, 2))).astype(np.float32)

    train_x, train_y, holdout_x, holdout_y = split_holdout(features, labels)

    spec = {"architecture": "mlp", "hidden_dim": 32, "depth": 2}
    result = train(
        {"model_name": "clusters", "model_spec": spec,
         "hyperparameters": {"steps": 300, "batch_size": 32, "learning_rate": 0.01}},
        log=lambda _m: None, dataset=(train_x, train_y),
    )

    report = verify_training_result(result["state_dict"], spec, holdout_x, holdout_y)

    assert report["verdict"] == "accepted"
    assert report["strength"] == STRENGTH_STRONG


def test_a_refused_result_is_not_graded():
    # Grading something we already believe is fabricated would dress up the
    # wrong question.
    report = verify_training_result({}, {"architecture": "mlp"},
                                    np.zeros((4, 2)), np.zeros(4))

    assert report["verdict"] == "rejected"
    assert "strength" not in report


def test_the_summary_says_the_grade_out_loud():
    from backend.service.verification import summarise

    line = summarise({
        "verdict": "accepted",
        "strength": STRENGTH_WEAK,
        "checks": [],
        "measured": {"holdout_accuracy": 0.405, "floor_accuracy": 0.159},
    })

    assert "weak" in line
    assert "0.405" in line

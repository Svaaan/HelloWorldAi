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
    MAX_SUGGESTED_STEPS, MIN_COVERAGE, TARGET_PASSES, advise, suggest_steps,
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


def test_a_run_that_barely_reads_the_data_is_flagged():
    notes = advise({"model_spec": {"architecture": "transformer"},
                    "hyperparameters": {"steps": 20, "batch_size": 8}},
                   {"rows": 5000})

    assert len(notes) == 1
    assert "3%" in notes[0]


def test_a_run_that_reads_all_of_it_is_left_alone():
    # 1000 steps x batch 32 = 32,000 draws over 897 rows: every row many times.
    assert advise(TEXT_JOB, {"rows": 897}) == []
    assert advise(TEXT_JOB, {"rows": 4000}) == []


def test_training_for_a_long_time_is_not_called_a_mistake():
    """The claim that was measured and found false.

    Same 897 sequences, same model, only the step count changed:

        8 passes   holdout accuracy 0.349, captured 0.225
       36 passes   holdout accuracy 0.407, captured 0.295

    Training longer over the same small corpus made the held-back score
    better. There used to be a warning here saying it would make it worse.
    """
    assert advise(TEXT_JOB, {"rows": 897}) == []
    assert advise({"model_spec": {"architecture": "transformer"},
                   "hyperparameters": {"steps": 100_000, "batch_size": 32}},
                  {"rows": 897}) == []


def test_coverage_accounts_for_sampling_with_replacement():
    # 4,000 draws over 5,000 rows is not 80% of the rows: repeats mean the
    # expected share reached is 1 - e^(-0.8), about 55%.
    notes = advise({"model_spec": {"architecture": "transformer"},
                    "hyperparameters": {"steps": 500, "batch_size": 8}},
                   {"rows": 5000})

    assert len(notes) == 1
    assert "55%" in notes[0]


# --- sizing a run to the data it was given -------------------------------

def test_a_big_corpus_raises_the_step_count():
    # 3 passes over 15,750 rows at batch 32.
    assert suggest_steps(15_750, 32, 1000) == 1477


def test_a_small_corpus_leaves_the_default_alone():
    # Lowering it was the obvious other half, and the measurement says no:
    # training less over too little data gives a worse model, not a safer one.
    assert suggest_steps(897, 32, 1000) == 1000


def test_a_suggestion_never_proposes_an_endless_job():
    assert suggest_steps(50_000_000, 32, 1000) == MAX_SUGGESTED_STEPS


def test_sizing_survives_nonsense_input():
    assert suggest_steps(0, 32, 1000) == 1000
    assert suggest_steps(500, 0, 1000) == 1000


def test_the_form_is_told_the_same_numbers():
    from backend.service.jobSpec import job_schema

    guidance = job_schema()["guidance"]

    assert guidance["target_passes"] == TARGET_PASSES
    assert guidance["min_coverage"] == MIN_COVERAGE
    assert guidance["max_suggested_steps"] == MAX_SUGGESTED_STEPS


def test_advice_needs_a_dataset_to_say_anything():
    assert advise(TEXT_JOB, None) == []
    assert advise(TEXT_JOB, {}) == []


def test_advice_never_raises_on_a_malformed_job():
    # It runs after validation, but it must not be the thing that breaks a
    # submission -- none of this makes a job wrong.
    assert advise({}, {"rows": 100}) == []
    assert advise({"hyperparameters": None}, {"rows": 100}) == []
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


# --- what a large corpus costs, and who pays it --------------------------

def test_byte_ids_are_stored_as_bytes():
    """The cap on a text upload was never about disk.

    Packed, a corpus compresses to a fraction of its source. The cost is memory
    on the contributor's machine, and it used to be 16x the source file there:
    int32 arrays for x and y, widened to int64 in one go before the first
    batch. A byte id fits in a uint8, and the widening belongs on the batch.
    """
    from backend.service.artifacts import parse_text_dataset

    text = "the quick brown fox jumps over the lazy dog. " * 400
    x, y, _ = parse_text_dataset(text, seq_len=64)

    assert x.dtype == np.uint8
    assert y.dtype == np.uint8

    source = len(text.encode("utf-8"))
    assert (x.nbytes + y.nbytes) / source <= 2.1


def test_batches_are_widened_rather_than_the_corpus():
    torch = pytest.importorskip("torch")
    from backend.service.artifacts import parse_text_dataset
    from backend.service.trainer import dataset_sampler, infer_spec_from_dataset

    x, y, _ = parse_text_dataset("the quick brown fox. " * 800, seq_len=32)
    spec = infer_spec_from_dataset(x, y, {"architecture": "transformer"})

    make_batch, _rows = dataset_sampler(x, y, spec)
    batch_x, batch_y = make_batch(8, torch.Generator().manual_seed(0),
                                  torch.device("cpu"))

    # The batch is what an embedding lookup needs...
    assert batch_x.dtype == torch.int64
    assert batch_y.dtype == torch.int64
    # ...and the corpus behind it is untouched.
    assert x.dtype == np.uint8


def test_a_classifier_still_gets_float_features():
    torch = pytest.importorskip("torch")
    from backend.service.trainer import dataset_sampler

    x = np.zeros((64, 4), dtype=np.float32)
    y = np.zeros(64, dtype=np.int64)

    make_batch, _rows = dataset_sampler(x, y, {"architecture": "mlp"})
    batch_x, batch_y = make_batch(8, torch.Generator().manual_seed(0),
                                  torch.device("cpu"))

    assert batch_x.dtype == torch.float32
    assert batch_y.dtype == torch.int64


def test_the_holdout_stops_growing_with_the_dataset():
    """Verification has to cost the same on any size of corpus.

    A flat 20% of a 200,000 sequence dataset is five million predictions, and
    scoring it took longer on the coordinator's CPU than the training did on a
    GPU -- while pinning every core. Everything above the cap goes to the node
    instead, so a bigger dataset buys more training, not a slower check.
    """
    from backend.service.verification import MAX_HOLDOUT_ROWS, split_holdout

    small_x = np.zeros((300, 4), dtype=np.float32)
    small_y = np.zeros(300, dtype=np.int64)
    _tx, _ty, hx, _hy = split_holdout(small_x, small_y)
    assert hx.shape[0] == 60          # still a plain 20% when that is small

    big_x = np.zeros((204_800, 8), dtype=np.uint8)
    big_y = np.zeros((204_800, 8), dtype=np.uint8)
    tx, _ty, hx, _hy = split_holdout(big_x, big_y)

    assert hx.shape[0] == MAX_HOLDOUT_ROWS
    assert tx.shape[0] == 204_800 - MAX_HOLDOUT_ROWS


def test_a_strict_fraction_is_still_available():
    from backend.service.verification import split_holdout

    x = np.zeros((10_000, 4), dtype=np.float32)
    y = np.zeros(10_000, dtype=np.int64)

    _tx, _ty, hx, _hy = split_holdout(x, y, max_rows=None)

    assert hx.shape[0] == 2000


def test_scoring_in_chunks_matches_scoring_all_at_once():
    """The batched path has to give the same numbers, not similar ones.

    Weighted by predictions per chunk rather than averaged over chunks -- the
    two differ whenever the last chunk is short, which is almost always.
    """
    torch = pytest.importorskip("torch")
    from backend.service.trainer import build_workload
    from backend.service.verification import EVAL_BATCH_ROWS, _score_batched

    spec = {"architecture": "mlp", "input_dim": 4, "output_dim": 3,
            "hidden_dim": 8, "depth": 1}
    torch.manual_seed(0)
    model = build_workload(spec)["factory"]()
    model.eval()

    rng = np.random.default_rng(1)
    # Deliberately not a multiple of the chunk size.
    rows = EVAL_BATCH_ROWS * 2 + 37
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    y = rng.integers(0, 3, size=rows).astype(np.int64)

    chunked = _score_batched(model, build_workload(spec), x, y, "mlp")

    with torch.no_grad():
        outputs = model(torch.as_tensor(x))
        whole_loss = float(torch.nn.functional.cross_entropy(
            outputs, torch.as_tensor(y)).item())
        whole_accuracy = float(
            (outputs.argmax(dim=-1) == torch.as_tensor(y)).float().mean().item())

    assert chunked["loss"] == pytest.approx(whole_loss, abs=1e-5)
    # Not exact to the last bit, and the chunked value is the better one: it
    # counts matches as integers, where the single-pass version takes a float32
    # mean over every prediction and loses precision doing it.
    assert chunked["accuracy"] == pytest.approx(whole_accuracy, abs=1e-6)

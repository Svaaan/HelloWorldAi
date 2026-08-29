"""Training a language model on text somebody actually has.

The transformer was offered in the job form from the start, but it could only
ever run on tokens it generated for itself: pointing it at a dataset raised
`Expected input batch_size (32) to match target batch_size (8)` -- because the
sampler handed it one label per row when a language model needs one per
position. These tests cover the path that fixes, from a .txt file to a model
whose shape matches the data it was given.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service.artifacts import (          # noqa: E402
    ArtifactError, MIN_TEXT_WINDOWS, TEXT_VOCAB_SIZE,
    pack_dataset, parse_text_dataset, unpack_dataset,
)
from backend.service.jobSpec import JobSpecError, validate_job   # noqa: E402
from backend.service.modelManifest import build_manifest         # noqa: E402
from backend.service.trainer import (            # noqa: E402
    dataset_sampler, infer_spec_from_dataset,
)

PROSE = "the quick brown fox jumps over the lazy dog. " * 80


# --- turning text into training data -------------------------------------

def test_text_becomes_windows_of_token_ids():
    x, y, info = parse_text_dataset(PROSE, seq_len=32)

    assert x.shape == y.shape
    assert x.shape[1] == 32
    assert x.dtype.kind in "iu" and y.dtype.kind in "iu"
    assert info["tokenizer"] == "bytes"
    assert info["vocab_size"] == TEXT_VOCAB_SIZE
    assert info["seq_len"] == 32
    assert info["rows"] == x.shape[0]


def test_labels_are_the_input_shifted_by_one():
    # This is the whole supervision signal for a language model: at every
    # position, the target is the token that actually followed.
    x, y, _ = parse_text_dataset(PROSE, seq_len=16)

    assert np.array_equal(y[:, :-1], x[:, 1:])


def test_windows_are_contiguous_and_do_not_overlap():
    x, y, _ = parse_text_dataset(PROSE, seq_len=16)

    # The last target of one window is the first input of the next, so the
    # stream is covered exactly once. Overlapping windows would put the same
    # text in both the training half and the holdout.
    assert y[0, -1] == x[1, 0]
    assert x.shape[0] * 16 <= len(PROSE.encode("utf-8"))


def test_ids_are_the_bytes_themselves():
    x, _, _ = parse_text_dataset(PROSE, seq_len=16)

    assert x.min() >= 0
    assert x.max() < TEXT_VOCAB_SIZE
    assert bytes(x[0].tolist()) == PROSE.encode("utf-8")[:16]


def test_non_english_text_survives_the_round_trip():
    text = "raksmorgas och kottbullar, kalimera. " * 40
    x, _y, info = parse_text_dataset(text, seq_len=32)

    rebuilt = bytes(x.reshape(-1).tolist())
    assert rebuilt == text.encode("utf-8")[:info["tokens"]]


def test_multi_byte_characters_are_kept_as_their_bytes():
    # A byte tokeniser has no idea what a character is, and does not need to:
    # the model learns the multi-byte sequences the same way it learns
    # spelling.
    text = "さよなら " * 200
    x, _y, info = parse_text_dataset(text, seq_len=32)

    assert info["vocab_size"] == TEXT_VOCAB_SIZE
    assert bytes(x.reshape(-1).tolist()) == text.encode("utf-8")[:info["tokens"]]


def test_a_byte_order_mark_is_not_learned_as_text():
    marked = "﻿" + PROSE
    x, _y, info = parse_text_dataset(marked, seq_len=16)

    assert info["source_bytes"] == len(PROSE.encode("utf-8"))
    assert bytes(x[0].tolist()) == PROSE.encode("utf-8")[:16]


def test_bytes_are_accepted_as_well_as_str():
    from_str = parse_text_dataset(PROSE, seq_len=16)[0]
    from_bytes = parse_text_dataset(PROSE.encode("utf-8"), seq_len=16)[0]

    assert np.array_equal(from_str, from_bytes)


def test_too_little_text_is_refused_with_the_amount_needed():
    with pytest.raises(ArtifactError) as excinfo:
        parse_text_dataset("hello", seq_len=32)

    message = str(excinfo.value)
    assert "too short" in message
    assert str(MIN_TEXT_WINDOWS * 32 + 1) in message.replace(",", "")


def test_an_unusable_sequence_length_is_refused():
    for seq_len in (1, 0, -8, 100_000):
        with pytest.raises(ArtifactError):
            parse_text_dataset(PROSE, seq_len=seq_len)


def test_the_dataset_packs_and_unpacks_unchanged():
    x, y, _ = parse_text_dataset(PROSE, seq_len=32)

    rt_x, rt_y = unpack_dataset(pack_dataset(x, y))

    assert np.array_equal(rt_x, x)
    assert np.array_equal(rt_y, y)


# --- deriving the model's shape from that data ---------------------------

def test_sequence_length_and_vocabulary_come_from_the_data():
    x, y, _ = parse_text_dataset(PROSE, seq_len=32)

    resolved = infer_spec_from_dataset(x, y, {"architecture": "transformer"})

    assert resolved["seq_len"] == 32
    # Only the alphabet this sample happens to use, unless told otherwise.
    assert resolved["vocab_size"] == int(max(x.max(), y.max())) + 1


def test_a_larger_stated_vocabulary_is_kept():
    # A tokeniser has ids that a given sample does not happen to contain;
    # shrinking to what was observed would break the model on new text.
    x, y, _ = parse_text_dataset(PROSE, seq_len=32)

    resolved = infer_spec_from_dataset(
        x, y, {"architecture": "transformer", "vocab_size": TEXT_VOCAB_SIZE}
    )

    assert resolved["vocab_size"] == TEXT_VOCAB_SIZE


def test_a_sequence_length_that_contradicts_the_data_is_refused():
    x, y, _ = parse_text_dataset(PROSE, seq_len=32)

    with pytest.raises(ValueError, match="seq_len"):
        infer_spec_from_dataset(x, y, {"architecture": "transformer", "seq_len": 64})


def test_one_label_per_row_is_refused_rather_than_reshaped():
    # The original bug: this used to reach cross_entropy and fail there, after
    # a contributor GPU had already been handed the job.
    x = np.zeros((32, 8), dtype=np.int32)
    y = np.zeros(32, dtype=np.int64)

    with pytest.raises(ValueError, match="2-D"):
        infer_spec_from_dataset(x, y, {"architecture": "transformer"})


def test_float_features_are_refused_as_token_ids():
    x = np.zeros((32, 8), dtype=np.float32)
    y = np.zeros((32, 8), dtype=np.float32)

    with pytest.raises(ValueError, match="whole-number"):
        infer_spec_from_dataset(x, y, {"architecture": "transformer"})


def test_negative_token_ids_are_refused():
    x = np.full((32, 8), -1, dtype=np.int64)
    y = np.zeros((32, 8), dtype=np.int64)

    with pytest.raises(ValueError, match="negative"):
        infer_spec_from_dataset(x, y, {"architecture": "transformer"})


# --- and sampling batches from it ----------------------------------------

def test_batches_keep_the_sequence_shape():
    torch = pytest.importorskip("torch")
    x, y, _ = parse_text_dataset(PROSE, seq_len=32)
    spec = infer_spec_from_dataset(x, y, {"architecture": "transformer"})

    make_batch, rows = dataset_sampler(x, y, spec)
    batch_x, batch_y = make_batch(8, torch.Generator().manual_seed(0),
                                  torch.device("cpu"))

    assert rows == x.shape[0]
    assert tuple(batch_x.shape) == (8, 32)
    assert tuple(batch_y.shape) == (8, 32)
    # Embedding lookups need int64 whatever the artifact stored.
    assert batch_x.dtype == torch.int64
    assert batch_y.dtype == torch.int64


def test_the_sampler_refuses_mismatched_shapes():
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="sequence length"):
        dataset_sampler(
            np.zeros((32, 8), dtype=np.int64),
            np.zeros(32, dtype=np.int64),
            {"architecture": "transformer"},
        )


def test_a_language_model_trains_on_real_text():
    """End to end on the CPU: the loss has to actually come down."""
    pytest.importorskip("torch")
    from backend.service.trainer import train

    x, y, _ = parse_text_dataset(PROSE, seq_len=16)
    task = {
        "model_name": "prose",
        "model_spec": {"architecture": "transformer", "d_model": 32,
                       "n_head": 2, "n_layer": 1},
        "hyperparameters": {"steps": 30, "batch_size": 8, "learning_rate": 0.003},
    }

    result = train(task, log=lambda _m: None, dataset=(x, y))
    metrics = result["metrics"]

    assert metrics["final_loss"] < metrics["initial_loss"]
    assert metrics["dataset_rows"] == x.shape[0]
    assert metrics["synthetic_data"] is False


def test_a_trained_language_model_can_be_scored_on_a_holdout():
    # Verification runs the same shape derivation, so a language model has to
    # survive the split that proves it learned something.
    pytest.importorskip("torch")
    from backend.service.trainer import train
    from backend.service.verification import evaluate, split_holdout

    x, y, _ = parse_text_dataset(PROSE, seq_len=16)
    train_x, train_y, holdout_x, holdout_y = split_holdout(x, y)

    task = {
        "model_name": "prose",
        "model_spec": {"architecture": "transformer", "d_model": 32,
                       "n_head": 2, "n_layer": 1, "vocab_size": TEXT_VOCAB_SIZE},
        "hyperparameters": {"steps": 40, "batch_size": 8, "learning_rate": 0.003},
    }
    result = train(task, log=lambda _m: None, dataset=(train_x, train_y))

    scored = evaluate(result["state_dict"], task["model_spec"],
                      holdout_x, holdout_y)

    assert 0.0 <= scored["accuracy"] <= 1.0
    assert scored["loss"] > 0.0


# --- what comes back has to say what the numbers meant -------------------

def test_the_manifest_records_the_tokenizer():
    manifest = build_manifest(
        {"architecture": "transformer", "seq_len": 32, "vocab_size": 256},
        {"head.bias": np.zeros(256, dtype=np.float32)},
        tokenizer="bytes",
    )

    assert manifest["tokenizer"]["kind"] == "bytes"
    assert manifest["tokenizer"]["vocab_size"] == 256
    assert manifest["tokenizer"]["encoding"] == "utf-8"


def test_a_model_without_a_tokenizer_does_not_claim_one():
    manifest = build_manifest(
        {"architecture": "mlp", "input_dim": 4, "output_dim": 3},
        {"0.bias": np.zeros(3, dtype=np.float32)},
    )

    assert "tokenizer" not in manifest


# --- the form has to offer values that can train -------------------------

def test_the_transformer_gets_its_own_starting_values():
    mlp, _ = validate_job({"model_spec": {"architecture": "mlp"}})
    transformer, _ = validate_job({"model_spec": {"architecture": "transformer"}})

    # 0.01 suits a two layer classifier and diverges a transformer.
    assert mlp["hyperparameters"]["learning_rate"] == 0.01
    assert transformer["hyperparameters"]["learning_rate"] < 0.01
    assert transformer["hyperparameters"]["steps"] > mlp["hyperparameters"]["steps"]


def test_a_stated_value_still_beats_the_default():
    clean, _ = validate_job({
        "model_spec": {"architecture": "transformer"},
        "hyperparameters": {"learning_rate": 0.02},
    })

    assert clean["hyperparameters"]["learning_rate"] == 0.02


def test_the_bounds_do_not_move_with_the_defaults():
    with pytest.raises(JobSpecError, match="between"):
        validate_job({
            "model_spec": {"architecture": "transformer"},
            "hyperparameters": {"learning_rate": 99},
        })


def test_the_form_no_longer_asks_for_what_the_data_decides():
    from backend.service.jobSpec import ARCHITECTURES

    transformer = ARCHITECTURES["transformer"]
    asked = {field["name"] for field in transformer["fields"]}

    assert "seq_len" not in asked
    assert "vocab_size" not in asked
    assert set(transformer["derived"]) == {"seq_len", "vocab_size"}


def test_each_architecture_declares_what_it_can_read():
    from backend.service.jobSpec import ARCHITECTURES

    assert ARCHITECTURES["mlp"]["accepts"] == "csv"
    assert ARCHITECTURES["transformer"]["accepts"] == "text"

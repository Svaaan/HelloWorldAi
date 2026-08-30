"""Asking a finished model to write something, without downloading it.

A trained language model came back as a number, a grade, and three
continuations of prompts the node had chosen. Finding out what it does with a
sentence of your own meant downloading the weights, installing torch and
running a script -- for a forward pass that takes two seconds on the machine
already holding the file.

These cover the generation itself. The bounds on prompt length and token count
live in the endpoint, because they are about what the coordinator's CPU will
agree to do rather than about the model.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service.trainer import (            # noqa: E402
    build_workload, continue_tokens, render_bytes, sample_text,
)

SPEC = {"architecture": "transformer", "d_model": 32, "n_head": 2,
        "n_layer": 1, "seq_len": 16, "vocab_size": 256}


def a_model(seed=0):
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    model = build_workload(SPEC)["factory"]()
    model.eval()
    return model


# --- bytes in, bytes out -------------------------------------------------

def test_bytes_render_back_to_the_text_they_came_from():
    text = "the coordinator held it back"
    assert render_bytes(list(text.encode("utf-8"))) == text


def test_a_split_character_does_not_raise():
    # The model works one byte at a time and can stop half way through a
    # multi-byte character. That has to render as a replacement, not a crash.
    ids = list("räksmörgås".encode("utf-8"))[:5]
    assert isinstance(render_bytes(ids), str)


def test_ids_outside_a_byte_are_folded_rather_than_thrown():
    assert render_bytes([65, 300, 66]) == "A" + chr(300 & 0xFF) + "B"


# --- continuing a prompt -------------------------------------------------

def test_it_returns_the_prompt_plus_what_it_wrote():
    model = a_model()
    ids = list("hello".encode("utf-8"))

    grown = continue_tokens(model, SPEC, ids, length=20)

    assert grown[:len(ids)] == ids          # the prompt is left alone
    assert len(grown) == len(ids) + 20


def test_every_token_is_inside_the_vocabulary():
    model = a_model()
    grown = continue_tokens(model, SPEC, list(b"hello"), length=40)

    assert all(0 <= token < SPEC["vocab_size"] for token in grown)


def test_a_prompt_longer_than_the_window_still_works():
    # The position embedding only reaches seq_len; anything longer has to be
    # cropped rather than indexed off the end.
    model = a_model()
    long_prompt = list(("x" * (SPEC["seq_len"] * 3)).encode("utf-8"))

    grown = continue_tokens(model, SPEC, long_prompt, length=5)

    assert len(grown) == len(long_prompt) + 5


def test_zero_temperature_is_repeatable():
    model = a_model()
    ids = list(b"same start")

    first = continue_tokens(model, SPEC, ids, length=15, temperature=0)
    second = continue_tokens(model, SPEC, ids, length=15, temperature=0)

    assert first == second


def test_sampling_is_not_repeatable():
    # Deliberately: always taking the most likely token makes a small model
    # repeat one phrase, which reads like a bug rather than a weak model.
    torch = pytest.importorskip("torch")
    model = a_model()
    ids = list(b"same start")

    torch.manual_seed(1)
    first = continue_tokens(model, SPEC, ids, length=30, temperature=1.0)
    torch.manual_seed(2)
    second = continue_tokens(model, SPEC, ids, length=30, temperature=1.0)

    assert first != second


def test_it_does_not_leave_the_model_in_training_mode():
    model = a_model()
    model.train()

    sample_text(model, SPEC, np.zeros((4, SPEC["seq_len"]), dtype=np.uint8))

    assert model.training     # restored to how it was found


# --- the samples packed with a finished job ------------------------------

def test_samples_carry_the_prompt_they_started_from():
    model = a_model()
    features = np.arange(4 * SPEC["seq_len"], dtype=np.uint8).reshape(4, SPEC["seq_len"])

    samples = sample_text(model, SPEC, features, count=2)

    assert len(samples) == 2
    for sample in samples:
        assert sample["prompt"]
        assert sample["continuation"]


def test_samples_are_drawn_from_across_the_data():
    # Taking the first rows of a sorted or structured file would show the same
    # thing three times.
    model = a_model()
    features = np.tile(np.arange(SPEC["seq_len"], dtype=np.uint8), (30, 1))
    features[:, 0] = np.arange(30, dtype=np.uint8)      # a marker per row

    samples = sample_text(model, SPEC, features, count=3)
    starts = {sample["prompt"][:1] for sample in samples}

    assert len(starts) == 3


def test_an_empty_dataset_yields_no_samples():
    model = a_model()
    assert sample_text(model, SPEC, np.zeros((0, SPEC["seq_len"]), dtype=np.uint8)) == []


def test_a_broken_model_costs_the_job_nothing():
    """A sample is a courtesy paid after the training is already done.

    It must never be the reason a finished job fails.
    """
    class Broken:
        training = False

        def eval(self):
            pass

        def parameters(self):
            raise RuntimeError("no parameters")

        def __call__(self, _x):
            raise RuntimeError("no forward pass")

    assert sample_text(Broken(), SPEC,
                       np.zeros((4, SPEC["seq_len"]), dtype=np.uint8)) == []

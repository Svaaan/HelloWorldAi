"""Four things a person sending data needs, found by being one.

These came out of running a real workload through the service rather than
reading the code: a stock-signal project that uploaded datasets, waited, and
collected models. Each check below is a thing that was wrong or missing when
that was done for the first time.

The one that matters most is the holdout. Verification takes a random slice,
which is correct for rows that are independent of each other and badly wrong for
anything recorded over time. Measured on this service, on the same weights:

    random holdout      54.2% against a 50.7% untrained floor
    graded on the end   51.7% against a 51.6% baseline

The first number is the one the submitter was shown, with an encouraging
sentence under it. There is no edge in that model at all.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service.verification import split_holdout        # noqa: E402
from backend.service.jobSpec import job_schema                # noqa: E402


# --- the holdout ----------------------------------------------------------

def ordered_data(rows=500):
    """Rows whose position carries meaning, as a time series does."""
    x = np.arange(rows, dtype=np.float32).reshape(-1, 1)
    y = (np.arange(rows) % 2).astype(np.int64)
    return x, y


def test_a_random_holdout_is_still_the_default():
    """Right for rows that are independent, which most datasets are."""
    x, y = ordered_data()
    train_x, _, holdout_x, _ = split_holdout(x, y, holdout_fraction=0.2, seed=1)

    held = set(holdout_x.ravel().tolist())
    # A random slice takes rows from throughout, so the earliest rows are not
    # all in the training half.
    assert min(held) < len(x) * 0.5, (
        "the default split no longer looks random; that is right for a series "
        "and wrong for everything else")


def test_ordered_data_is_graded_on_the_end():
    """Trained on the past, judged on the future -- the question actually asked."""
    x, y = ordered_data(rows=500)
    train_x, _, holdout_x, _ = split_holdout(
        x, y, holdout_fraction=0.2, seed=1, ordered=True)

    train_rows = train_x.ravel()
    held_rows = holdout_x.ravel()

    assert train_rows.max() < held_rows.min(), (
        "training rows appear after held-back rows. For a time series that "
        "means the model was graded on a day it had both neighbours of, which "
        "is why a random holdout reported 54.2% on a model with no edge.")

    # And nothing is shuffled: the order is the information.
    assert list(held_rows) == sorted(held_rows)
    assert list(train_rows) == sorted(train_rows)


def test_the_ordered_split_still_leaves_something_to_train_on():
    x, y = ordered_data(rows=10)
    train_x, train_y, holdout_x, _ = split_holdout(
        x, y, holdout_fraction=0.9, seed=0, ordered=True)

    assert len(train_x) >= 1 and len(holdout_x) >= 1
    assert len(train_x) == len(train_y)


def test_the_form_can_ask_whether_the_data_is_ordered():
    """A split nobody can choose is a split nobody uses."""
    questions = job_schema().get("data_questions") or []
    names = {q["name"] for q in questions}

    assert "time_ordered" in names, (
        "the schema offers no way to say the rows are in time order, so the "
        "honest split can never be selected")

    question = next(q for q in questions if q["name"] == "time_ordered")
    assert question["type"] == "bool"
    assert question.get("hint"), "an unexplained checkbox is not a question"


# --- explaining a wait ----------------------------------------------------

def test_the_reaper_thresholds_are_generous_but_finite():
    """A job on a switched-off machine is moved; it is not stuck forever.

    Worth pinning because it was misread once: a shallow look at
    requeue_stale_tasks suggested only running tasks were rescued, and the
    conclusion drawn was that pending jobs sit forever. They do not -- they are
    moved after NODE_GONE_MINUTES. What was actually missing was telling the
    submitter that during the wait.
    """
    from backend.routes.tasks import NODE_GONE_MINUTES

    assert 5 <= NODE_GONE_MINUTES <= 60, (
        "the window before a job is moved should be longer than a few missed "
        "heartbeats and short enough that nobody waits an hour")


def test_the_page_is_given_what_it_needs_to_explain_the_wait():
    """The arithmetic belongs on the server, where the threshold lives."""
    import inspect
    from backend.routes import tasks

    source = inspect.getsource(tasks._explain_the_wait)

    for field in ("silent_seconds", "moves_after_seconds", "can_be_moved"):
        assert field in source, (
            f"the page is not told {field}, so it cannot say when the job moves "
            f"without hardcoding a threshold it would not see change")


# --- estimating the time --------------------------------------------------

def test_throughput_refuses_to_guess_from_too_little():
    """An estimate invented from one run is worse than admitting there is none."""
    import inspect
    from backend.routes import tasks

    source = inspect.getsource(tasks.throughput)

    assert "len(rates) < 3" in source, (
        "throughput should decline to estimate from one or two jobs")
    assert '"why"' in source, (
        "when it declines it should say why, so the form prints a reason "
        "rather than a blank")


def test_the_estimate_comes_from_measurement_not_theory():
    """A card's theoretical TFLOPS is wrong by an order of magnitude here.

    A two-layer network at batch 32 spends its time on overhead rather than
    arithmetic, so the only honest source is what finished jobs actually did.
    """
    import ast
    import inspect
    from backend.routes import tasks

    source = inspect.getsource(tasks.throughput)
    assert "samples_per_second" in source

    # The docstring explains why theoretical rates are not used, so checking the
    # raw text would match its own explanation. Only the code counts.
    tree = ast.parse(source.strip())
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            node.value.value = ""          # blank out docstrings
    code = ast.unparse(tree)

    assert "tflops" not in code.lower(), (
        "the estimate is derived from a theoretical rate; on a two-layer "
        "network at batch 32 that is wrong by about a factor of ten")

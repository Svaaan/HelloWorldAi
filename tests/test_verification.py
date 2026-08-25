"""Tests for verifying returned work.

Most of these simulate a dishonest node. The honest path passing is necessary
but not interesting; what matters is that each way of faking a result is caught,
and that honest work is never wrongly condemned.

Requires torch and numpy; skips cleanly without them.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

try:
    import numpy as np
    import torch
    HAVE_DEPS = True
except ImportError:
    HAVE_DEPS = False

if HAVE_DEPS:
    from backend.service.trainer import train
    from backend.service.verification import (
        MIN_ACCURACY_MARGIN,
        VERDICT_ACCEPTED,
        VERDICT_REJECTED,
        VERDICT_SUSPICIOUS,
        baseline_accuracy,
        evaluate,
        untrained_reference,
        split_holdout,
        summarise,
        verify_training_result,
    )

SPEC = {"architecture": "mlp", "hidden_dim": 64, "depth": 2}


def separable(rows=500, features=8, classes=3, seed=0):
    rng = np.random.default_rng(seed)
    centroids = rng.normal(scale=4.0, size=(classes, features))
    labels = rng.integers(0, classes, size=rows)
    x = centroids[labels] + rng.normal(scale=0.5, size=(rows, features))
    return x.astype("float32"), labels.astype("int64")


def honest_run(train_x, train_y, steps=60):
    """What a node that actually does the work returns."""
    torch.manual_seed(0)
    return train({"model_spec": SPEC,
                  "hyperparameters": {"steps": steps, "batch_size": 64,
                                      "learning_rate": 1e-2}},
                 lambda _m: None, plan=None, dataset=(train_x, train_y))


def failed_check(report, name):
    return any(c["name"] == name and not c["passed"] for c in report["checks"])


def fake_weights(reference, seed=0):
    """Same shapes as a real result, pure noise inside. Seeded: an unseeded
    fake makes the verdict vary run to run and the test meaningless."""
    rng = np.random.default_rng(seed)
    return {name: rng.standard_normal(np.asarray(v).shape).astype("float32")
            for name, v in reference.items()}


# --- holdout splitting ---------------------------------------------------

def test_holdout_is_disjoint_from_the_training_half():
    x, y = separable(rows=100)
    tx, ty, hx, hy = split_holdout(x, y, holdout_fraction=0.2, seed=1)

    assert tx.shape[0] + hx.shape[0] == 100
    assert hx.shape[0] == 20

    # No holdout row may appear in the training half -- otherwise the node
    # could have tuned against the very data used to judge it.
    train_rows = {r.tobytes() for r in tx}
    assert not any(r.tobytes() in train_rows for r in hx)


def test_holdout_split_is_deterministic_per_seed():
    x, y = separable(rows=80)
    a = split_holdout(x, y, seed=7)[2]
    b = split_holdout(x, y, seed=7)[2]
    c = split_holdout(x, y, seed=8)[2]
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_split_always_leaves_something_to_train_on():
    x, y = separable(rows=10)
    tx, _ty, hx, _hy = split_holdout(x, y, holdout_fraction=0.99, seed=0)
    assert tx.shape[0] >= 1 and hx.shape[0] >= 1


def test_invalid_splits_are_rejected():
    x, y = separable(rows=50)
    for fraction in [0.0, 1.0, -0.2, 1.5]:
        try:
            split_holdout(x, y, holdout_fraction=fraction)
            assert False, "expected ValueError for %s" % fraction
        except ValueError:
            pass

    try:
        split_holdout(np.zeros((2, 3)), np.zeros(2))
        assert False, "expected ValueError for a tiny dataset"
    except ValueError:
        pass


def test_baseline_is_the_most_common_class():
    assert abs(baseline_accuracy(np.array([0, 0, 0, 1])) - 0.75) < 1e-9
    assert abs(baseline_accuracy(np.array([0, 1, 2])) - (1 / 3)) < 1e-9
    assert baseline_accuracy(np.array([])) == 0.0


# --- the honest path -----------------------------------------------------

def test_honest_training_is_accepted():
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=0)
    result = honest_run(tx, ty)

    report = verify_training_result(
        result["state_dict"], SPEC, hx, hy,
        claimed_metrics=result["metrics"], train_sample=(tx[:128], ty[:128]),
    )

    assert report["verdict"] == VERDICT_ACCEPTED, report
    assert report["measured"]["holdout_accuracy"] > report["measured"]["baseline_accuracy"]
    assert "accepted" in summarise(report)


def test_honest_work_generalises_to_data_it_never_saw():
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=0)
    result = honest_run(tx, ty)

    scored = evaluate(result["state_dict"], SPEC, hx, hy)
    assert scored["accuracy"] > 0.8, scored


# --- fabricated results --------------------------------------------------

def test_random_weights_with_an_invented_loss_are_rejected():
    """The classic fraud: return noise, claim a beautiful loss curve."""
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=0)
    honest = honest_run(tx, ty)

    fake = fake_weights(honest["state_dict"], seed=1)

    report = verify_training_result(
        fake, SPEC, hx, hy,
        claimed_metrics={"final_loss": 0.0001},
        train_sample=(tx[:128], ty[:128]),
    )

    assert report["verdict"] == VERDICT_REJECTED, report
    assert failed_check(report, "loss_claim_matches"), report["checks"]


def test_random_weights_with_an_honest_loss_are_at_least_flagged():
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=0)
    honest = honest_run(tx, ty)

    fake = fake_weights(honest["state_dict"], seed=2)
    measured = evaluate(fake, SPEC, tx[:128], ty[:128])

    report = verify_training_result(
        fake, SPEC, hx, hy,
        claimed_metrics={"final_loss": measured["loss"]},
        train_sample=(tx[:128], ty[:128]),
    )

    # Not provably fraud, but it learned nothing and must not pass silently.
    assert report["verdict"] == VERDICT_SUSPICIOUS, report
    assert (failed_check(report, "beats_baseline")
            or failed_check(report, "beats_untrained_loss")), report["checks"]


def test_returning_the_initial_weights_untouched_is_rejected():
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=0)

    torch.manual_seed(0)
    untrained = train({"model_spec": SPEC,
                       "hyperparameters": {"steps": 1, "batch_size": 8}},
                      lambda _m: None, plan=None, dataset=(tx, ty))["state_dict"]

    report = verify_training_result(
        untrained, SPEC, hx, hy,
        claimed_metrics={"final_loss": 0.001},
        initial_state=untrained,
    )

    assert report["verdict"] == VERDICT_REJECTED, report
    assert failed_check(report, "weights_changed")


def test_nan_weights_are_rejected():
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=0)
    honest = honest_run(tx, ty, steps=5)

    broken = {k: np.asarray(v).copy() for k, v in honest["state_dict"].items()}
    first = list(broken)[0]
    broken[first] = broken[first].astype("float32")
    broken[first].flat[0] = np.nan

    report = verify_training_result(broken, SPEC, hx, hy)
    assert report["verdict"] == VERDICT_REJECTED, report
    assert failed_check(report, "weights_finite")


def test_wrongly_shaped_weights_are_rejected():
    x, y = separable()
    _tx, _ty, hx, hy = split_holdout(x, y, seed=0)

    bogus = {"0.weight": np.zeros((3, 3), dtype="float32"),
             "0.bias": np.zeros(3, dtype="float32")}

    report = verify_training_result(bogus, SPEC, hx, hy)
    assert report["verdict"] == VERDICT_REJECTED, report
    assert failed_check(report, "weights_loadable")


def test_empty_weights_are_rejected():
    x, y = separable()
    _tx, _ty, hx, hy = split_holdout(x, y, seed=0)
    report = verify_training_result({}, SPEC, hx, hy)
    assert report["verdict"] == VERDICT_REJECTED
    assert failed_check(report, "weights_present")


def test_a_node_cannot_pass_by_memorising_the_holdout_it_never_received():
    """Training only on the training half still generalises; that is the point."""
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=3)
    result = honest_run(tx, ty)

    # The holdout rows were never in the training data.
    train_rows = {r.tobytes() for r in tx}
    assert not any(r.tobytes() in train_rows for r in hx)

    report = verify_training_result(result["state_dict"], SPEC, hx, hy,
                                    claimed_metrics=result["metrics"])
    assert report["verdict"] == VERDICT_ACCEPTED


def test_random_weights_can_beat_a_majority_class_baseline():
    """Why the untrained floor exists.

    On well-separated data a random projection classifies far better than
    guessing, so a majority-class baseline alone would pass fabricated weights.
    """
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=0)
    honest = honest_run(tx, ty, steps=5)
    fake = fake_weights(honest["state_dict"], seed=1)

    scored = evaluate(fake, SPEC, hx, hy)
    assert scored["accuracy"] > baseline_accuracy(hy), (
        "if this ever stops holding, the untrained-floor check is less critical, "
        "but it must never be weakened on the assumption that it does not")


def test_untrained_reference_is_the_stricter_floor():
    x, y = separable()
    _tx, _ty, hx, hy = split_holdout(x, y, seed=0)

    reference = untrained_reference(SPEC, hx, hy)
    assert reference["accuracy"] >= baseline_accuracy(hy) - 0.2
    assert reference["loss"] > 0


def test_honest_training_beats_the_untrained_reference_on_loss():
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=0)
    result = honest_run(tx, ty)

    trained = evaluate(result["state_dict"], SPEC, hx, hy)
    reference = untrained_reference(SPEC, hx, hy)
    assert trained["loss"] < reference["loss"], (trained, reference)


def test_verification_reports_the_untrained_reference_it_used():
    x, y = separable()
    tx, ty, hx, hy = split_holdout(x, y, seed=0)
    result = honest_run(tx, ty)

    report = verify_training_result(result["state_dict"], SPEC, hx, hy,
                                    claimed_metrics=result["metrics"])
    assert "untrained_accuracy" in report["measured"], report["measured"]
    assert "untrained_loss" in report["measured"], report["measured"]


# --- reporting -----------------------------------------------------------

def test_report_explains_every_failure():
    x, y = separable()
    _tx, _ty, hx, hy = split_holdout(x, y, seed=0)
    report = verify_training_result({"nonsense": np.zeros(3)}, SPEC, hx, hy)

    assert report["verdict"] == VERDICT_REJECTED
    for check in report["checks"]:
        assert check["detail"], check          # a node operator must see why
    assert "failed:" in summarise(report)


def test_margin_is_applied_not_just_a_bare_comparison():
    # A model that only ties the baseline must not scrape through.
    assert MIN_ACCURACY_MARGIN > 0


# --- standalone runner ---------------------------------------------------

def _main():
    if not HAVE_DEPS:
        print("  SKIP  torch/numpy not installed - verification tests not run")
        return 0

    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as e:
            failed.append(name)
            print("  FAIL  %s: %s" % (name, str(e)[:160]))
        except Exception as e:
            failed.append(name)
            print("  ERROR %s: %s: %s" % (name, type(e).__name__, str(e)[:160]))
    print("")
    summary = "%d/%d passed" % (len(tests) - len(failed), len(tests))
    if failed:
        summary += " -- FAILED: %s" % ", ".join(failed)
    print(summary)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())

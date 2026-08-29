"""Checking a job before a contributor's GPU is handed it.

Two failures motivated this. A malformed job used to be queued unread: it sat
through the approval window, was claimed, span up someone's card, and only then
failed -- so the submitter waited minutes to learn about a typo. And a value
that would not parse was not rejected but silently replaced by a default, so
`"steps": "5ooo"` became 20 steps and the job "succeeded" undertrained with
nothing saying why.

The same definition validates a submission and generates the form, so a third
failure -- a form offering fields the server rejects -- cannot happen either.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

from backend.service.jobSpec import (  # noqa: E402
    ARCHITECTURES,
    HYPERPARAMETERS,
    JobSpecError,
    job_schema,
    validate_job,
)


def good_job(**overrides):
    job = {
        "task_type": "llm_training",
        "model_name": "test",
        "model_spec": {"architecture": "mlp", "hidden_dim": 32, "depth": 2},
        "hyperparameters": {"steps": 100, "batch_size": 32, "learning_rate": 0.01},
    }
    job.update(overrides)
    return job


def refused(job):
    """The message a submitter would see, or None if it was accepted."""
    try:
        validate_job(job)
        return None
    except JobSpecError as e:
        return str(e)


# --- values that used to be silently replaced ------------------------------

def test_a_number_that_will_not_parse_is_refused_not_defaulted():
    """The bug this exists for: "5ooo" became 20 and the job looked fine."""
    message = refused(good_job(hyperparameters={"steps": "5ooo"}))
    assert message and "steps" in message and "5ooo" in message, message


def test_the_message_names_the_field_that_is_wrong():
    message = refused(good_job(hyperparameters={"batch_size": "big"}))
    assert "hyperparameters.batch_size" in message, message


def test_out_of_range_numbers_are_refused_with_the_range():
    for hyper, needle in (
        ({"steps": -5}, "steps"),
        ({"steps": 0}, "steps"),
        ({"batch_size": 999999}, "batch_size"),
        ({"learning_rate": 500}, "learning_rate"),
    ):
        message = refused(good_job(hyperparameters=hyper))
        assert message and needle in message, (hyper, message)
        assert "between" in message, message


def test_missing_values_fall_back_to_the_documented_default():
    cleaned, _ = validate_job(good_job(hyperparameters={}))
    defaults = {h["name"]: h["default"] for h in HYPERPARAMETERS}
    assert cleaned["hyperparameters"] == defaults


# --- structure -------------------------------------------------------------

def test_an_unknown_task_type_is_refused():
    message = refused(good_job(task_type="mine_bitcoin"))
    assert message and "mine_bitcoin" in message
    assert "llm_training" in message, "the message should say what is supported"


def test_an_unknown_architecture_is_refused_and_lists_the_real_ones():
    message = refused(good_job(model_spec={"architecture": "resnet"}))
    assert message and "resnet" in message
    assert "mlp" in message and "transformer" in message, message


def test_the_trainers_architecture_aliases_are_accepted():
    # trainer.py treats these as the same thing; the validator must agree or
    # it would reject jobs the node could happily run.
    for alias, expected in (("feedforward", "mlp"), ("tiny_lm", "transformer"),
                            ("tinylm", "transformer"), ("lm", "transformer")):
        cleaned, _ = validate_job(good_job(model_spec={"architecture": alias}))
        assert cleaned["model_spec"]["architecture"] == expected, alias


def test_a_non_object_spec_is_refused_rather_than_crashing():
    for bad in ("mlp", 42, []):
        assert refused(good_job(model_spec=bad)), bad
    for bad in ("fast", 7):
        assert refused(good_job(hyperparameters=bad)), bad


def test_an_empty_or_non_dict_job_is_refused():
    for bad in (None, {}, [], "job"):
        assert refused(bad), bad


def test_an_overlong_model_name_is_refused():
    assert refused(good_job(model_name="x" * 200))


def test_a_missing_name_gets_a_placeholder_rather_than_failing():
    cleaned, _ = validate_job(good_job(model_name="   "))
    assert cleaned["model_name"] == "model"


# --- the constraint torch enforces at runtime ------------------------------

def test_heads_that_do_not_divide_the_width_are_refused():
    """nn.TransformerEncoderLayer splits d_model across heads. Catching it
    here saves a job that would die inside the model on someone else's GPU."""
    message = refused(good_job(model_spec={
        "architecture": "transformer", "d_model": 100, "n_head": 3,
    }))
    assert message and "divide" in message, message


def test_a_width_that_does_divide_is_accepted():
    cleaned, _ = validate_job(good_job(model_spec={
        "architecture": "transformer", "d_model": 256, "n_head": 8,
    }))
    assert cleaned["model_spec"]["d_model"] == 256


# --- fields that come from the data ---------------------------------------

def test_dimensions_taken_from_the_dataset_are_flagged_not_honoured():
    # A stale input_dim would contradict the CSV and fail the run.
    _, notes = validate_job(good_job(model_spec={
        "architecture": "mlp", "input_dim": 999,
    }))
    assert any("input_dim" in n for n in notes), notes


def test_unknown_fields_are_passed_through_with_a_note():
    cleaned, notes = validate_job(good_job(future_option="x"))
    assert cleaned["future_option"] == "x"
    assert any("future_option" in n for n in notes), notes


def test_dataset_id_is_not_carried_into_the_cleaned_job():
    # The coordinator takes it out separately; leaving a copy in task_data
    # would store the id in two places that could disagree.
    cleaned, _ = validate_job(good_job(dataset_id="abc123"))
    assert "dataset_id" not in cleaned


# --- the schema the form is built from ------------------------------------

def test_every_field_the_form_offers_survives_validation():
    """The form is generated from this schema, so its defaults must be
    accepted -- otherwise the page offers a job the server refuses."""
    schema = job_schema()
    for name, definition in schema["architectures"].items():
        spec = {"architecture": name}
        spec.update({f["name"]: f["default"] for f in definition["fields"]})
        hyper = {h["name"]: h["default"] for h in schema["hyperparameters"]}
        cleaned, _ = validate_job(good_job(model_spec=spec, hyperparameters=hyper))
        assert cleaned["model_spec"]["architecture"] == name


def test_every_field_declares_the_bounds_the_form_needs():
    for name, definition in ARCHITECTURES.items():
        for field in definition["fields"]:
            for key in ("name", "label", "type", "default", "min", "max"):
                assert key in field, (name, field.get("name"), key)
            assert field["min"] <= field["default"] <= field["max"], field


def test_defaults_sit_inside_their_own_bounds():
    for field in HYPERPARAMETERS:
        assert field["min"] <= field["default"] <= field["max"], field


# --- standalone runner ---------------------------------------------------

def _main():
    tests = [(n, o) for n, o in sorted(globals().items())
             if n.startswith('test_') and callable(o)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("  PASS  %s" % name)
        except AssertionError as e:
            failed.append(name)
            print("  FAIL  %s: %s" % (name, e))
        except Exception as e:
            failed.append(name)
            print("  ERROR %s: %s: %s" % (name, type(e).__name__, e))
    print("")
    summary = "%d/%d passed" % (len(tests) - len(failed), len(tests))
    if failed:
        summary += " -- FAILED: %s" % ", ".join(failed)
    print(summary)
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(_main())

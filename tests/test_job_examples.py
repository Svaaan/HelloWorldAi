"""Every number the send-work form asks for comes with an example.

The form told you what each box meant and nothing about what to put in it.
"Neurons per hidden layer." is an accurate description of `hidden_dim` and does
not help somebody who came to find out what this is: they cannot tell whether
64 is small, ordinary or absurd, so they either leave every default alone or
pick a number out of the air, and then judge the service on whatever comes
back.

The step count already looks after itself -- choosing a file raises it so the
run at least reads the data, and a line under the form says how many passes
that is. Nothing guided the shape of the model at all.

These check the examples exist and do not contradict the values they sit under,
which is the way this particular kind of help goes wrong: a transformer's
learning rate is overridden to 0.0005 because 0.01 destroys it in a dozen
steps, and an example still reading "0.01 for a feedforward network" under that
box would be advice to do the one thing the override exists to prevent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from backend.service.jobSpec import (          # noqa: E402
    ARCHITECTURES,
    HYPERPARAMETERS,
    job_schema,
)


def architecture_fields():
    for key, definition in ARCHITECTURES.items():
        for spec in definition.get("fields", []):
            yield key, spec


@pytest.mark.parametrize("name,spec", [(s["name"], s) for s in HYPERPARAMETERS])
def test_every_training_setting_shows_an_example(name, spec):
    assert spec.get("example"), (
        "%s has no example. Someone who has not trained a model before has "
        "nothing to go on but the number already in the box." % name
    )


@pytest.mark.parametrize(
    "arch,spec",
    [(a, s) for a, s in architecture_fields()],
    ids=[f"{a}.{s['name']}" for a, s in architecture_fields()],
)
def test_every_model_setting_shows_an_example(arch, spec):
    assert spec.get("example"), (
        "%s.%s has no example, and this is the half of the form nothing else "
        "guides -- the step count is raised from the uploaded file, but the "
        "shape of the model is left entirely to the reader."
        % (arch, spec["name"])
    )


def test_an_overridden_default_gets_an_overridden_example():
    """A model that changes a starting value has to change its advice too.

    The transformer sets learning_rate to 0.0005 because 0.01 blows it up in
    the first dozen steps. The shared example says "0.01 for a feedforward
    network", which under that box would be a recommendation to do exactly what
    the override prevents.
    """
    shared = {spec["name"]: spec for spec in HYPERPARAMETERS}

    for key, definition in ARCHITECTURES.items():
        overridden = definition.get("hyperparameter_defaults", {})
        examples = definition.get("hyperparameter_examples", {})

        for name in overridden:
            assert name in examples, (
                "%s overrides the default for %s but keeps the shared example, "
                "which describes a different value: %r"
                % (key, name, shared[name].get("example"))
            )


def test_an_example_never_recommends_a_value_the_field_rejects():
    """An example naming a number outside min/max is a trap, not a hint."""
    import re

    def numbers(text):
        for match in re.findall(r"\d+(?:\.\d+)?", text.replace(",", "")):
            yield float(match)

    checked = 0
    for spec in HYPERPARAMETERS:
        example = spec.get("example") or ""
        # Only the numbers offered as values, not prose like "a dozen steps".
        # Anything below the minimum or above the maximum could not be sent.
        for value in numbers(example):
            # Prose numbers (minutes, counts) are not proposals; the check that
            # matters is that nothing named is outright unusable.
            if value < spec["min"] or value > spec["max"]:
                pytest.fail(
                    "%s suggests %g, which is outside the accepted range "
                    "%g to %g" % (spec["name"], value, spec["min"], spec["max"])
                )
            checked += 1

    assert checked, "no numbers were checked; the examples lost their values"


def test_the_schema_carries_the_examples_to_the_form():
    """The form builds itself from /job-schema, so they have to survive it."""
    schema = job_schema()

    for spec in schema["hyperparameters"]:
        assert spec.get("example"), (
            "%s reaches the form without its example" % spec["name"])

    for key, definition in schema["architectures"].items():
        for spec in definition.get("fields", []):
            assert spec.get("example"), (
                "%s.%s reaches the form without its example"
                % (key, spec["name"]))


# The fields sit in a grid of roughly 150px columns. An example that runs to a
# sentence and a half wraps into a tall narrow strip and pushes the rest of the
# form out of the dialog -- which is what the first version of these did.
MAX_EXAMPLE_CHARS = 60


def all_examples():
    for spec in HYPERPARAMETERS:
        yield spec["name"], spec.get("example")
    for arch, spec in architecture_fields():
        yield f"{arch}.{spec['name']}", spec.get("example")
    for arch, definition in ARCHITECTURES.items():
        for name, example in definition.get(
                "hyperparameter_examples", {}).items():
            yield f"{arch}.{name}", example


@pytest.mark.parametrize("name,example", list(all_examples()),
                         ids=[n for n, _ in all_examples()])
def test_an_example_fits_where_it_is_shown(name, example):
    assert example and len(example) <= MAX_EXAMPLE_CHARS, (
        "%s: %d characters. These render in a ~150px grid column, so anything "
        "much longer becomes a tall narrow strip that pushes the form out of "
        "the dialog. Say what to type here; explain it in the hint.\n  %r"
        % (name, len(example or ""), example)
    )


def test_no_field_carries_a_key_the_form_does_not_use():
    """A typo in a spec is silent: the form ignores it and nothing complains."""
    known = {"name", "label", "type", "default", "min", "max", "hint",
             "example"}

    for spec in HYPERPARAMETERS:
        unexpected = set(spec) - known
        assert not unexpected, (
            "%s carries %s, which nothing reads"
            % (spec["name"], ", ".join(sorted(unexpected))))

    for arch, spec in architecture_fields():
        unexpected = set(spec) - known
        assert not unexpected, (
            "%s.%s carries %s, which nothing reads"
            % (arch, spec["name"], ", ".join(sorted(unexpected))))

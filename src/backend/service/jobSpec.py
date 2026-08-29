"""What a job may ask for, and whether a given request makes sense.

Why this exists
---------------
A job used to be a JSON blob typed into a textarea and queued unread. Two things
went wrong with that.

First, nothing was checked before queueing. A typo was accepted, sat through the
approval window, got claimed by a contributor, spun up their GPU, and only then
failed. The submitter waited minutes to be told about a misspelled key.

Second, bad values were not rejected but silently replaced: the trainer falls
back to a default whenever a number will not parse, so `"steps": "5ooo"` quietly
became 20 steps. The job "succeeded" and the model was undertrained, with
nothing anywhere saying why.

So the rules live here, in one place, and are used twice: to check a submission
before it is queued, and to describe the form the submitter fills in. A single
definition means the form and the validator cannot drift apart.
"""

from typing import Any, Dict, List, Optional, Tuple

TASK_TYPES = ("llm_training",)
DEFAULT_TASK_TYPE = "llm_training"

MAX_NAME_CHARS = 80


class JobSpecError(ValueError):
    """The submitted job is not something a node could run."""


def _number(name, value, *, minimum, maximum, integer=True, default=None):
    """Parse one field, or say precisely what is wrong with it."""
    if value is None or value == "":
        if default is None:
            raise JobSpecError(f"{name} is required.")
        return default

    try:
        parsed = int(value) if integer else float(value)
    except (TypeError, ValueError):
        kind = "whole number" if integer else "number"
        raise JobSpecError(f"{name} must be a {kind}; got {value!r}.")

    if parsed < minimum or parsed > maximum:
        raise JobSpecError(f"{name} must be between {minimum} and {maximum}; got {parsed}.")

    return parsed


# Every field the form offers and the validator accepts. `hint` is shown under
# the input; `derived` fields are filled in from the dataset on the node, so the
# form does not ask for them.
ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "label": "Feedforward network (MLP)",
        "summary": "Classifies rows of numbers. Use this for CSV data.",
        "accepts": "csv",
        "fields": [
            {"name": "hidden_dim", "label": "Hidden width", "type": "int",
             "default": 64, "min": 1, "max": 16384,
             "hint": "Neurons per hidden layer."},
            {"name": "depth", "label": "Hidden layers", "type": "int",
             "default": 2, "min": 1, "max": 64,
             "hint": "How many hidden layers to stack."},
        ],
        "derived": ["input_dim", "output_dim"],
        "derived_note": "Inputs and classes are read from your CSV.",
    },
    "transformer": {
        "label": "Small transformer",
        "summary": "Learns to continue text. Use this for a .txt file.",
        "accepts": "text",
        "fields": [
            {"name": "d_model", "label": "Model width", "type": "int",
             "default": 256, "min": 8, "max": 8192,
             "hint": "Must divide evenly by the head count."},
            {"name": "n_head", "label": "Attention heads", "type": "int",
             "default": 4, "min": 1, "max": 128},
            {"name": "n_layer", "label": "Layers", "type": "int",
             "default": 2, "min": 1, "max": 96},
        ],
        # Sequence length and vocabulary are properties of the uploaded text,
        # not choices: the position embedding has to be exactly as long as the
        # sequences and the token embedding has to cover every id in them. The
        # form used to ask for both, and whatever was typed was contradicted by
        # the data on the node.
        "derived": ["seq_len", "vocab_size"],
        "derived_note": "Sequence length and vocabulary are read from your text.",
        # A transformer wants a far smaller step than a two-layer classifier:
        # at 0.01 the attention weights blow up in the first dozen steps and
        # the loss never comes back. The defaults below were shared by both
        # models, which was harmless while this one only ever ran on synthetic
        # data -- and is a job that cannot work now that it trains on real
        # text. It also needs more steps, because it is learning a language
        # rather than a decision boundary.
        "hyperparameter_defaults": {
            "learning_rate": 0.0005,
            "steps": 1000,
        },
    },
}

HYPERPARAMETERS: List[Dict[str, Any]] = [
    {"name": "steps", "label": "Training steps", "type": "int",
     "default": 200, "min": 1, "max": 1_000_000,
     "hint": "More steps means a better model and a longer job."},
    {"name": "batch_size", "label": "Batch size", "type": "int",
     "default": 32, "min": 1, "max": 8192,
     "hint": "Samples per step, split across the node's GPUs."},
    {"name": "learning_rate", "label": "Learning rate", "type": "float",
     "default": 0.01, "min": 1e-8, "max": 10.0},
]


def job_schema() -> Dict[str, Any]:
    """The whole contract, for building a form against."""
    return {
        "task_types": list(TASK_TYPES),
        "default_task_type": DEFAULT_TASK_TYPE,
        "architectures": ARCHITECTURES,
        "hyperparameters": HYPERPARAMETERS,
    }


# How many times a run will go over the whole dataset. Past roughly this many
# passes a language model stops generalising and starts memorising: it reports
# a beautiful training loss and does no better on anything it has not already
# seen.
#
# Deliberately not applied to the classifier. Measured on this service: 240
# rows of well-separated data, 53 passes, and the result scored 100% on the
# holdout -- many epochs over a small table is normal practice, not a mistake.
# Warning there would be noise, and overfitting a table shows up in the
# strength grade afterwards, where it is measured rather than guessed.
MAX_COMFORTABLE_PASSES = 12

# And the other end -- a run that does not finish one pass has not shown the
# model most of the data it was given.
MIN_USEFUL_PASSES = 1.0


def advise(job: Dict[str, Any], dataset_info: Optional[Dict[str, Any]] = None) -> List[str]:
    """Things worth saying about a job that is nonetheless valid.

    Separate from validation on purpose: none of this makes a job wrong, so
    none of it should refuse one. It is the arithmetic the submitter cannot do
    in their head, said before a contributor's GPU spends an hour on it.
    """
    notes: List[str] = []

    rows = (dataset_info or {}).get("rows")
    if not rows:
        return notes

    architecture = str((job.get("model_spec") or {}).get("architecture", "mlp")).lower()
    if (ARCHITECTURES.get(architecture) or {}).get("accepts") != "text":
        return notes

    hyper = job.get("hyperparameters") or {}
    steps = int(hyper.get("steps") or 0)
    batch = int(hyper.get("batch_size") or 0)
    if steps <= 0 or batch <= 0:
        return notes

    passes = (steps * batch) / float(rows)

    if passes > MAX_COMFORTABLE_PASSES:
        notes.append(
            f"This run goes over your {int(rows):,} rows about {passes:.0f} times. "
            f"Past roughly {MAX_COMFORTABLE_PASSES} the model tends to memorise "
            f"rather than learn — the training loss keeps falling while the "
            f"result gets no better. Fewer steps, or more data, would help."
        )
    elif passes < MIN_USEFUL_PASSES:
        notes.append(
            f"This run sees only about {passes * 100:.0f}% of your "
            f"{int(rows):,} rows. More steps would let it read the rest."
        )

    return notes


def _validate_fields(fields, supplied, where, overrides=None) -> Dict[str, Any]:
    """Check each field, falling back to the architecture's default then the
    shared one. The bounds never move -- only what a blank box means."""
    overrides = overrides or {}

    clean = {}
    for field in fields:
        fallback = overrides.get(field["name"], field["default"])
        clean[field["name"]] = _number(
            f"{where}.{field['name']}",
            supplied.get(field["name"], fallback),
            minimum=field["min"],
            maximum=field["max"],
            integer=field["type"] == "int",
            default=fallback,
        )
    return clean


def validate_job(task_data: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Check a submitted job, returning (cleaned job, notes).

    Raises JobSpecError with a message meant for the person who typed it.
    """
    if not isinstance(task_data, dict) or not task_data:
        raise JobSpecError("A job must be a JSON object.")

    notes: List[str] = []

    task_type = task_data.get("task_type") or DEFAULT_TASK_TYPE
    if task_type not in TASK_TYPES:
        raise JobSpecError(
            f"Unknown task type {task_type!r}. Supported: {', '.join(TASK_TYPES)}."
        )

    name = str(task_data.get("model_name") or "").strip()
    if len(name) > MAX_NAME_CHARS:
        raise JobSpecError(f"model_name is longer than {MAX_NAME_CHARS} characters.")

    spec = task_data.get("model_spec")
    if spec is None or spec == "":
        spec = {}
    if not isinstance(spec, dict):
        raise JobSpecError("model_spec must be a JSON object.")

    architecture = str(spec.get("architecture") or "mlp").lower()
    # The trainer accepts several spellings; normalise so the rest of this
    # function has one name to reason about.
    aliases = {"feedforward": "mlp", "tiny_lm": "transformer",
               "tinylm": "transformer", "lm": "transformer"}
    architecture = aliases.get(architecture, architecture)

    if architecture not in ARCHITECTURES:
        raise JobSpecError(
            f"Unknown architecture {architecture!r}. "
            f"Supported: {', '.join(sorted(ARCHITECTURES))}."
        )

    definition = ARCHITECTURES[architecture]
    clean_spec = _validate_fields(definition["fields"], spec, "model_spec")
    clean_spec["architecture"] = architecture

    # torch splits the embedding across heads, so an indivisible pair fails
    # inside the model rather than here -- after a contributor's GPU has
    # already been handed the job.
    if architecture == "transformer":
        if clean_spec["d_model"] % clean_spec["n_head"] != 0:
            raise JobSpecError(
                f"Model width {clean_spec['d_model']} must divide evenly by "
                f"{clean_spec['n_head']} attention heads."
            )

    # Dimensions that come from the data are filled in on the node; keeping a
    # stale value here would contradict the dataset and fail the run.
    for key in definition.get("derived", []):
        if key in spec:
            notes.append(f"{key} is taken from your dataset; the value you gave was ignored.")

    hyper = task_data.get("hyperparameters")
    if hyper is None or hyper == "":
        hyper = {}
    if not isinstance(hyper, dict):
        raise JobSpecError("hyperparameters must be a JSON object.")

    clean_hyper = _validate_fields(
        HYPERPARAMETERS, hyper, "hyperparameters",
        definition.get("hyperparameter_defaults"),
    )

    cleaned = {
        "task_type": task_type,
        "model_name": name or "model",
        "model_spec": clean_spec,
        "hyperparameters": clean_hyper,
    }

    # Anything we do not recognise is passed through rather than dropped, so a
    # newer field does not silently disappear on an older coordinator.
    for key, value in task_data.items():
        if key not in cleaned and key != "dataset_id":
            cleaned[key] = value
            notes.append(f"{key} is not a field this coordinator knows; passing it through.")

    return cleaned, notes

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
        "summary": "A causal language model over token ids.",
        "fields": [
            {"name": "d_model", "label": "Model width", "type": "int",
             "default": 256, "min": 8, "max": 8192,
             "hint": "Embedding size. Must divide evenly by the head count."},
            {"name": "n_head", "label": "Attention heads", "type": "int",
             "default": 4, "min": 1, "max": 128},
            {"name": "n_layer", "label": "Layers", "type": "int",
             "default": 2, "min": 1, "max": 96},
            {"name": "seq_len", "label": "Sequence length", "type": "int",
             "default": 64, "min": 2, "max": 8192},
            {"name": "vocab_size", "label": "Vocabulary size", "type": "int",
             "default": 1024, "min": 2, "max": 1_000_000},
        ],
        "derived": [],
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


def _validate_fields(fields, supplied, where) -> Dict[str, Any]:
    clean = {}
    for field in fields:
        clean[field["name"]] = _number(
            f"{where}.{field['name']}",
            supplied.get(field["name"], field["default"]),
            minimum=field["min"],
            maximum=field["max"],
            integer=field["type"] == "int",
            default=field["default"],
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

    clean_hyper = _validate_fields(HYPERPARAMETERS, hyper, "hyperparameters")

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

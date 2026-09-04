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

import math
import re
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


TRUE = ("true", "yes", "on", "1")
FALSE = ("false", "no", "off", "0", "")


def _boolean(name, value, *, default=False):
    """Parse one yes/no field, or say precisely what is wrong with it.

    Strings are accepted because not every caller sends JSON booleans -- a form
    post and a hand-written client both tend to send "true". A value that means
    nothing is an error rather than a quiet False: these answers change how the
    result is judged, and guessing wrong is exactly the failure the field
    exists to prevent.
    """
    if value is None:
        return default

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in TRUE:
            return True
        if lowered in FALSE:
            return False

    raise JobSpecError(
        f"{name} must be true or false; got {value!r}.")


# Every field the form offers and the validator accepts. `hint` is shown under
# the input; `derived` fields are filled in from the dataset on the node, so the
# form does not ask for them.
#
# `example` is shown beside the hint, and exists for the person who came to find
# out what this is. The step count looks after itself -- it is raised from the
# file when one is chosen, so that a run at least reads the data -- but nothing
# guided the shape of the model at all. "Neurons per hidden layer." tells you
# what the box means and not one thing about whether 64 is small, ordinary or
# absurd, and somebody with no reason to know will either leave every default
# alone or pick a number out of the air.
#
# These are starting points, not measurements: conventional values that train
# something recognisable on a first attempt. Where this project has actually
# measured a number it says so in the comment beside it.
#
# Keep them to a line. These fields sit in a grid roughly 150px wide, so an
# example that runs to a sentence and a half wraps into a tall narrow column and
# pushes the rest of the form out of the dialog. Anything that needs explaining
# rather than stating belongs in the hint above it.
ARCHITECTURES: Dict[str, Dict[str, Any]] = {
    "mlp": {
        "label": "Feedforward network (MLP)",
        "summary": "Classifies rows of numbers. Use this for CSV data.",
        "accepts": "csv",
        "fields": [
            {"name": "hidden_dim", "label": "Hidden width", "type": "int",
             "default": 64, "min": 1, "max": 16384,
             "hint": "Neurons per hidden layer.",
             "example": "64 for a few columns, 256 for wider data."},
            {"name": "depth", "label": "Hidden layers", "type": "int",
             "default": 2, "min": 1, "max": 64,
             "hint": "How many hidden layers to stack.",
             "example": "2. Deeper rarely helps on rows of numbers."},
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
             "hint": "Must divide evenly by the head count.",
             "example": "256 for a small model, 384 to learn more."},
            {"name": "n_head", "label": "Attention heads", "type": "int",
             "default": 4, "min": 1, "max": 128,
             "example": "4, and the width must divide by it."},
            {"name": "n_layer", "label": "Layers", "type": "int",
             "default": 2, "min": 1, "max": 96,
             "example": "2 to start; each one adds training time."},
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
        # The shared examples describe the feedforward model, and two of them
        # are actively wrong here: 0.01 is the rate that blows this model up in
        # a dozen steps, which is the reason for the override above. An example
        # that contradicts the value sitting in the box beside it is worse than
        # none, so the ones that differ are replaced rather than left to be read
        # as advice.
        "hyperparameter_examples": {
            "learning_rate": "0.0005. At 0.01 this model does not recover.",
            "steps": "Raised from your file; text needs more than a CSV.",
        },
    },
}

HYPERPARAMETERS: List[Dict[str, Any]] = [
    {"name": "steps", "label": "Training steps", "type": "int",
     "default": 200, "min": 1, "max": 1_000_000,
     # The measured rate lives here rather than in the example: it is the
     # thing worth knowing before spending somebody else's card, and the
     # example next to it has one line to say what to type.
     "hint": "More steps means a better model and a longer job — about 1,000 "
             "a minute on a mid-range card.",
     # This one is not left to the reader: choosing a file raises it to at
     # least read the data. The example says so, because 200 sitting in the box
     # before anything is uploaded looks like the number you are meant to send.
     "example": "Raised from your file when you choose one."},
    {"name": "batch_size", "label": "Batch size", "type": "int",
     "default": 32, "min": 1, "max": 8192,
     "hint": "Samples per step, split across the node's GPUs.",
     "example": "32 is safe; 64 or 128 is faster on a big card."},
    {"name": "learning_rate", "label": "Learning rate", "type": "float",
     "default": 0.01, "min": 1e-8, "max": 10.0,
     "hint": "How big a correction each step makes.",
     # The transformer overrides both this and its example; see
     # hyperparameter_examples on that architecture.
     "example": "0.01 here. Lower to 0.001 if the loss jumps."},
]


# One question about the data rather than the model, which is why it is not in
# HYPERPARAMETERS: it changes how the result is judged, not how it is trained.
#
# The verification holdout is a random slice by default, and for rows recorded
# over time that is a much easier question than the one the submitter means to
# ask -- see split_holdout. Answering yes here holds back the end of the data
# instead, so the model is trained on the past and graded on the future.
DATA_QUESTIONS: List[Dict[str, Any]] = [
    {
        "name": "time_ordered",
        "label": "These rows are sorted oldest first",
        "type": "bool",
        "default": False,
        # "in time order" was the first wording and it was read as "this is a
        # time series" rather than "the rows are sorted". Somebody sent a panel
        # of ten companies stacked one after another -- each internally in date
        # order, the file as a whole not -- and ticking the box held back the
        # tail of the last company instead of the most recent period. The score
        # came back at 54.9%, indistinguishable from the random slice it was
        # meant to replace, and nothing could have told them.
        "hint": "Prices, logs, sales — and the file itself runs oldest to "
                "newest, not grouped by product or symbol.",
        "example": "Checked: judged on the newest rows, not a random slice.",
    },
]


def job_schema() -> Dict[str, Any]:
    """The whole contract, for building a form against."""
    return {
        "task_types": list(TASK_TYPES),
        "default_task_type": DEFAULT_TASK_TYPE,
        "architectures": ARCHITECTURES,
        "hyperparameters": HYPERPARAMETERS,
        "data_questions": DATA_QUESTIONS,
        # So the form can show the same arithmetic the coordinator will,
        # before the job is sent rather than in the reply after it.
        "guidance": {
            "target_passes": TARGET_PASSES,
            "min_coverage": MIN_COVERAGE,
            "max_suggested_steps": MAX_SUGGESTED_STEPS,
        },
    }


# How many passes over the data a run should make at minimum.
#
# There was a warning here for the other end too -- "past about 12 passes the
# model memorises rather than learns" -- and measuring it on this service
# proved it wrong. Same 897 sequences, same model, only the step count changed:
#
#     8 passes   train loss 2.14   holdout accuracy 0.349   captured 0.225
#    36 passes   train loss 0.99   holdout accuracy 0.407   captured 0.295
#
# Training four times longer over the same small corpus made the held-back
# score better, not worse. The textbook intuition about epochs did not survive
# contact with the measurement, so the warning is gone rather than softened.
# What was actually wrong in that run was the size of the corpus, and there is
# a warning for that at upload, where it is grounded.
#
# This floor stays because it is close to arithmetic rather than a claim about
# learning: a run drawing far fewer samples than there are rows leaves most of
# the data unused.
TARGET_PASSES = 3.0

# Sampling is with replacement, so n draws from N rows do not touch n distinct
# rows. The expected share actually seen is 1 - e^(-n/N), which at low ratios
# is close to n/N and at high ratios is not.
MIN_COVERAGE = 0.95

# A suggestion should not quietly propose an hour of somebody else's GPU.
# Measured on an RTX 3070: about 1,000 steps a minute for the default model.
MAX_SUGGESTED_STEPS = 20_000


def _coverage(samples: float, rows: float) -> float:
    """Expected share of distinct rows drawn, sampling with replacement."""
    if rows <= 0:
        return 0.0
    return 1.0 - math.exp(-samples / rows)


def suggest_steps(rows: int, batch_size: int, default: int) -> int:
    """A step count that at least reads the data the submitter uploaded.

    Only ever raises the default. Lowering it was the obvious other half and
    the measurement above says not to: on a corpus too small to learn from,
    training less produces a worse model, not a less overfitted one.
    """
    if rows <= 0 or batch_size <= 0:
        return default

    wanted = math.ceil((TARGET_PASSES * rows) / batch_size)
    return int(max(default, min(wanted, MAX_SUGGESTED_STEPS)))


# A re-run that keeps its parent's name leaves a list of identical labels, and
# the download it produces overwrites the one before it. Runs are a sequence,
# so they should be named like one.
#
# A dash rather than a space: the name becomes a filename, and
# `load_model.py my-model v2-task_1234.npz` is two arguments.
RUN_SUFFIX = re.compile(r"-v(\d+)$")


def next_run_name(parent_name: str, taken: Optional[List[str]] = None) -> str:
    """The next name in a series, given what the earlier runs were called.

    Numbers follow the family rather than the chain, so two adjustments of the
    same run do not both come out as v2.
    """
    base = RUN_SUFFIX.sub("", str(parent_name or "model").strip()) or "model"

    highest = 1
    for name in (taken or []):
        name = str(name).strip()
        if name == base:
            continue
        match = RUN_SUFFIX.match(name[len(base):]) if name.startswith(base) else None
        if match:
            highest = max(highest, int(match.group(1)))

    suffix = f"-v{highest + 1}"
    # The name has a length limit, and the suffix is the part that matters.
    return base[:MAX_NAME_CHARS - len(suffix)] + suffix


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

    hyper = job.get("hyperparameters") or {}
    steps = int(hyper.get("steps") or 0)
    batch = int(hyper.get("batch_size") or 0)
    if steps <= 0 or batch <= 0:
        return notes

    seen = _coverage(steps * batch, rows)
    if seen < MIN_COVERAGE:
        notes.append(
            f"This run draws {steps * batch:,} samples from {int(rows):,} rows, "
            f"so it will only reach about {seen * 100:.0f}% of your data. "
            f"More steps would use the rest."
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

    # The questions about the data rather than the model. They live at the top
    # level because that is where they are read from -- prepare_dataset_split
    # takes task_data["time_ordered"] straight off the job.
    #
    # They were declared, published in /job-schema, drawn in the form and acted
    # on by the coordinator, and never validated here. So every job that
    # answered one came back with
    #
    #     time_ordered is not a field this coordinator knows; passing it through
    #
    # which is the note for a field from a newer client -- alarming, wrong, and
    # sitting under a job that had in fact been understood perfectly.
    for question in DATA_QUESTIONS:
        cleaned[question["name"]] = _boolean(
            f"data.{question['name']}",
            task_data.get(question["name"]),
            default=question.get("default", False),
        )

    # Anything we do not recognise is passed through rather than dropped, so a
    # newer field does not silently disappear on an older coordinator.
    for key, value in task_data.items():
        if key not in cleaned and key != "dataset_id":
            cleaned[key] = value
            notes.append(f"{key} is not a field this coordinator knows; passing it through.")

    return cleaned, notes

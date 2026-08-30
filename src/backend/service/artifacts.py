"""Safe serialisation for anything that crosses the network.

Threat model
------------
Datasets travel from whoever needs compute to a stranger's machine, and trained
weights travel back. Both directions are untrusted.

`pickle.loads` and `torch.load` execute arbitrary code while deserialising. Using
either on network data would turn this network into a remote code execution
service in both directions -- a malicious job would own the contributor's
machine, and a malicious contributor would own the submitter's.

So everything on the wire is a numpy `.npz` archive loaded with
`allow_pickle=False`, which is pure data and cannot execute anything. Object
arrays are rejected explicitly rather than trusted to be harmless.

This module deliberately imports numpy but not torch, so it can be tested and
reasoned about on its own. Tensor conversion happens at the call site.
"""

import io
import json
import logging
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Guard rails. A contributor should never be handed an unbounded download.
MAX_ARTIFACT_BYTES = 512 * 1024 * 1024   # 512 MB
MAX_ARRAY_COUNT = 512

# numpy kinds that are plain numbers. Anything else ('O' for object above all)
# can carry a pickle payload and is refused.
SAFE_KINDS = frozenset("biufc")  # bool, int, uint, float, complex


class ArtifactError(ValueError):
    """Raised when an artifact is malformed, unsafe, or too large."""


def _check_array(name: str, array: np.ndarray) -> None:
    if array.dtype.kind not in SAFE_KINDS:
        raise ArtifactError(
            f"Array {name!r} has dtype {array.dtype!r}; only plain numeric "
            f"arrays are allowed (object arrays can carry executable payloads)."
        )
    if array.dtype.hasobject:
        raise ArtifactError(f"Array {name!r} contains Python objects; refusing to load.")


def pack_arrays(arrays: Dict[str, Any]) -> bytes:
    """Serialise a mapping of name -> numeric array into an .npz archive."""
    if not arrays:
        raise ArtifactError("Refusing to pack an empty artifact.")
    if len(arrays) > MAX_ARRAY_COUNT:
        raise ArtifactError(f"Too many arrays: {len(arrays)} > {MAX_ARRAY_COUNT}")

    prepared = {}
    for name, value in arrays.items():
        if not isinstance(name, str):
            raise ArtifactError(f"Array names must be strings, got {type(name).__name__}")
        array = np.asarray(value)
        _check_array(name, array)
        prepared[name] = array

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **prepared)
    payload = buffer.getvalue()

    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ArtifactError(
            f"Artifact is {len(payload)} bytes, over the {MAX_ARTIFACT_BYTES} limit."
        )
    return payload


def unpack_arrays(payload: bytes) -> Dict[str, np.ndarray]:
    """Load an .npz archive without ever unpickling.

    Every array is checked before it is returned, so a hostile archive fails
    loudly instead of handing back something executable.
    """
    if not isinstance(payload, (bytes, bytearray, memoryview)):
        raise ArtifactError("Artifact payload must be bytes.")
    if len(payload) == 0:
        raise ArtifactError("Artifact payload is empty.")
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise ArtifactError(
            f"Artifact is {len(payload)} bytes, over the {MAX_ARTIFACT_BYTES} limit."
        )

    try:
        # allow_pickle=False is the entire point of this function.
        with np.load(io.BytesIO(bytes(payload)), allow_pickle=False) as archive:
            names = list(archive.files)
            if not names:
                raise ArtifactError("Artifact contains no arrays.")
            if len(names) > MAX_ARRAY_COUNT:
                raise ArtifactError(f"Too many arrays: {len(names)} > {MAX_ARRAY_COUNT}")

            out = {}
            for name in names:
                array = archive[name]
                _check_array(name, array)
                out[name] = array
            return out

    except ArtifactError:
        raise
    except ValueError as e:
        # numpy raises ValueError when it refuses to unpickle an object array.
        raise ArtifactError(f"Refused to load artifact: {e}") from e
    except Exception as e:
        raise ArtifactError(f"Artifact is not a readable .npz archive: {e}") from e


# --- datasets ------------------------------------------------------------

def pack_dataset(features: Any, labels: Any) -> bytes:
    """Serialise a supervised dataset (x, y) for transport to a node."""
    x = np.asarray(features)
    y = np.asarray(labels)

    if x.shape[0] != y.shape[0]:
        raise ArtifactError(
            f"Feature/label length mismatch: {x.shape[0]} vs {y.shape[0]}"
        )
    if x.shape[0] == 0:
        raise ArtifactError("Dataset is empty.")

    return pack_arrays({"x": x, "y": y})


def unpack_dataset(payload: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """Load a dataset artifact, returning (x, y)."""
    arrays = unpack_arrays(payload)

    missing = {"x", "y"} - set(arrays)
    if missing:
        raise ArtifactError(f"Dataset artifact is missing {sorted(missing)}")

    x, y = arrays["x"], arrays["y"]
    if x.shape[0] != y.shape[0]:
        raise ArtifactError(
            f"Feature/label length mismatch: {x.shape[0]} vs {y.shape[0]}"
        )
    if x.shape[0] == 0:
        raise ArtifactError("Dataset artifact is empty.")

    return x, y


# --- model weights -------------------------------------------------------

# A description of the model, carried inside the weights file itself.
#
# Without it the download is a bag of arrays named "0.weight", "2.weight",
# "4.bias" -- reusable only by someone who already knows the exact module list
# those indices refer to. The submitter has to reverse-engineer the very thing
# they asked the network to build for them, so the manifest travels with the
# weights rather than alongside them, where the two could be separated.
#
# It is stored as UTF-8 bytes in a uint8 array: the loader refuses anything but
# plain numeric arrays (no pickles), and that rule is worth more than the
# convenience of a string dtype.
MANIFEST_KEY = "__manifest__"
MAX_MANIFEST_BYTES = 64 * 1024


def pack_state_dict(state_dict: Dict[str, Any],
                    manifest: Optional[Dict[str, Any]] = None) -> bytes:
    """Serialise trained weights. Accepts torch tensors or numpy arrays.

    Returned to whoever submitted the job, so it must be loadable without
    trusting the contributor who produced it. `manifest` describes how to
    rebuild the model these weights belong to.
    """
    arrays = {}
    for name, value in state_dict.items():
        if name == MANIFEST_KEY:
            raise ArtifactError(f"{MANIFEST_KEY!r} is reserved for the model description.")
        if hasattr(value, "detach"):          # a torch tensor
            value = value.detach().cpu().numpy()
        arrays[name] = np.asarray(value)

    if manifest is not None:
        encoded = json.dumps(manifest, separators=(",", ":"), default=str).encode("utf-8")
        if len(encoded) > MAX_MANIFEST_BYTES:
            raise ArtifactError(
                f"Model description is {len(encoded)} bytes, over the "
                f"{MAX_MANIFEST_BYTES} limit."
            )
        arrays[MANIFEST_KEY] = np.frombuffer(encoded, dtype=np.uint8)

    return pack_arrays(arrays)


def unpack_state_dict(payload: bytes) -> Dict[str, np.ndarray]:
    """Load trained weights produced by someone else's machine.

    The manifest is not a weight, so it is not returned here; read it with
    read_manifest().
    """
    arrays = unpack_arrays(payload)
    arrays.pop(MANIFEST_KEY, None)
    return arrays


def read_manifest(payload: bytes) -> Optional[Dict[str, Any]]:
    """The model description packed with these weights, if there is one.

    Returns None rather than raising for older files that predate manifests,
    and for anything that does not decode: a description that cannot be read
    should not stop someone loading their own weights.
    """
    try:
        arrays = unpack_arrays(payload)
    except ArtifactError:
        raise

    raw = arrays.get(MANIFEST_KEY)
    if raw is None:
        return None

    try:
        decoded = bytes(np.asarray(raw, dtype=np.uint8)).decode("utf-8")
        manifest = json.loads(decoded)
    except Exception as e:
        logger.warning(f"Model description could not be read: {e}")
        return None

    return manifest if isinstance(manifest, dict) else None


# --- CSV intake ----------------------------------------------------------

MAX_CSV_ROWS = 500_000
MAX_CSV_COLUMNS = 4096


def _looks_numeric(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


class CsvDataset(NamedTuple):
    """A parsed spreadsheet, and what its numbers were called.

    The names matter more than they look. A model trained on six columns comes
    back knowing it takes six floats and nothing about which is which, so
    feeding them in a different order gives a confident wrong answer with no
    error -- and the header row that would have prevented that was read, used
    to decide "this is a header", and thrown away.
    """
    features: np.ndarray
    labels: np.ndarray
    class_names: Optional[List[str]]
    feature_names: Optional[List[str]]
    label_name: Optional[str]


def parse_csv_dataset(text: str) -> CsvDataset:
    """Turn a CSV into features, labels, and the names of both.

    Convention: every column is a feature except the LAST, which is the label.
    A non-numeric first row is treated as a header, kept, and reported. Labels
    may be numbers or names; names are mapped to 0..n-1 and returned so the
    caller can show which index means what.

    This is plain text parsing -- nothing here evaluates or imports anything, so
    an uploaded file cannot execute code.
    """
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ArtifactError(f"CSV must be UTF-8 text: {e}")

    rows: List[List[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        delimiter = ";" if (line.count(";") > line.count(",")) else ","
        rows.append([cell.strip() for cell in line.split(delimiter)])

    if not rows:
        raise ArtifactError("CSV contains no data rows.")

    width = len(rows[0])
    if width < 2:
        raise ArtifactError(
            "CSV needs at least two columns: one or more features, then the label."
        )
    if width > MAX_CSV_COLUMNS:
        raise ArtifactError(f"CSV has {width} columns, over the {MAX_CSV_COLUMNS} limit.")

    # A first row whose feature cells are not numeric is a header.
    feature_names: Optional[List[str]] = None
    label_name: Optional[str] = None

    if not all(_looks_numeric(cell) for cell in rows[0][:-1]):
        header = rows[0]
        feature_names = [name.strip() for name in header[:-1]]
        label_name = header[-1].strip()

        # A column with no name is worse than no names at all: it would let a
        # caller line rows up by name and be wrong about one of them.
        if not all(feature_names) or not label_name:
            feature_names = None
            label_name = None
        elif len(set(feature_names)) != len(feature_names):
            # Two columns with the same name cannot be told apart either.
            feature_names = None
            label_name = None

        rows = rows[1:]
        if not rows:
            raise ArtifactError("CSV contains a header but no data rows.")

    if len(rows) > MAX_CSV_ROWS:
        raise ArtifactError(f"CSV has {len(rows)} rows, over the {MAX_CSV_ROWS} limit.")

    features: List[List[float]] = []
    raw_labels: List[str] = []

    for number, row in enumerate(rows, start=1):
        if len(row) != width:
            raise ArtifactError(
                f"Row {number} has {len(row)} columns; expected {width}."
            )
        try:
            features.append([float(cell) for cell in row[:-1]])
        except ValueError:
            raise ArtifactError(
                f"Row {number} has a non-numeric feature value. "
                f"Only the last column may be text."
            )
        raw_labels.append(row[-1])

    class_names: Optional[List[str]] = None
    if all(_looks_numeric(v) for v in raw_labels):
        label_values = [float(v) for v in raw_labels]
        if not all(float(v).is_integer() for v in label_values):
            raise ArtifactError(
                "Labels must be whole numbers (class indices) or text names."
            )
        labels = np.asarray(label_values, dtype=np.int64)
        if labels.min() < 0:
            raise ArtifactError("Class indices cannot be negative.")
    else:
        class_names = sorted(set(raw_labels))
        lookup = {name: index for index, name in enumerate(class_names)}
        labels = np.asarray([lookup[v] for v in raw_labels], dtype=np.int64)

    x = np.asarray(features, dtype=np.float32)

    if x.shape[0] == 0:
        raise ArtifactError("CSV contains no usable rows.")
    if len(set(labels.tolist())) < 2:
        raise ArtifactError(
            "The label column has only one distinct value; there is nothing to learn."
        )

    return CsvDataset(x, labels, class_names, feature_names, label_name)


# --- scoring a spreadsheet -----------------------------------------------

MAX_SCORE_ROWS = 100_000


def read_rows_for_scoring(text: Any, feature_names: Optional[List[str]] = None
                          ) -> Tuple[np.ndarray, List[str], List[List[str]]]:
    """Read a CSV of rows to be predicted, in the order the model expects.

    Returns (features, header, original rows). The header and the rows are
    handed back so the answer can be written beside the data it came from
    rather than in place of it.

    The person who uploaded a spreadsheet is a spreadsheet person. They are
    not going to install PyTorch to use the thing they trained, so the model
    has to come to the data rather than the other way round.

    When the model knows what its columns were called, they are matched by
    name and extra columns are ignored -- so the very file that was trained
    on can be sent back, labels and all, and the answer appears next to the
    truth. Only when the model has no names is position trusted, because
    getting the order wrong is a confident wrong answer rather than an error.
    """
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise ArtifactError(f"CSV must be UTF-8 text: {e}")

    rows: List[List[str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        delimiter = ";" if (line.count(";") > line.count(",")) else ","
        rows.append([cell.strip() for cell in line.split(delimiter)])

    if not rows:
        raise ArtifactError("That file has no rows in it.")
    if len(rows) > MAX_SCORE_ROWS:
        raise ArtifactError(
            f"That file has {len(rows):,} rows; the limit is {MAX_SCORE_ROWS:,}."
        )

    header: List[str] = []
    if not all(_looks_numeric(cell) for cell in rows[0]):
        header = [name.strip() for name in rows[0]]
        rows = rows[1:]
        if not rows:
            raise ArtifactError("That file has a header but no rows under it.")

    wanted = list(feature_names or [])

    if wanted and header:
        missing = [name for name in wanted if name not in header]
        if missing:
            raise ArtifactError(
                f"This model needs {', '.join(wanted)}. "
                f"Your file is missing {', '.join(missing)}."
            )
        picked = [header.index(name) for name in wanted]
    else:
        # No names on one side or the other, so position is all there is.
        width = len(wanted) if wanted else len(rows[0])
        if len(rows[0]) < width:
            raise ArtifactError(
                f"This model needs {width} columns; the first row has "
                f"{len(rows[0])}."
            )
        picked = list(range(width))

    features: List[List[float]] = []
    for number, row in enumerate(rows, start=1):
        if max(picked) >= len(row):
            raise ArtifactError(
                f"Row {number} has {len(row)} columns; it needs at least "
                f"{max(picked) + 1}."
            )
        try:
            features.append([float(row[i]) for i in picked])
        except ValueError:
            raise ArtifactError(
                f"Row {number} has a value that is not a number in one of the "
                f"columns this model reads."
            )

    return np.asarray(features, dtype=np.float32), header, rows


def write_scored_csv(header: List[str], rows: List[List[str]],
                     predictions: List[str], confidence: List[float],
                     label_name: Optional[str] = None) -> str:
    """The rows that came in, with the model's answer added on the end."""
    answer = f"predicted_{label_name}" if label_name else "predicted"

    def quote(value: str) -> str:
        text = str(value)
        if any(c in text for c in ',"\n'):
            return '"' + text.replace('"', '""') + '"'
        return text

    lines = []
    if header:
        lines.append(",".join(quote(h) for h in header + [answer, "confidence"]))

    for row, prediction, score in zip(rows, predictions, confidence):
        lines.append(",".join(
            quote(v) for v in list(row) + [prediction, f"{score:.4f}"]))

    return "\n".join(lines) + "\n"


# --- growing a dataset ---------------------------------------------------

def merge_datasets(first: Tuple[Any, Any, Dict[str, Any]],
                   second: Tuple[Any, Any, Dict[str, Any]]
                   ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Combine two datasets into one, or say exactly why they do not fit.

    Data size is the strongest single predictor of a usable model on this
    service -- 41%, 62% and 77% held-back accuracy at 71 KB, 310 KB and 25 MB
    of text. Someone whose corpus is too small has one useful move, which is to
    find more of it, and until now there was no way to hand that over except by
    concatenating files by hand before uploading.

    Merged as arrays rather than as raw text on purpose. For a language model
    the rows are independent windows, so joining the row sets is exactly right;
    joining the source text instead would splice the end of one file onto the
    start of the next and teach the model a transition that never happened.
    """
    x1, y1, info1 = first
    x2, y2, info2 = second

    kind1 = (info1 or {}).get("format")
    kind2 = (info2 or {}).get("format")
    if kind1 and kind2 and kind1 != kind2:
        raise ArtifactError(
            f"This dataset is {kind1} and the file you added is {kind2}. "
            f"A dataset can only grow with more of the same kind."
        )

    x1, y1 = np.asarray(x1), np.asarray(y1)
    x2, y2 = np.asarray(x2), np.asarray(y2)

    if x1.ndim != x2.ndim or x1.shape[1:] != x2.shape[1:]:
        raise ArtifactError(
            f"The file you added has rows of {x2.shape[1:]} where this dataset "
            f"has {x1.shape[1:]}. They have to match to be trained together."
        )
    if y1.shape[1:] != y2.shape[1:]:
        raise ArtifactError("The two label sets are different shapes.")

    info = dict(info1 or {})

    # Class names are the hard part. Two spreadsheets sorted their own labels
    # independently, so index 0 means something different in each -- appending
    # the numbers as they stand would silently relabel half the data.
    names1 = (info1 or {}).get("class_names")
    names2 = (info2 or {}).get("class_names")

    if kind1 == "csv" and bool(names1) != bool(names2):
        raise ArtifactError(
            "One of these files has named labels and the other has numbers. "
            "Use the same kind of label in both."
        )

    if names1 and names2:
        merged_names = list(names1)
        for name in names2:
            if name not in merged_names:
                merged_names.append(name)

        # Only the added file is remapped; the existing rows keep the indices
        # they already have, so anything already trained against them still
        # means what it meant.
        lookup = {name: merged_names.index(name) for name in names2}
        remap = np.array([lookup[name] for name in names2], dtype=np.int64)
        y2 = remap[np.asarray(y2, dtype=np.int64)]
        info["class_names"] = merged_names

    x = np.concatenate([x1, x2.astype(x1.dtype, copy=False)], axis=0)
    y = np.concatenate([y1, y2.astype(y1.dtype, copy=False)], axis=0)

    info["rows"] = int(x.shape[0])
    for key in ("tokens", "source_bytes"):
        if key in info and key in (info2 or {}):
            info[key] = int(info[key]) + int(info2[key])
    info["parts"] = int(info.get("parts", 1)) + 1

    return x, y, info


# --- text intake ---------------------------------------------------------

# Bytes, not words. A word or sub-word tokeniser needs a vocabulary file built
# from a corpus, and that file then has to travel with the model for anything
# it produces to be readable again. Bytes need nothing: every file in every
# language already is a sequence of them, the vocabulary is fixed at 256, and
# decoding is `bytes(ids)`. The model has to learn spelling as well as
# language, which costs it capacity -- but a small model that works on any
# .txt beats a larger one that only works next to the vocabulary it was
# built with.
TEXT_VOCAB_SIZE = 256

# What a text corpus actually costs, measured rather than assumed.
#
# The packed archive is not the constraint: byte ids compress to a fraction of
# the source (0.03x on repetitive text, well under 1x on prose). The constraint
# is memory on the contributor's machine, which is somebody's home PC.
#
# It used to be 16x the source file there -- int32 arrays for x and y, cast to
# int64 in one go before the first batch. Both were waste: a byte id fits in a
# uint8, and the cast belongs on the batch rather than the corpus. That is now
# 2x, so the same memory holds eight times the text.
#
#     16 MB source, before:  256 MB on the node
#     16 MB source, after:    32 MB on the node
MAX_TEXT_BYTES = 128 * 1024 * 1024

MIN_SEQ_LEN, MAX_SEQ_LEN = 8, 2048

# Fewer than this and there is nothing to hold back for verification, let
# alone learn from.
MIN_TEXT_WINDOWS = 16

# How much text is enough, measured rather than guessed. Two runs of this
# service on the same hardware, same architecture family:
#
#     71 KB  -> 41% next-byte accuracy on held-back text, output is
#               word-shaped nonsense: it learned spelling and nothing else
#    310 KB  -> 62% accuracy, output is syntactically plausible lines
#
# Both passed verification. Both were real training. Only one produced
# something a person would keep, and the difference was the size of the file.
# Saying so at upload costs nothing; finding out afterwards costs a
# contributor's GPU time and the submitter's afternoon.
TEXT_THIN_BYTES = 256 * 1024
TEXT_COMFORTABLE_BYTES = 1024 * 1024


def text_size_advice(source_bytes: int) -> Optional[str]:
    """Plain warning about a text corpus that is too small to learn from.

    Returns None when there is nothing useful to say -- silence is the right
    output for a file that is big enough.
    """
    if source_bytes >= TEXT_COMFORTABLE_BYTES:
        return None

    def size(value):
        # "1024 KB" is a true answer to a question nobody asked.
        if value >= 1024 * 1024:
            return f"{value / (1024 * 1024):.0f} MB"
        return f"{value // 1024} KB"

    if source_bytes < TEXT_THIN_BYTES:
        return (
            f"{size(source_bytes)} is a small corpus. This will train, and it "
            f"will pass verification, but expect output that looks like words "
            f"without meaning any. Around {size(TEXT_COMFORTABLE_BYTES)} is "
            f"where sentences start to hold together."
        )

    return (
        f"{size(source_bytes)} is on the thin side. Expect recognisable "
        f"phrasing but loose meaning; around {size(TEXT_COMFORTABLE_BYTES)} "
        f"reads better."
    )


def parse_text_dataset(text: Any, seq_len: int = 64) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Turn a text file into (x, y) windows for a causal language model.

    A language model is trained to predict the next token, so the labels are
    the input shifted by one. Cutting the stream into fixed windows here rather
    than on the node keeps the dataset the same shape as every other one: rows
    that can be shuffled and split, so the holdout that verifies the returned
    model works unchanged.

    Windows do not overlap. Overlapping them multiplies the data by the stride
    for no new information, and -- worse -- puts the same text in both the
    training half and the holdout, which would quietly turn verification into a
    memory test.

    This is byte counting, not parsing: nothing here evaluates or imports
    anything, so an uploaded file cannot execute code.
    """
    seq_len = int(seq_len)
    if seq_len < MIN_SEQ_LEN or seq_len > MAX_SEQ_LEN:
        raise ArtifactError(
            f"Sequence length must be between {MIN_SEQ_LEN} and {MAX_SEQ_LEN}; got {seq_len}."
        )

    raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)

    # A UTF-8 BOM is not part of the text and would be learned as if it were.
    if raw[:3] == b'\xef\xbb\xbf':
        raw = raw[3:]

    if len(raw) > MAX_TEXT_BYTES:
        raise ArtifactError(
            f"Text is {len(raw):,} bytes, over the {MAX_TEXT_BYTES:,} limit."
        )

    data = np.frombuffer(raw, dtype=np.uint8)

    # Each row needs one byte of lookahead for its last target.
    rows = (len(data) - 1) // seq_len if len(data) else 0
    if rows < MIN_TEXT_WINDOWS:
        needed = (MIN_TEXT_WINDOWS * seq_len) + 1
        raise ArtifactError(
            f"Text is too short: {len(raw):,} bytes makes {rows} training "
            f"sequence(s) of {seq_len}. At least {needed:,} bytes are needed."
        )

    span = rows * seq_len
    # uint8, because that is what a byte is. Every widening from here -- to the
    # int64 an embedding lookup needs -- happens per batch, on the few hundred
    # rows being trained on, rather than over the whole corpus at rest.
    x = data[:span].reshape(rows, seq_len).copy()
    y = data[1:span + 1].reshape(rows, seq_len).copy()

    info = {
        "tokenizer": "bytes",
        "vocab_size": TEXT_VOCAB_SIZE,
        "seq_len": seq_len,
        "rows": int(rows),
        "tokens": int(span),
        "source_bytes": len(raw),
    }
    return x, y, info

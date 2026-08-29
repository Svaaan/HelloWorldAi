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
from typing import Any, Dict, List, Optional, Tuple

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


def parse_csv_dataset(text: str) -> Tuple[np.ndarray, np.ndarray, Optional[List[str]]]:
    """Turn a CSV into (features, labels, class_names).

    Convention: every column is a feature except the LAST, which is the label.
    A non-numeric first row is treated as a header and skipped. Labels may be
    numbers or names; names are mapped to 0..n-1 and returned so the caller can
    show which index means what.

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
    if not all(_looks_numeric(cell) for cell in rows[0][:-1]):
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

    return x, labels, class_names


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

# Text expands: every byte becomes an int32 in both x and y, so the packed
# archive is far larger than the file uploaded. Compression claws most of it
# back for natural language, but the limit is on the input where the person
# uploading can see it.
MAX_TEXT_BYTES = 16 * 1024 * 1024

MIN_SEQ_LEN, MAX_SEQ_LEN = 8, 2048

# Fewer than this and there is nothing to hold back for verification, let
# alone learn from.
MIN_TEXT_WINDOWS = 16


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
    # int32 rather than int64: the ids only go up to 255, and this halves what
    # crosses the network. The trainer casts to long when it builds the batch.
    x = data[:span].reshape(rows, seq_len).astype(np.int32)
    y = data[1:span + 1].reshape(rows, seq_len).astype(np.int32)

    info = {
        "tokenizer": "bytes",
        "vocab_size": TEXT_VOCAB_SIZE,
        "seq_len": seq_len,
        "rows": int(rows),
        "tokens": int(span),
        "source_bytes": len(raw),
    }
    return x, y, info

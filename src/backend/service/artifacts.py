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

def pack_state_dict(state_dict: Dict[str, Any]) -> bytes:
    """Serialise trained weights. Accepts torch tensors or numpy arrays.

    Returned to whoever submitted the job, so it must be loadable without
    trusting the contributor who produced it.
    """
    arrays = {}
    for name, value in state_dict.items():
        if hasattr(value, "detach"):          # a torch tensor
            value = value.detach().cpu().numpy()
        arrays[name] = np.asarray(value)
    return pack_arrays(arrays)


def unpack_state_dict(payload: bytes) -> Dict[str, np.ndarray]:
    """Load trained weights produced by someone else's machine."""
    return unpack_arrays(payload)


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

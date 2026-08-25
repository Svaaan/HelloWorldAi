"""Resolve an NVML-reported GPU name to an entry in the scraped GPU database.

NVML and the database spell the same part differently — NVML reports
"NVIDIA GeForce RTX 4090 Laptop GPU" where the database says
"GeForce RTX 4090 Mobile" — so names are normalised to token lists before being
compared, and comparisons happen on whole-token boundaries.

Matching is deliberately conservative. A plain substring test makes
"RTX A4000" match the entry "RTX A400" and "RTX A2000" match "A2", and it lets
a laptop part inherit the specs of its desktop namesake. When a name is
ambiguous we return None and the caller reports 0 TFLOPS: for a network that
schedules on advertised compute, under-reporting a node is recoverable and
over-reporting it is not.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_DB_PATH = os.path.join(BASE_DIR, 'gpu-db.json')


def load_gpu_db(path: str = GPU_DB_PATH) -> List[Dict[str, Any]]:
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load GPU database: {e}")
        return []


gpu_db = load_gpu_db()


def normalize(name: str) -> List[str]:
    """Reduce a GPU name to comparable lowercase tokens."""
    n = (name or "").lower()
    n = n.replace("laptop gpu", " mobile ")
    n = n.replace("laptop", " mobile ")
    n = n.replace("nvidia", " ")
    n = re.sub(r"[^a-z0-9]+", " ", n)
    return n.split()


def _is_mobile(tokens: List[str]) -> bool:
    return "mobile" in tokens


def _is_token_prefix(shorter: List[str], longer: List[str]) -> bool:
    """True if `shorter` is a whole-token prefix of `longer`.

    Token boundaries are what stop "rtx a400" from matching "rtx a4000".
    """
    return len(shorter) < len(longer) and longer[:len(shorter)] == shorter


# Precomputed so lookups do not re-tokenise the whole database each call.
_INDEX: List[Tuple[List[str], Dict[str, Any]]] = [
    (normalize(entry.get('name', '')), entry) for entry in gpu_db
]


def find_gpu_entry(gpu_name: str) -> Optional[Dict[str, Any]]:
    """Return the database entry for a GPU, or None if there is no safe match."""
    target = normalize(gpu_name)
    if not target:
        return None

    # 1. Exact match on normalised tokens.
    for tokens, entry in _INDEX:
        if tokens == target:
            return entry

    # 2. Whole-token prefix in either direction. A mobile part may only ever
    #    match another mobile part, and vice versa.
    target_mobile = _is_mobile(target)
    candidates = [
        (tokens, entry) for tokens, entry in _INDEX
        if _is_mobile(tokens) == target_mobile
        and (_is_token_prefix(tokens, target) or _is_token_prefix(target, tokens))
    ]

    if not candidates:
        logger.warning(f"No GPU database entry matched '{gpu_name}'.")
        return None

    shader_counts = {entry.get('shaders') for _, entry in candidates}
    if len(shader_counts) > 1:
        logger.warning(
            f"'{gpu_name}' matched {len(candidates)} entries with differing shader "
            f"counts {sorted(c for c in shader_counts if c)} — refusing to guess."
        )
        return None

    # Same shader count across all candidates; prefer the closest-length name so
    # the clock we read alongside it is the most specific one available.
    candidates.sort(key=lambda pair: abs(len(pair[0]) - len(target)))
    return candidates[0][1]


def get_cuda_cores(gpu_name: str) -> Optional[int]:
    """Shader/CUDA core count for a GPU, or None when it cannot be resolved."""
    entry = find_gpu_entry(gpu_name)
    return entry.get('shaders') if entry else None


def get_database_clock_mhz(entry: Optional[Dict[str, Any]]) -> Optional[int]:
    """Reference clock from the database, used when NVML reports an idle clock."""
    if not entry:
        return None
    raw = str(entry.get('gpu_clock', '')).replace('MHz', '').strip()
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        logger.warning(
            f"Could not parse gpu_clock {entry.get('gpu_clock')!r} for {entry.get('name')!r}."
        )
        return None

"""What a finished model can do: sample, download, score a CSV.

Split out of coordinator.py. The routes are registered on a router here and
included by coordinator.py, so the URLs and behaviour are unchanged.
"""

import asyncio
import logging
import os
import re
import secrets
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pynvml
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request
from fastapi.responses import Response
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorCollection,
    AsyncIOMotorDatabase,
    AsyncIOMotorGridFSBucket,
)
from pydantic import BaseModel
from pymongo import ReturnDocument

from backend.database.nodedb import db
from backend.service import artifactCrypto
from backend.service.artifacts import MAX_ARTIFACT_BYTES
from backend.service.authNodeService import verify_signature
from backend.service.jobSpec import (
    ARCHITECTURES, JobSpecError, advise, job_schema, next_run_name, validate_job,
)
from backend.service.nodePicker import (
    BUSY_STATUSES, NoNodeAvailable, pick_node, summarise_choice,
)
from backend.service.submitterService import read_submitter_key
from backend.service.tokenService import (
    NODE_TOKEN_TTL, issue_node_token, read_node_token,
)

logger = logging.getLogger("coordinator")

from backend.routes.deps import (
    DATASET_RETENTION_MINUTES, Database, HOLDOUT_FRACTION, MAX_TASK_ATTEMPTS,
    MONGODB_URL, NodeConnection, CPUCapabilities, GPUCapabilities,
    TASK_CLAIM_TIMEOUT_MINUTES, authenticated_node, connected_nodes, get_db,
    node_challenges, optional_node, optional_submitter, require_node_token,
    require_uploader, system_usage, task_results,
)
from backend.routes.artifacts import _read_artifact
from backend.routes.tasks import _owned_task

router = APIRouter()


# What a submitter may ask a finished model to write, and how much of it.
# This is a forward pass per token on the coordinator's CPU, so it is bounded
# rather than trusted: a small model is cheap, and cheap times unbounded is not.
MAX_PROMPT_CHARS = 500

MAX_SAMPLE_TOKENS = 400

DEFAULT_SAMPLE_TOKENS = 200

@router.post("/my-tasks/{task_id}/sample")
async def sample_from_model(
    task_id: str,
    body: dict = Body(None),
    db: Database = Depends(get_db),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """Ask a finished language model to continue a prompt you type.

    A model came back as a number, a grade, and three continuations of prompts
    the node picked. Finding out what it does with a prompt of your own meant
    downloading the weights, installing torch and running a script -- for a
    forward pass that takes a fraction of a second on the machine already
    holding the file.
    """
    task = await _owned_task(db, task_id, submitter)

    weights_id = task.get("weights_id")
    if not weights_id:
        raise HTTPException(status_code=409, detail="This job has no model to ask.")

    spec = (task.get("task_data") or {}).get("model_spec") or {}
    architecture = str(spec.get("architecture", "mlp")).lower()
    if architecture in ("mlp", "feedforward"):
        raise HTTPException(
            status_code=400,
            detail="This model classifies rows of numbers; it does not write text.",
        )

    prompt = str((body or {}).get("prompt") or "")
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="Type something for it to continue.")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"The prompt is longer than {MAX_PROMPT_CHARS} characters.",
        )

    try:
        length = int((body or {}).get("length") or DEFAULT_SAMPLE_TOKENS)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="length must be a whole number.")
    length = max(1, min(length, MAX_SAMPLE_TOKENS))

    try:
        temperature = float((body or {}).get("temperature", 0.8))
    except (TypeError, ValueError):
        temperature = 0.8
    temperature = max(0.0, min(temperature, 2.0))

    from backend.service.artifacts import ArtifactError, read_manifest, unpack_state_dict
    from backend.service.trainer import (
        build_workload, continue_tokens, render_bytes,
    )

    try:
        payload = await _read_artifact(db, str(weights_id))
        state_dict = unpack_state_dict(payload)
        manifest = read_manifest(payload) or {}
    except Exception as e:
        logger.error(f"Could not read weights {weights_id}: {e}")
        raise HTTPException(status_code=400, detail="Could not read the model file.")

    # The manifest travels inside the weights and is the authority on what was
    # actually built -- the task's spec is what was asked for.
    resolved = manifest.get("spec") or spec
    tokenizer = (manifest.get("tokenizer") or {}).get("kind")
    if tokenizer and tokenizer != "bytes":
        raise HTTPException(
            status_code=400,
            detail=f"This model's tokeniser is {tokenizer!r}, which this "
                   f"service cannot encode for.",
        )

    def run():
        import torch

        model = build_workload(resolved)["factory"]()
        model.load_state_dict(
            {name: torch.as_tensor(np.asarray(value))
             for name, value in state_dict.items()},
            strict=True,
        )
        model.eval()

        ids = list(prompt.encode("utf-8"))
        grown = continue_tokens(model, resolved, ids,
                                length=length, temperature=temperature)
        return render_bytes(grown[len(ids):])

    try:
        # Off the event loop: this is a few hundred forward passes, and the
        # coordinator still has heartbeats to answer while it runs.
        continuation = await asyncio.to_thread(run)
    except ArtifactError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Could not sample task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="The model could not be run.")

    return {
        "status": "success",
        "prompt": prompt,
        "continuation": continuation,
        "tokens": length,
    }

# The standalone loader, shipped inside the bundle so the download runs
# without hunting for anything. Read from disk once rather than on every
# request, and tolerated as missing: a bundle without it is still a model.
LOADER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "frontend", "static", "scripts", "load_model.py",
)

def _loader_source() -> Optional[str]:
    try:
        with open(LOADER_PATH, encoding="utf-8") as handle:
            return handle.read()
    except OSError as e:
        logger.warning(f"Could not read {LOADER_PATH} for the bundle: {e}")
        return None

@router.get("/my-tasks/{task_id}/bundle")
async def download_bundle(
    task_id: str,
    db: Database = Depends(get_db),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """The finished model as a folder, the way models are normally shipped.

    Weights in safetensors, a config.json describing them, a tokenizer when
    there is one, and the loader. The `.npz` remains available for anyone who
    wants the raw arrays, but it is this project's own arrangement and nothing
    else reads it.
    """
    task = await _owned_task(db, task_id, submitter)

    weights_id = task.get("weights_id")
    if not weights_id:
        raise HTTPException(status_code=409, detail="This job produced no model.")

    from backend.service.artifacts import read_manifest, unpack_state_dict
    from backend.service.modelBundle import build_bundle

    try:
        payload = await _read_artifact(db, str(weights_id))
        state_dict = unpack_state_dict(payload)
        manifest = read_manifest(payload) or {}
    except Exception as e:
        logger.error(f"Could not read weights {weights_id}: {e}")
        raise HTTPException(status_code=400, detail="Could not read the model file.")

    manifest.setdefault("model_name",
                        (task.get("task_data") or {}).get("model_name") or "model")

    try:
        archive = await asyncio.to_thread(
            build_bundle, state_dict, manifest, _loader_source()
        )
    except Exception as e:
        logger.error(f"Could not build a bundle for {task_id}: {e}")
        raise HTTPException(status_code=500, detail="Could not package the model.")

    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", str(manifest["model_name"])) or "model"
    logger.info(f"Packaged {task_id} as {safe}.zip ({len(archive)} bytes)")

    return Response(
        content=archive,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{safe}.zip"'},
    )

# A spreadsheet sent for scoring is the submitter's data, exactly like the
# training set was. It is read, answered and dropped: never written to the
# database, never kept after the response.
MAX_SCORE_BYTES = 32 * 1024 * 1024

SCORE_BATCH_ROWS = 1024

@router.post("/my-tasks/{task_id}/predict")
async def predict_from_csv(
    task_id: str,
    request: Request,
    db: Database = Depends(get_db),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """Run a finished classifier over a CSV and hand back the same rows, answered.

    The person who uploads a spreadsheet is a spreadsheet person. Handing them
    a weights file -- in any format -- hands them something they cannot open,
    and "export it to ONNX" needs the Python and PyTorch they do not have.
    Using the model has to be possible without leaving the page.
    """
    task = await _owned_task(db, task_id, submitter)

    weights_id = task.get("weights_id")
    if not weights_id:
        raise HTTPException(status_code=409, detail="This job has no model to run.")

    spec = (task.get("task_data") or {}).get("model_spec") or {}
    if str(spec.get("architecture", "mlp")).lower() not in ("mlp", "feedforward"):
        raise HTTPException(
            status_code=400,
            detail="This model writes text rather than sorting rows. "
                   "Ask it to continue something instead.",
        )

    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="That file is empty.")
    if len(payload) > MAX_SCORE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"That file is {len(payload):,} bytes; the limit is "
                   f"{MAX_SCORE_BYTES:,}.",
        )

    from backend.service.artifacts import (
        ArtifactError, read_manifest, read_rows_for_scoring, unpack_state_dict,
        write_scored_csv,
    )
    from backend.service.trainer import build_workload

    try:
        weights = await _read_artifact(db, str(weights_id))
        state_dict = unpack_state_dict(weights)
        manifest = read_manifest(weights) or {}
    except Exception as e:
        logger.error(f"Could not read weights {weights_id}: {e}")
        raise HTTPException(status_code=400, detail="Could not read the model file.")

    resolved = manifest.get("spec") or spec
    class_names = manifest.get("class_names") or []
    feature_names = (manifest.get("input") or {}).get("names")

    try:
        features, header, rows = read_rows_for_scoring(payload, feature_names)
    except ArtifactError as e:
        raise HTTPException(status_code=400, detail=str(e))

    expected = int(resolved.get("input_dim", features.shape[1]))
    if features.shape[1] != expected:
        raise HTTPException(
            status_code=400,
            detail=f"This model reads {expected} columns; that file gave "
                   f"{features.shape[1]}.",
        )

    def run():
        import torch

        model = build_workload(resolved)["factory"]()
        model.load_state_dict(
            {name: torch.as_tensor(np.asarray(value))
             for name, value in state_dict.items()},
            strict=True,
        )
        model.eval()

        chosen, scores = [], []
        with torch.no_grad():
            for start in range(0, features.shape[0], SCORE_BATCH_ROWS):
                chunk = torch.as_tensor(features[start:start + SCORE_BATCH_ROWS])
                probabilities = torch.softmax(model(chunk), dim=-1)
                best = probabilities.argmax(dim=-1)
                chosen.extend(int(i) for i in best)
                scores.extend(
                    float(p[i]) for p, i in zip(probabilities, best))
        return chosen, scores

    try:
        # A forward pass over up to a hundred thousand rows is real work, and
        # the coordinator still has heartbeats to answer while it runs.
        indices, scores = await asyncio.to_thread(run)
    except Exception as e:
        logger.error(f"Could not score for task {task_id}: {e}")
        raise HTTPException(status_code=500, detail="The model could not be run.")

    named = [class_names[i] if i < len(class_names) else str(i) for i in indices]
    body = write_scored_csv(header, rows, named, scores,
                            manifest.get("label_name"))

    name = str((task.get("task_data") or {}).get("model_name") or "model")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name) or "model"

    logger.info(f"Scored {len(rows)} rows with task {task_id}.")

    return Response(
        content=body,
        media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="{safe}-predictions.csv"'},
    )

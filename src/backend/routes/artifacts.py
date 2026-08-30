"""Storing and fetching datasets, holdouts and weights.

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
from backend.service import quota
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
    require_uploader, system_usage, task_results, FINISHED_STATES,
)


router = APIRouter()


async def _forget_dataset(db, task: dict) -> int:
    """Delete the dataset copies a finished job no longer needs.

    Weights are never touched here -- neither the ones a job produced nor the
    ones it started from, which belong to the run before it and are still that
    run's result.

    Submitted data used to live in the database for ever: the training split,
    the holdout, and the original upload that prepare_dataset_split replaced.
    Keeping someone's data after the job it was for has finished is a liability
    with no purpose, so a completed task drops them.

    The trained weights are kept -- that is the thing the submitter came for.
    """
    bucket = AsyncIOMotorGridFSBucket(db.db, bucket_name="artifacts")
    removed = 0

    for key in ("dataset_id", "holdout_artifact_id"):
        artifact_id = task.get(key)
        if not artifact_id:
            continue
        try:
            await bucket.delete(ObjectId(artifact_id))
            removed += 1
        except Exception as e:
            logger.debug(f"Could not delete {key} {artifact_id}: {e}")

    if removed:
        await db.tasks_collection.update_one(
            {"_id": task["_id"]},
            {"$set": {"dataset_forgotten_at": datetime.utcnow()},
             "$unset": {"dataset_id": "", "holdout_artifact_id": ""}},
        )
        logger.info(f"Deleted {removed} dataset artifact(s) for finished task {task['_id']}.")

    return removed

async def _forget_orphaned_datasets(db, older_than: datetime) -> int:
    """Delete uploaded datasets that no task ever referenced.

    prepare_dataset_split writes a training half and a holdout and the task
    points at those, leaving the original upload referenced by nothing. A
    dataset that was uploaded and then abandoned -- the submitter changed their
    mind, or the job was refused -- was in the same position. Either way it sat
    in storage for ever with nothing pointing at it and nobody to delete it.
    """
    bucket = AsyncIOMotorGridFSBucket(db.db, bucket_name="artifacts")

    # Every artifact id any task still depends on.
    referenced = set()
    async for task in db.tasks_collection.find(
        {}, {"dataset_id": 1, "holdout_artifact_id": 1, "weights_id": 1}
    ):
        for key in ("dataset_id", "holdout_artifact_id", "weights_id"):
            if task.get(key):
                referenced.add(str(task[key]))

    removed = 0
    async for stored in db.db["artifacts.files"].find(
        {"metadata.kind": {"$in": ["dataset", "holdout"]},
         "metadata.uploaded_at": {"$lt": older_than}}
    ):
        if str(stored["_id"]) in referenced:
            continue
        try:
            await bucket.delete(stored["_id"])
            removed += 1
        except Exception as e:
            logger.debug(f"Could not delete orphaned artifact {stored['_id']}: {e}")

    if removed:
        logger.info(f"Deleted {removed} dataset artifact(s) no task referenced.")

    return removed

async def forget_finished_datasets():
    """Drop the data behind jobs that have finished and been verified.

    Runs on a delay rather than the instant a job completes: verification reads
    the holdout after the result lands, and a retry reuses the same split.
    """
    grace = timedelta(minutes=DATASET_RETENTION_MINUTES)

    while True:
        try:
            cutoff = datetime.utcnow() - grace
            finished = await Database.tasks_collection.find({
                "status": {"$in": list(FINISHED_STATES)},
                "finished_at": {"$lt": cutoff},
                "dataset_forgotten_at": {"$exists": False},
                "$or": [{"dataset_id": {"$ne": None}},
                        {"holdout_artifact_id": {"$ne": None}}],
            }).to_list(length=100)

            for task in finished:
                await _forget_dataset(Database, task)

            # Uploads that never became a job, and the pre-split originals the
            # split replaced, are nobody's data to keep.
            await _forget_orphaned_datasets(Database, cutoff)

        except Exception as e:
            logger.error(f"Error clearing finished datasets: {e}")

        await asyncio.sleep(300)

def _object_id(artifact_id: str) -> ObjectId:
    """An artifact id, or a 400 rather than a 500 on a malformed one."""
    try:
        return ObjectId(artifact_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed artifact id.")

def _convert_upload(payload: bytes, fmt: str, seq_len: int):
    """Turn an uploaded CSV or text file into (x, y, info).

    Shared by the upload and the append, so a file added to a dataset is read
    exactly the way the first one was.
    """
    from backend.service.artifacts import (
        ArtifactError, parse_csv_dataset, parse_text_dataset,
    )

    try:
        if fmt == "csv":
            features, labels, class_names = parse_csv_dataset(payload)
            info = {"format": "csv", "rows": int(features.shape[0])}
            if class_names:
                info["class_names"] = class_names
            return features, labels, info

        if fmt == "text":
            features, labels, info = parse_text_dataset(payload, seq_len=seq_len)
            info["format"] = "text"
            return features, labels, info

    except ArtifactError as e:
        raise HTTPException(status_code=400, detail=str(e))

    raise HTTPException(
        status_code=400,
        detail=f"Unknown upload format {fmt!r}. Supported: csv, text, "
               f"or omit it to upload a packed .npz.",
    )

def _describe_dataset(features, labels, info: dict) -> dict:
    """The numbers worth showing about a dataset, for whichever page asked."""
    from backend.service.artifacts import text_size_advice

    if info.get("format") == "text":
        summary = {
            "rows": int(features.shape[0]),
            "seq_len": info.get("seq_len"),
            "tokens": info.get("tokens"),
            "vocab_size": info.get("vocab_size"),
            "tokenizer": info.get("tokenizer"),
        }
        advice = text_size_advice(int(info.get("source_bytes") or 0))
        if advice:
            summary["advice"] = advice
        return summary

    return {
        "rows": int(features.shape[0]),
        "features": int(features.shape[1]) if features.ndim > 1 else 1,
        "classes": len(set(np.asarray(labels).reshape(-1).tolist())),
        "class_names": info.get("class_names"),
    }

@router.post("/artifacts/{artifact_id}/append")
async def append_to_artifact(artifact_id: str, request: Request,
                             db: Database = Depends(get_db),
                             uploader: str = Depends(require_uploader)):
    """Add another file to a dataset, returning a new, larger one.

    More data is the strongest thing a submitter can do for their result, and
    the only way to do it was to concatenate files by hand before uploading.

    A new artifact rather than a change to the old one: a queued job may
    already point at it, and a dataset that changes under a job that is
    training on it is not a dataset.
    """
    from backend.service.artifacts import (
        ArtifactError, merge_datasets, pack_dataset, unpack_dataset,
    )

    payload = await request.body()
    if not payload:
        raise HTTPException(status_code=400, detail="Artifact body is empty.")
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Artifact is {len(payload)} bytes; the limit is {MAX_ARTIFACT_BYTES}.",
        )

    # One file may be under the size cap and the hundredth still be a problem.
    # 429 rather than 413: the request is not too large, it is too soon.
    try:
        await quota.check_upload(db, uploader, len(payload))
    except quota.QuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    stored = await db.db["artifacts.files"].find_one(
        {"_id": _object_id(artifact_id)}, {"metadata": 1}
    )
    if not stored:
        raise HTTPException(status_code=404, detail="No such dataset.")
    if (stored.get("metadata") or {}).get("kind") != "dataset":
        raise HTTPException(status_code=400, detail="That artifact is not a dataset.")

    existing_info = (stored.get("metadata") or {}).get("info") or {}
    fmt = existing_info.get("format")
    if not fmt:
        raise HTTPException(
            status_code=400,
            detail="This dataset was uploaded as a packed .npz, so the "
                   "coordinator cannot tell what to add to it. Combine the "
                   "arrays yourself and upload the result.",
        )

    try:
        existing_x, existing_y = unpack_dataset(await _read_artifact(db, artifact_id))
    except ArtifactError as e:
        raise HTTPException(status_code=400, detail=f"Could not read that dataset: {e}")

    # A caller that knows what it picked says so, and a mismatch is refused
    # here. Without this, adding a spreadsheet to a text dataset read the CSV
    # as raw bytes -- which succeeds, and trains the model on commas.
    claimed = (request.query_params.get("format") or "").lower()
    if claimed and claimed != fmt:
        wanted = {"csv": "a CSV", "text": "a text file"}
        raise HTTPException(
            status_code=400,
            detail=(
                f"This dataset was built from {wanted.get(fmt, fmt)} and you "
                f"added {wanted.get(claimed, claimed)}. A dataset can only "
                f"grow with more of the same kind."
            ),
        )

    # Read the new file exactly as the first one was: same format, and for text
    # the same window length, or the rows would not line up.
    added_x, added_y, added_info = _convert_upload(
        payload, fmt, int(existing_info.get("seq_len") or 64)
    )

    try:
        features, labels, info = merge_datasets(
            (existing_x, existing_y, existing_info),
            (added_x, added_y, added_info),
        )
    except ArtifactError as e:
        raise HTTPException(status_code=400, detail=str(e))

    combined = pack_dataset(features, labels)
    new_id = await _write_artifact(db, combined, "dataset", info,
                                   uploader=uploader)

    logger.info(
        f"Appended to dataset {artifact_id}: "
        f"{existing_x.shape[0]} + {added_x.shape[0]} = {features.shape[0]} rows "
        f"-> {new_id}"
    )

    return {
        "status": "success",
        "artifact_id": str(new_id),
        "bytes": len(combined),
        "added_rows": int(added_x.shape[0]),
        "parts": info.get("parts"),
        **_describe_dataset(features, labels, info),
    }

@router.post("/artifacts")
async def upload_artifact(
    request: Request,
    db: Database = Depends(get_db),
    uploader: str = Depends(require_uploader),
):
    """Store a blob (a dataset, or trained weights) and return its id.

    The body is raw bytes rather than multipart so no extra dependency is
    needed. The coordinator never deserialises the contents -- it only moves
    them -- so a hostile payload cannot execute anything here. The node and the
    submitter both parse with artifacts.unpack_*, which refuses pickles.
    """
    payload = await request.body()

    if not payload:
        raise HTTPException(status_code=400, detail="Artifact body is empty.")
    if len(payload) > MAX_ARTIFACT_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Artifact is {len(payload)} bytes; the limit is {MAX_ARTIFACT_BYTES}.",
        )

    # One file may be under the size cap and the hundredth still be a problem.
    # 429 rather than 413: the request is not too large, it is too soon.
    try:
        await quota.check_upload(db, uploader, len(payload))
    except quota.QuotaExceeded as e:
        raise HTTPException(status_code=429, detail=str(e))

    kind = request.query_params.get("kind", "dataset")
    if kind not in ("dataset", "weights"):
        raise HTTPException(status_code=400, detail=f"Unknown artifact kind: {kind}")

    summary = {}
    info = {}

    # A browser cannot easily produce .npz, so the formats people actually have
    # are converted here. Both are plain text handling -- an uploaded file
    # still cannot execute anything.
    fmt = request.query_params.get("format", "").lower()

    if fmt == "csv":
        from backend.service.artifacts import ArtifactError, pack_dataset, parse_csv_dataset
        try:
            parsed = parse_csv_dataset(payload)
            features, labels, class_names = (
                parsed.features, parsed.labels, parsed.class_names)
            payload = pack_dataset(features, labels)
        except ArtifactError as e:
            raise HTTPException(status_code=400, detail=str(e))

        summary = {
            "rows": int(features.shape[0]),
            "features": int(features.shape[1]),
            "classes": len(set(labels.tolist())),
            "class_names": class_names,
            "feature_names": parsed.feature_names,
        }
        info["format"] = "csv"
        info["rows"] = int(features.shape[0])
        if class_names:
            info["class_names"] = class_names
        # What the columns were called. Without these the finished model takes
        # N numbers and cannot say which is which, so presenting them in a
        # different order gives a confident wrong answer rather than an error.
        if parsed.feature_names:
            info["feature_names"] = parsed.feature_names
            info["label_name"] = parsed.label_name
        logger.info(
            f"Converted CSV upload: {summary['rows']} rows x {summary['features']} features, "
            f"{summary['classes']} classes"
        )

    elif fmt == "text":
        # Plain text, cut into the next-token windows a language model trains
        # on. Doing it here rather than on the node keeps the dataset the same
        # shape as every other one, so the holdout split that verifies the
        # returned model needs no special case.
        from backend.service.artifacts import (
            ArtifactError, pack_dataset, parse_text_dataset, text_size_advice,
        )
        try:
            seq_len = int(request.query_params.get("seq_len", 64))
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="seq_len must be a whole number.")

        try:
            features, labels, info = parse_text_dataset(payload, seq_len=seq_len)
            payload = pack_dataset(features, labels)
        except ArtifactError as e:
            raise HTTPException(status_code=400, detail=str(e))

        info["format"] = "text"

        summary = {
            "rows": info["rows"],
            "seq_len": info["seq_len"],
            "tokens": info["tokens"],
            "vocab_size": info["vocab_size"],
            "tokenizer": info["tokenizer"],
        }

        # Said here rather than after the job, because after the job it costs a
        # contributor's GPU time to have learned it.
        advice = text_size_advice(info["source_bytes"])
        if advice:
            summary["advice"] = advice
            logger.info(f"Small text upload ({info['source_bytes']} bytes): {advice}")
        logger.info(
            f"Converted text upload: {info['source_bytes']:,} bytes into "
            f"{info['rows']:,} sequences of {info['seq_len']} tokens"
        )

    elif fmt:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown upload format {fmt!r}. Supported: csv, text, "
                   f"or omit it to upload a packed .npz.",
        )

    # Through _write_artifact rather than its own upload: this path had its own
    # copy of the write, so encryption reached the split halves but not the
    # original upload the submitter sent.
    artifact_id = await _write_artifact(db, payload, kind, info,
                                        uploader=uploader)

    logger.info(f"Stored {kind} artifact {artifact_id} ({len(payload)} bytes)")
    return {
        "status": "success",
        "artifact_id": str(artifact_id),
        "bytes": len(payload),
        **summary,
    }

@router.get("/artifacts/{artifact_id}")
async def download_artifact(
    artifact_id: str,
    db: Database = Depends(get_db),
    caller: Optional[str] = Depends(optional_node),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """Return a stored blob to the node entitled to it.

    This endpoint used to be open. Combined with the task listings, which
    published every dataset, holdout and weights id, that meant anyone able to
    reach the coordinator could read a submitter's private training data -- and
    a node could fetch the exact holdout its own work was about to be scored
    against, which quietly defeats verification altogether.

    Two rules close that:

      * a holdout is never served over HTTP to anybody, whatever token they
        hold. Verification reads it in-process via _read_artifact; nothing
        outside this service has any reason to see it.
      * every other blob is served only to a party with a claim on the task
        that references it: the node that ran the job, or the submitter who
        asked for it and is collecting the trained model.
    """
    if not caller and not submitter:
        raise HTTPException(
            status_code=401,
            detail="Send a node session token or a submitter key.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    object_id = _object_id(artifact_id)

    bucket = AsyncIOMotorGridFSBucket(db.db, bucket_name="artifacts")

    try:
        stream = await bucket.open_download_stream(object_id)
    except Exception as e:
        logger.warning(f"Artifact {artifact_id} could not be opened: {e}")
        raise HTTPException(status_code=404, detail="Artifact not found.")

    kind = (getattr(stream, "metadata", None) or {}).get("kind")

    if kind == "holdout":
        logger.warning(
            f"{caller or 'submitter'} asked for holdout {artifact_id}; refused. "
            f"A holdout is only ever read inside the coordinator."
        )
        raise HTTPException(status_code=403, detail="Artifact not available.")

    # A node may read the training data and weights of its own task; a
    # submitter may read the weights their own job produced -- but never the
    # dataset by id, which they already hold.
    if caller:
        # initial_weights_id is the model a continuing job starts from. The
        # node has to read it to load it, and only for a task assigned to it.
        claim = {"node_id": caller,
                 "$or": [{"dataset_id": artifact_id},
                         {"weights_id": artifact_id},
                         {"initial_weights_id": artifact_id}]}
    else:
        claim = {"submitter_id": submitter, "weights_id": artifact_id}

    owns = await db.tasks_collection.find_one(claim, {"_id": 1})

    if not owns:
        logger.warning(
            f"{caller or 'A submitter'} asked for artifact {artifact_id}, which is not theirs."
        )
        # 404 rather than 403: whether an id exists is itself worth not leaking.
        raise HTTPException(status_code=404, detail="Artifact not found.")

    try:
        payload = artifactCrypto.decrypt(await stream.read())
    except RuntimeError as e:
        logger.error(f"Artifact {artifact_id} could not be decrypted: {e}")
        raise HTTPException(status_code=500, detail="Artifact could not be read.")

    return Response(content=payload, media_type="application/octet-stream")

async def _read_artifact(db, artifact_id: str) -> bytes:
    bucket = AsyncIOMotorGridFSBucket(db.db, bucket_name="artifacts")
    stream = await bucket.open_download_stream(ObjectId(artifact_id))
    return artifactCrypto.decrypt(await stream.read())

async def _write_artifact(db, payload: bytes, kind: str,
                          info: Optional[dict] = None,
                          uploader: Optional[str] = None) -> str:
    bucket = AsyncIOMotorGridFSBucket(db.db, bucket_name="artifacts")

    # Encrypted before it reaches storage, so a database dump does not hand
    # over every submitter's training data.
    stored = artifactCrypto.encrypt(payload)

    metadata = {
        "kind": kind,
        "uploaded_at": datetime.utcnow(),
        "bytes": len(payload),
        "encrypted": artifactCrypto.is_enabled(),
    }
    # Who stored it. Uploading needed a caller before this, but nothing wrote
    # the caller down, so there was a name at the door and no name on the box:
    # nothing to count against, attribute, or clean up by.
    if uploader:
        metadata["uploader"] = uploader
    # What the numbers in this dataset stood for: the class names behind the
    # label column, or the tokeniser behind the ids. Small, non-secret, and
    # useless without the artifact itself -- but without it the trained model
    # comes back as bare indices.
    if info:
        metadata["info"] = dict(info)

    artifact_id = await bucket.upload_from_stream(kind, stored, metadata=metadata)
    return str(artifact_id)

async def _artifact_info(db, artifact_id: str) -> dict:
    """The stored description of a dataset, or {} if it has none."""
    try:
        stored = await db.db["artifacts.files"].find_one(
            {"_id": ObjectId(artifact_id)}, {"metadata.info": 1}
        )
    except Exception:
        return {}
    return ((stored or {}).get("metadata") or {}).get("info") or {}

async def prepare_dataset_split(db, dataset_id: str, seed: int):
    """Split a submitted dataset, keeping a holdout the node will never see.

    Returns (train_artifact_id, holdout_artifact_id). The node is handed only
    the training half, so scoring the returned weights on the holdout is a
    genuine test of whether it learned anything.
    """
    from backend.service.artifacts import pack_dataset, unpack_dataset
    from backend.service.verification import split_holdout

    raw = await _read_artifact(db, dataset_id)
    x, y = unpack_dataset(raw)          # safe loader: refuses anything executable

    # Carried onto both halves. The upload this replaces is deleted once the
    # job finishes, so a description left only on it would not survive.
    info = await _artifact_info(db, dataset_id)

    train_x, train_y, holdout_x, holdout_y = split_holdout(
        x, y, holdout_fraction=HOLDOUT_FRACTION, seed=seed
    )

    train_id = await _write_artifact(
        db, pack_dataset(train_x, train_y), "dataset", info
    )
    holdout_id = await _write_artifact(
        db, pack_dataset(holdout_x, holdout_y), "holdout", info
    )

    logger.info(
        f"Split dataset {dataset_id}: {train_x.shape[0]} train rows to the node, "
        f"{holdout_x.shape[0]} held back for verification."
    )
    return train_id, holdout_id

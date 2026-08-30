"""Submitting work, handing it to nodes, and taking results back.

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

logger = logging.getLogger("NodeDbTest")

from backend.routes.deps import (
    DATASET_RETENTION_MINUTES, Database, HOLDOUT_FRACTION, MAX_TASK_ATTEMPTS,
    MONGODB_URL, NodeConnection, CPUCapabilities, GPUCapabilities,
    TASK_CLAIM_TIMEOUT_MINUTES, authenticated_node, connected_nodes, get_db,
    node_challenges, optional_node, optional_submitter, require_node_token,
    require_uploader, system_usage, task_results, FINISHED_STATES,
)
from backend.routes.artifacts import (
    _artifact_info, _forget_dataset, _read_artifact, prepare_dataset_split,
)
from backend.routes.nodes import _node_loads

router = APIRouter()


@router.get("/get-task-results")
async def get_task_results(node_id: Optional[str] = None, db: Database = Depends(get_db)):
    try:
        # Build query based on optional node_id filter
        query = {}
        if node_id:
            query["node_id"] = node_id
            
        cursor = db.tasks_collection.find(query).sort("received_at", -1).limit(50)
        results = await cursor.to_list(length=50)
        
        for result in results:
    
            if '_id' in result:
                result['task_id'] = str(result['_id'])
                
            if 'nodeId' in result and 'node_id' not in result:
                result['node_id'] = result['nodeId']

        return [public_task(r) for r in results]
    except Exception as e:
        logger.error(f"Error retrieving task results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve task results: {str(e)}")

@router.post("/receive-task-result")
async def receive_task_result(
    result: dict,
    db: Database = Depends(get_db),
    caller: str = Depends(authenticated_node),
):
    """Legacy result sink from the old push-based flow.

    Live nodes report through /task-result/{task_id}, which checks that the
    reporting node actually owns the task. This one took an unauthenticated
    body and inserted it straight into the tasks collection, so anyone could
    write arbitrary documents into the dashboard's view of the network.
    Requiring a node token is the least it should do; the node_id is now taken
    from the token rather than the body, so a caller cannot report as someone
    else.
    """
    try:
        # Taken from the token, never the body, so the old `nodeId` fallback
        # that used to sit here can no longer apply.
        result["node_id"] = caller
        result.pop("nodeId", None)

        logger.info(f"Task result received with status: {result.get('status', 'unknown')}")

        # Every result needs its own primary key. This previously used node_id,
        # so a node's second result collided with its first.
        result["_id"] = result.get("task_id") or str(uuid.uuid4())
        
        result["received_at"] = datetime.utcnow()

        # Save to MongoDB
        await db.tasks_collection.insert_one(result)

        return {"status": "success", "message": "Result received"}
    except Exception as e:
        logger.error(f"Error storing task result: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store task result: {str(e)}")

async def _queue_task(db, node_id, task_data, submitter, client_host,
                      placement="chosen"):
    """Validate a job and put it on a node's queue.

    Shared by both submit paths so a change to validation, dataset splitting or
    the stored shape cannot apply to one and not the other.
    """
    # Optional: a dataset the node should download before training. It is split
    # here so the node only ever receives the training half.
    #
    # Taken out before validation: it is not part of the model description, and
    # validate_job returns only the fields it knows about.
    dataset_id = task_data.pop("dataset_id", None)

    # Removed for the same reason, and one more: this is the coordinator's
    # description of the uploaded data. Left in, a submitter could send their
    # own and have the node pack it into the manifest as fact.
    task_data.pop("dataset_info", None)

    # Check the job before it is queued. Without this a typo was accepted,
    # waited out the approval window, was claimed by a contributor, span up
    # their GPU and only then failed -- and a value that would not parse was
    # silently replaced by a default, so the job "succeeded" undertrained.
    try:
        task_data, spec_notes = validate_job(task_data)
    except JobSpecError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # A job with no data trained on random numbers. It burned a contributor's
    # GPU and their electricity, could not be verified -- there was nothing to
    # hold back -- and produced a model of nothing. It existed as a way to
    # prove the plumbing worked, which a contributor now does for themselves
    # from their own node page, without involving anyone else's machine.
    if not dataset_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "A job needs data to train on. Attach a CSV or a text file. "
                "To check that a machine works, run a test from its own node "
                "page instead."
            ),
        )

    holdout_id = None
    try:
        # Read before the split: prepare_dataset_split writes new artifacts
        # and returns the training half's id, so asking afterwards would
        # describe the copy rather than what was uploaded.
        dataset_info = await _artifact_info(db, dataset_id)
        dataset_id, holdout_id = await prepare_dataset_split(
            db, dataset_id, seed=int(task_data.get("holdout_seed", 0) or 0)
        )
    except Exception as e:
        logger.error(f"Could not split dataset {dataset_id}: {e}")
        raise HTTPException(status_code=400, detail=f"Dataset could not be prepared: {e}")

    # Travels inside the task so the node can put it in the manifest it
    # packs with the weights, where the submitter will actually find it.
    if dataset_info:
        task_data["dataset_info"] = dataset_info

        # Shape that came from the data rather than the form. Written into
        # the spec here so the stored task says what will actually be
        # built, and so the node and the verifier read the same numbers.
        #
        # Vocabulary matters more than it looks: inferring it from the
        # sample gives the size of the alphabet this text happens to use,
        # so a model trained on lowercase English would fail on a prompt
        # containing a capital letter. The tokeniser's own size is the
        # right answer.
        spec = task_data.setdefault("model_spec", {})
        for key in ("seq_len", "vocab_size"):
            if key in dataset_info:
                spec[key] = dataset_info[key]

        # How many classes the model predicts, for the same reason and with a
        # sharper edge. Nobody wrote this down, so the node counted the classes
        # in the half it was given and the verifier counted them in the holdout
        # -- and on a small dataset the holdout can easily hold only one of
        # them. The node then trained a two-output model, the verifier rebuilt
        # a one-output model to score it, and the weights would not load:
        #
        #   size mismatch for 4.weight: checkpoint [2, 64], model [1, 64]
        #
        # which is reported as a failed verification. A perfectly good model,
        # rejected because of where the split happened to fall. Counting once
        # here, on the whole dataset before it is split, is what makes the two
        # sides agree.
        # class_names is the list the CSV parser built from the whole file, so
        # its length is the number of classes in the data as uploaded -- not in
        # whichever half a particular reader happens to hold.
        class_names = dataset_info.get("class_names")
        if class_names:
            spec["output_dim"] = len(class_names)

        # A spreadsheet is rows of features with one label each; text is a
        # stream of tokens each predicting the next. Neither model can read
        # the other's shape, and the failure is an unreadable tensor error
        # part-way through somebody else's job.
        architecture = str(spec.get("architecture", "mlp")).lower()
        accepts = (ARCHITECTURES.get(architecture) or {}).get("accepts")
        uploaded = dataset_info.get("format")
        if accepts and uploaded and accepts != uploaded:
            wanted = {"csv": "a CSV", "text": "a text file"}
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{ARCHITECTURES[architecture]['label']} trains on "
                    f"{wanted.get(accepts, accepts)}, but you uploaded "
                    f"{wanted.get(uploaded, uploaded)}."
                ),
            )

    task = {
        "_id": f"task_{uuid.uuid4()}",
        "node_id": node_id,
        "task_data": task_data,
        "dataset_id": dataset_id,
        "holdout_artifact_id": holdout_id,
        "status": "pending",
        "attempts": 0,
        "submitted_at": datetime.utcnow(),
        "submitted_from": client_host,
        # Only the digest of the submitter's key. Without this a finished job
        # had no owner, so there was nobody to hand the trained model back to.
        "submitter_id": submitter,
        # Whether the submitter named this machine or the coordinator chose it.
        # A declined job may be moved on only when nobody picked the node.
        "placement": placement,
        "declined_by": [],
    }

    # Valid, but perhaps not wise. Advice never refuses a job -- it is shown
    # alongside the confirmation so the submitter can decide.
    #
    # Counted against the training half rather than the upload: the holdout
    # never reaches the node, so it is not what the model goes over.
    advice_info = dict(dataset_info)
    if advice_info.get("rows"):
        advice_info["rows"] = max(
            1, int(round(advice_info["rows"] * (1 - HOLDOUT_FRACTION)))
        )
    spec_notes = list(spec_notes) + advise(task_data, advice_info)

    await db.tasks_collection.insert_one(task)
    logger.info(f"Queued task {task['_id']} for node {node_id}")

    if not submitter:
        logger.info(f"Task {task['_id']} was submitted without a key; nobody can claim its result.")

    return {
        "status": "success",
        "task_id": task["_id"],
        "task_status": "pending",
        "node_id": node_id,
        "verifiable": bool(holdout_id),
        "claimable": bool(submitter),
        "notes": spec_notes,
    }

@router.post("/submit-task")
async def submit_task_anywhere(
    task_data: dict = Body(...),
    request: Request = None,
    db: Database = Depends(get_db),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """Queue work without naming a node; the coordinator picks one.

    Naming a machine by hand meant queueing behind whatever it was already
    doing, and failing outright if it went offline between the page loading and
    the job being sent -- while other GPUs sat idle.
    """
    nodes = []
    async for node in db.nodes_collection.find({"isConnected": True}):
        node["node_id"] = node.pop("_id", None)
        live = connected_nodes.get(node["node_id"])
        live_tflops = live.capabilities.get("total_gpu_tflops") if live else None
        if live_tflops is not None:
            node["total_gpu_tflops"] = live_tflops
        nodes.append(node)

    try:
        choice = pick_node(nodes, await _node_loads(db))
    except NoNodeAvailable as e:
        # 503, not 400: the request was fine, the network just has nothing to
        # run it on right now.
        raise HTTPException(status_code=503, detail=str(e))

    result = await _queue_task(
        db, choice["node_id"], task_data, submitter,
        request.client.host if request else None,
        placement="auto",
    )
    result["chosen"] = {
        "reason": choice["reason"],
        "considered": choice["considered"],
        "idle": choice["idle"],
        "queued_ahead": choice["queued_ahead"],
        "summary": summarise_choice(choice),
    }
    return result

@router.post("/submit-task/{node_id}")
async def submit_task(
    node_id: str,
    task_data: dict = Body(...),
    request: Request = None,
    db: Database = Depends(get_db),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """Queue work for a node. Called by whoever needs compute (person B).

    The task sits in the database until the node claims it via /next-task. The
    coordinator never connects to the node: contributors are behind home routers
    that drop unsolicited inbound connections.
    """
    node = await db.nodes_collection.find_one({"_id": node_id})
    if not node:
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found")

    if not node.get("isConnected"):
        raise HTTPException(status_code=409, detail="Node is not currently connected.")
    if not node.get("isAvailable"):
        raise HTTPException(status_code=409, detail="Node is not accepting work.")

    return await _queue_task(
        db, node_id, task_data, submitter,
        request.client.host if request else None,
    )


async def _owned_task(db, task_id: str, submitter: Optional[str]):
    """The task, if this submitter owns it. Raises otherwise."""
    if not submitter:
        raise HTTPException(
            status_code=401,
            detail="Send your submitter key in the X-Submitter-Key header.",
        )

    task = await db.tasks_collection.find_one({"_id": task_id})

    # A task owned by someone else is reported as missing: whether a given id
    # exists is not something a stranger should be able to probe.
    if not task or task.get("submitter_id") != submitter:
        raise HTTPException(status_code=404, detail="Task not found.")

    return task

@router.post("/cancel-task/{task_id}")
async def cancel_task(
    task_id: str,
    db: Database = Depends(get_db),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """Stop a job you submitted.

    A queued job is dropped outright. A running one cannot be killed from here
    -- the work is happening inside someone else's machine -- so the request is
    recorded and the node stops at its next step and reports back. That keeps
    one authority over the task's state instead of the coordinator and the node
    disagreeing about whether it is still running.
    """
    task = await _owned_task(db, task_id, submitter)
    status = task.get("status")

    if status in FINISHED_STATES:
        raise HTTPException(status_code=409, detail=f"That job already {status}.")

    if status == "pending":
        await db.tasks_collection.update_one(
            {"_id": task_id, "status": "pending"},
            {"$set": {"status": "cancelled",
                      "result": "Cancelled before any node picked it up.",
                      "finished_at": datetime.utcnow()}},
        )
        logger.info(f"Task {task_id} cancelled while queued.")
        return {"status": "success", "task_status": "cancelled", "stopped": True}

    await db.tasks_collection.update_one(
        {"_id": task_id}, {"$set": {"cancel_requested": True}}
    )
    logger.info(f"Cancellation requested for running task {task_id}.")
    return {"status": "success", "task_status": "running", "stopped": False}

@router.get("/task-cancelled/{task_id}")
async def task_cancelled(
    task_id: str,
    db: Database = Depends(get_db),
    caller: str = Depends(authenticated_node),
):
    """Whether the node running this task has been asked to stop."""
    task = await db.tasks_collection.find_one(
        {"_id": task_id, "node_id": caller}, {"cancel_requested": 1}
    )
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    return {"cancel_requested": bool(task.get("cancel_requested"))}

@router.post("/retry-task/{task_id}")
async def retry_task(
    task_id: str,
    changes: Optional[dict] = Body(None),
    db: Database = Depends(get_db),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """Queue the job again on the same data, optionally with different settings.

    The dataset split is reused rather than rebuilt, so a retry is scored
    against the same held-back rows as the original. That is what makes two
    runs comparable: change the model or the step count, and the difference in
    the result is the change rather than a different slice of the data.

    Re-running with new settings used to mean uploading the file again, which
    both made the comparison meaningless and put the data on the wire twice.
    """
    task = await _owned_task(db, task_id, submitter)

    if task.get("status") not in FINISHED_STATES:
        raise HTTPException(
            status_code=409,
            detail="That job has not finished yet. Cancel it first if you want to start over.",
        )

    # The data behind a finished job is deleted after a grace period, and the
    # task keeps pointing at nothing. Without this a retry queued a job with no
    # dataset at all: it trained on random numbers, could not be verified, and
    # came back marked completed -- the very thing a job submitted the normal
    # way is now refused for.
    if not task.get("dataset_id"):
        raise HTTPException(
            status_code=409,
            detail=(
                f"The data for this job was deleted "
                f"{DATASET_RETENTION_MINUTES} minutes after it finished, so it "
                f"cannot be run again. Send it as a new job with the file."
            ),
        )

    node = await db.nodes_collection.find_one({"_id": task.get("node_id")})
    if not node or not node.get("isConnected"):
        raise HTTPException(status_code=409, detail="That node is no longer connected.")
    if not node.get("isAvailable"):
        raise HTTPException(status_code=409, detail="That node is not accepting work.")

    task_data = dict(task.get("task_data") or {})
    notes: List[str] = []

    # Name the run after the one it came from. Every re-run inheriting the
    # parent's name left a list of identical labels and a download that
    # overwrote the one before it -- and the series view, which exists to
    # compare them, had nothing to tell them apart by.
    if not (changes or {}).get("model_name"):
        parent_name = str(task_data.get("model_name") or "model")
        base = re.sub(r"-v\d+$", "", parent_name.strip()) or "model"

        # Everything already called this, so two adjustments of the same run
        # do not both come out as v2.
        family = await db.tasks_collection.find(
            {"submitter_id": submitter,
             "task_data.model_name": {"$regex": rf"^{re.escape(base)}(-v\d+)?$"}},
            {"task_data.model_name": 1},
        ).to_list(length=200)

        task_data["model_name"] = next_run_name(
            parent_name,
            [(t.get("task_data") or {}).get("model_name") for t in family],
        )

    # Carry on from what the last run learned instead of starting over.
    #
    # Weights only fit the model they came from, so the shape cannot change:
    # a wider or deeper network has different tensors and load_state_dict
    # would refuse them -- on the contributor's machine, after the job had
    # been claimed. Refused here instead, where it is a sentence rather than a
    # failed run.
    warm_start = bool((changes or {}).get("continue_from"))
    initial_weights_id = None

    if warm_start:
        initial_weights_id = task.get("weights_id")
        if not initial_weights_id:
            raise HTTPException(
                status_code=409,
                detail="That job produced no model to carry on from.",
            )

        spec_changes = {k: v for k, v in ((changes or {}).get("model_spec") or {}).items()
                        if k != "architecture"}
        current = task_data.get("model_spec") or {}
        differing = [k for k, v in spec_changes.items() if current.get(k) != v]
        if differing:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Carrying on from a trained model keeps its shape, so "
                    f"{', '.join(sorted(differing))} cannot change. Train "
                    f"longer, or change the model and start fresh."
                ),
            )

    # Only the two things worth changing between runs. The dataset is fixed by
    # definition here, and its shape decided the rest of the model already.
    if changes:
        # The one thing a retry must not touch. A CSV cannot be read by a
        # language model and text cannot be read by a classifier, and the check
        # that refuses that pairing runs at upload -- which a retry does not
        # repeat. Without this the switch was accepted and failed on the node.
        original = str((task_data.get("model_spec") or {}).get("architecture", "mlp"))
        wanted = (changes.get("model_spec") or {}).get("architecture")
        if wanted and str(wanted).lower() != original.lower():
            raise HTTPException(
                status_code=400,
                detail=(
                    "A re-run keeps the same data, so it keeps the same kind of "
                    "model. Send a new job to train a different one."
                ),
            )

        for field in ("model_spec", "hyperparameters"):
            supplied = changes.get(field)
            if isinstance(supplied, dict):
                task_data[field] = {**(task_data.get(field) or {}), **supplied}

        if changes.get("model_name"):
            task_data["model_name"] = changes["model_name"]

        # Through the same validator as a first submission, so a retry cannot
        # smuggle in a value the form would have refused.
        dataset_info = task_data.pop("dataset_info", None)

        # The shape this coordinator wrote into the spec when the job was first
        # queued -- sequence length and vocabulary, read from the data. Taken
        # out before validation and put back after, for two reasons.
        #
        # validate_job drops fields the form does not offer, so leaving them in
        # meant a re-run came back without them and the node re-derived the
        # vocabulary from whichever bytes its half of the data happened to
        # contain. The re-run then had a different, smaller alphabet than the
        # original, which is both a worse model and not a comparison.
        #
        # And it warned "vocab_size is taken from your dataset; the value you
        # gave was ignored" at somebody who had typed no such thing.
        spec = task_data.get("model_spec") or {}
        derived = {key: spec[key] for key in ("seq_len", "vocab_size") if key in spec}
        if derived:
            task_data["model_spec"] = {k: v for k, v in spec.items() if k not in derived}

        try:
            task_data, notes = validate_job(task_data)
        except JobSpecError as e:
            raise HTTPException(status_code=400, detail=str(e))

        task_data["model_spec"].update(derived)

        if dataset_info:
            task_data["dataset_info"] = dataset_info
            notes = list(notes) + advise(task_data, dataset_info)

    retry = {
        "_id": f"task_{uuid.uuid4()}",
        "node_id": task["node_id"],
        "task_data": task_data,
        "dataset_id": task.get("dataset_id"),
        "holdout_artifact_id": task.get("holdout_artifact_id"),
        # The model this run begins from, or None to begin from noise.
        "initial_weights_id": initial_weights_id,
        "status": "pending",
        "attempts": 0,
        "submitted_at": datetime.utcnow(),
        "submitter_id": submitter,
        "retry_of": task_id,
        "placement": task.get("placement", "chosen"),
        "declined_by": [],
    }
    await db.tasks_collection.insert_one(retry)
    logger.info(f"Task {retry['_id']} queued as a retry of {task_id}.")

    return {
        "status": "success",
        "task_id": retry["_id"],
        "task_status": "pending",
        "verifiable": bool(retry["holdout_artifact_id"]),
        "continued": bool(initial_weights_id),
        "notes": notes,
    }

@router.get("/job-schema")
async def get_job_schema():
    """What a job may contain, so the form and the validator agree."""
    return job_schema()

@router.get("/next-task/{node_id}")
async def next_task(
    node_id: str,
    claim: bool = True,
    db: Database = Depends(get_db),
    _node: str = Depends(require_node_token),
):
    """The oldest pending task for this node. Polled by the node itself.

    With `claim=true` (the default) the task is atomically marked running, so
    two concurrent polls can never be handed the same one.

    With `claim=false` the task is only looked at. That is what the node uses
    when its owner has asked to approve each job by hand: claiming first would
    mark the task running while a human decides, and the stale-task reaper
    would then take it back mid-decision.
    """
    if not claim:
        task = await db.tasks_collection.find_one(
            {"node_id": node_id, "status": "pending"},
            sort=[("submitted_at", 1)],
        )
        if not task:
            return {"task": None}
        return {
            "task": {
                "task_id": task["_id"],
                "task_data": task.get("task_data", {}),
                "dataset_id": task.get("dataset_id"),
                # The model this job carries on from, if it carries on from one.
                "initial_weights_id": task.get("initial_weights_id"),
                "attempts": task.get("attempts", 0),
                "submitted_at": (task.get("submitted_at").isoformat()
                                 if task.get("submitted_at") else None),
            },
            "claimed": False,
        }

    task = await db.tasks_collection.find_one_and_update(
        {"node_id": node_id, "status": "pending"},
        {
            "$set": {"status": "running", "started_at": datetime.utcnow()},
            "$inc": {"attempts": 1},
        },
        sort=[("submitted_at", 1)],
        return_document=ReturnDocument.AFTER,
    )

    if not task:
        return {"task": None}

    logger.info(f"Node {node_id} claimed task {task['_id']} (attempt {task.get('attempts')})")

    return {
        "task": {
            "task_id": task["_id"],
            "task_data": task.get("task_data", {}),
            "dataset_id": task.get("dataset_id"),
            "initial_weights_id": task.get("initial_weights_id"),
            "attempts": task.get("attempts", 1),
        }
    }

@router.post("/task-result/{task_id}")
async def submit_task_result(
    task_id: str,
    payload: dict = Body(...),
    db: Database = Depends(get_db),
    caller: str = Depends(authenticated_node),
):
    """Record the outcome of a task. Only the node that owns it may report."""
    task = await db.tasks_collection.find_one({"_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.get("node_id") != caller:
        logger.warning(f"Node {caller} tried to report on task {task_id} owned by {task.get('node_id')}")
        raise HTTPException(status_code=403, detail="This task belongs to another node.")

    status = payload.get("status", "completed")
    if status not in ("completed", "failed", "rejected", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Invalid task status: {status}")

    # Nobody said no; nobody said anything. The owner was away from the screen,
    # which is the ordinary case for a machine whose whole appeal is that it
    # works while you are not there.
    #
    # This used to arrive as a plain "rejected", indistinguishable from the
    # owner actually refusing the job. For work the submitter had addressed to
    # one machine that meant the job died two minutes after it was sent, and
    # the only sign was a row in their workspace that never started. So an
    # unanswered peek puts the job back where it was rather than ending it:
    # the node offers it again, and it is still waiting when the owner returns.
    #
    # Deliberately not recorded in declined_by. That list is what stops a node
    # being offered the same job twice, and this node has not turned it down.
    if status == "rejected" and payload.get("unanswered"):
        await db.tasks_collection.update_one(
            {"_id": task_id},
            {"$set": {"status": "pending"}, "$unset": {"started_at": ""}},
        )
        logger.info(f"Task {task_id} went unanswered on {caller}; back to pending.")
        return {"status": "success", "task_id": task_id, "task_status": "pending"}

    # A declined job is not necessarily finished: if the coordinator placed it,
    # another node may still want it.
    if status == "rejected":
        moved_to = await _redispatch(db, task, caller)
        if moved_to:
            return {
                "status": "success",
                "task_id": task_id,
                "task_status": "pending",
                "redispatched_to": moved_to,
            }

    await db.tasks_collection.update_one(
        {"_id": task_id},
        {"$set": {
            "status": status,
            "result": payload.get("result"),
            "logs": payload.get("logs", []),
            "metrics": payload.get("metrics", {}),
            "weights_id": payload.get("weights_id"),
            "finished_at": datetime.utcnow(),
            "received_at": datetime.utcnow(),
            "declined_by": sorted(set(task.get("declined_by") or []) | ({caller} if status == "rejected" else set())),
        }},
    )

    logger.info(f"Task {task_id} reported {status} by node {caller}")

    # Verify in the background so the node is not held open for it.
    if status == "completed" and task.get("holdout_artifact_id") and payload.get("weights_id"):
        asyncio.create_task(_verify_quietly(task_id))

    return {"status": "success", "task_id": task_id, "task_status": status}

async def _redispatch(db, task: dict, declined_by: str) -> Optional[str]:
    """Offer a declined job to a different node.

    A decline used to end the job: the task went to "rejected" and the
    submitter had to notice and resubmit by hand, even though the network might
    have twenty other machines happy to run it.

    Only jobs the coordinator placed are moved. If the submitter named a
    machine, sending their work somewhere else would quietly override a choice
    they made deliberately -- they may have picked it for a reason.
    """
    if task.get("placement") != "auto":
        return None

    refused = set(task.get("declined_by") or []) | {declined_by}

    nodes = []
    async for node in db.nodes_collection.find({"isConnected": True}):
        node["node_id"] = node.pop("_id", None)
        if node["node_id"] in refused:
            continue            # already said no to this job
        live = connected_nodes.get(node["node_id"])
        live_tflops = live.capabilities.get("total_gpu_tflops") if live else None
        if live_tflops is not None:
            node["total_gpu_tflops"] = live_tflops
        nodes.append(node)

    try:
        choice = pick_node(nodes, await _node_loads(db))
    except NoNodeAvailable:
        return None

    await db.tasks_collection.update_one(
        {"_id": task["_id"]},
        {
            "$set": {
                "node_id": choice["node_id"],
                "status": "pending",
                "declined_by": sorted(refused),
            },
            "$unset": {"started_at": "", "result": "", "finished_at": ""},
        },
    )
    logger.info(
        f"Task {task['_id']} declined by {declined_by}; offered to {choice['node_id']}."
    )
    return choice["node_id"]

async def _verify_quietly(task_id: str):
    """Run verification without letting a failure disturb result reporting."""
    try:
        await verify_task(task_id, await get_db())
    except Exception as e:
        logger.warning(f"Verification of {task_id} did not complete: {e}")

# The id of a holdout must not leave this service. Anyone holding it could ask
# for the rows their work is scored against; publishing it in a task listing
# was how that became possible in the first place.
# A submitter id is a digest, not a credential, but a node has no business
# learning which submitters exist or correlating jobs across them.
INTERNAL_TASK_FIELDS = ("holdout_artifact_id", "submitter_id")

def public_task(task: dict, owner: bool = False) -> dict:
    """A task document safe to hand to a caller.

    `owner` means the caller proved the submitter key this job was sent with.
    A few things belong to them and to nobody else -- above all the writing
    samples, which are the model continuing verbatim snippets of their own
    text, and so are their data coming back out.
    """
    clean = {k: v for k, v in task.items() if k not in INTERNAL_TASK_FIELDS}
    # Keep the fact of a dataset, which the dashboard shows, without the id.
    clean["has_holdout"] = bool(task.get("holdout_artifact_id"))

    # Whether the data behind this job still exists. It is deleted a while
    # after the job finishes, and once it is gone the job cannot be run again
    # -- the page needs to know that before offering the button.
    clean["can_rerun"] = bool(task.get("dataset_id"))

    # dataset_info describes what the submitter's numbers meant -- the class
    # names from their label column, above all. /tasks needs no key, so
    # leaving it in published every submitter's labels to anyone who asked:
    # harmless for "setosa, versicolor", not for "relapsed" or "defaulted".
    #
    # It is internal plumbing, carried in the task so the node can put it in
    # the manifest it packs with the weights. The node reads the task through
    # /next-task, which does not come through here, so nothing needs it in an
    # HTTP response. The submitter gets it back inside their model file.
    # dataset_info describes what the submitter's numbers meant -- the class
    # names from their label column, and what the columns were called. /tasks
    # needs no key, so leaving it in published every submitter's labels to
    # anyone who asked: harmless for "setosa, versicolor", not for "relapsed".
    #
    # The owner is a different matter. It is their own description of their
    # own data, and the page needs it to tell them which columns a model reads.
    task_data = clean.get("task_data")
    if isinstance(task_data, dict) and "dataset_info" in task_data and not owner:
        clean["task_data"] = {k: v for k, v in task_data.items()
                              if k != "dataset_info"}

    metrics = clean.get("metrics")
    if isinstance(metrics, dict) and "samples" in metrics and not owner:
        clean["metrics"] = {k: v for k, v in metrics.items() if k != "samples"}

    return clean

@router.get("/tasks")
async def list_tasks(
    node_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    db: Database = Depends(get_db),
):
    """List tasks for the dashboard, newest first."""
    query = {}
    if node_id:
        query["node_id"] = node_id
    if status:
        query["status"] = status

    limit = max(1, min(limit, 200))
    cursor = db.tasks_collection.find(query).sort("submitted_at", -1).limit(limit)
    tasks = await cursor.to_list(length=limit)

    for task in tasks:
        task["task_id"] = task.pop("_id")

    return [public_task(t) for t in tasks]

@router.get("/my-tasks")
async def list_my_tasks(
    limit: int = 25,
    db: Database = Depends(get_db),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """The jobs submitted with this key, newest first.

    /tasks answers "what has this node run", which is the contributor's view.
    Until now the person who supplied the data had no view at all: they sent a
    job and lost sight of it. This is the other half.
    """
    if not submitter:
        raise HTTPException(
            status_code=401,
            detail="Send your submitter key in the X-Submitter-Key header.",
        )

    limit = max(1, min(limit, 100))
    cursor = (db.tasks_collection
              .find({"submitter_id": submitter})
              .sort("submitted_at", -1)
              .limit(limit))
    tasks = await cursor.to_list(length=limit)

    for task in tasks:
        task["task_id"] = task.pop("_id")

    return [public_task(t, owner=True) for t in tasks]

async def requeue_stale_tasks():
    """Return tasks abandoned by a node that went away back to the queue.

    Contributor machines get shut down mid-job; without this those tasks would
    sit in 'running' forever.
    """
    while True:
        try:
            cutoff = datetime.utcnow() - timedelta(minutes=TASK_CLAIM_TIMEOUT_MINUTES)
            stale = await Database.tasks_collection.find(
                {"status": "running", "started_at": {"$lt": cutoff}}
            ).to_list(length=100)

            for task in stale:
                if task.get("attempts", 0) >= MAX_TASK_ATTEMPTS:
                    logger.warning(
                        f"Task {task['_id']} abandoned after {task.get('attempts')} attempts."
                    )
                    await Database.tasks_collection.update_one(
                        {"_id": task["_id"]},
                        {"$set": {
                            "status": "failed",
                            "result": "Abandoned: node stopped responding.",
                            "finished_at": datetime.utcnow(),
                        }},
                    )
                else:
                    logger.info(f"Requeueing stale task {task['_id']}")
                    await Database.tasks_collection.update_one(
                        {"_id": task["_id"]},
                        {"$set": {"status": "pending"}, "$unset": {"started_at": ""}},
                    )

        except Exception as e:
            logger.error(f"Error requeueing stale tasks: {e}")

        await asyncio.sleep(60)

@router.post("/verify-task/{task_id}")
async def verify_task(task_id: str, db: Database = Depends(get_db)):
    """Score a returned model against the holdout the node never received."""
    task = await db.tasks_collection.find_one({"_id": task_id})
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    if task.get("status") != "completed":
        raise HTTPException(
            status_code=409,
            detail=f"Task is {task.get('status')}; only completed tasks can be verified.",
        )

    holdout_id = task.get("holdout_artifact_id")
    weights_id = task.get("weights_id")

    if not holdout_id:
        raise HTTPException(status_code=409, detail="This task has no holdout to verify against.")
    if not weights_id:
        raise HTTPException(status_code=409, detail="The node returned no weights to verify.")

    from backend.service.artifacts import unpack_dataset, unpack_state_dict
    from backend.service.verification import summarise, verify_training_result

    try:
        holdout_x, holdout_y = unpack_dataset(await _read_artifact(db, holdout_id))
        state_dict = unpack_state_dict(await _read_artifact(db, weights_id))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not load artifacts: {e}")

    task_data = task.get("task_data", {}) or {}
    spec = task_data.get("model_spec") or {}

    # A job continuing an earlier one is handed a model that already works.
    # Nothing stops a node from claiming it, sleeping, and handing the same
    # weights straight back -- and they would score well, because they were
    # good when they arrived. So the starting point goes to the verifier,
    # which already knows how to refuse weights identical to what it began
    # with; it simply had nothing to compare against until now.
    initial_state = None
    initial_id = task.get("initial_weights_id")
    if initial_id:
        try:
            initial_state = unpack_state_dict(await _read_artifact(db, str(initial_id)))
        except Exception as e:
            logger.warning(
                f"Could not read the starting weights for {task_id}: {e}. "
                f"Verifying without the unchanged-weights check."
            )

    # Evaluation is CPU-bound; keep it off the event loop.
    report = await asyncio.to_thread(
        verify_training_result,
        state_dict, spec, holdout_x, holdout_y,
        task.get("metrics", {}),
        None,
        initial_state,
    )

    await db.tasks_collection.update_one(
        {"_id": task_id},
        {"$set": {"verification": report, "verified_at": datetime.utcnow()}},
    )

    logger.info(f"Task {task_id} verification: {summarise(report)}")
    return {"task_id": task_id, **report}

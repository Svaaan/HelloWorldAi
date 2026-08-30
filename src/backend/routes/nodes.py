"""Registering nodes, their heartbeats, and proving who they are.

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
    require_uploader, system_usage, task_results,
)


router = APIRouter()


def get_gpu_info():
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        gpu_info = []

        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            temperature = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU)

            gpu_info.append({
                "name": name,
                "total_memory": round(memory_info.total / 1024 ** 2),
                "free_memory": round(memory_info.free / 1024 ** 2),
                "used_memory": round(memory_info.used / 1024 ** 2),
                "load_percentage": utilization.gpu,
                "temperature": temperature
            })

        pynvml.nvmlShutdown()
        return gpu_info

    except pynvml.NVMLError as e:
        logger.warning(f"⚠️ NVML error: {e}")
        return []
    except Exception as e:
        logger.error(f"Error in get_gpu_info: {e}")
        return []

# Database synchronization task
async def sync_nodes_with_db():

    while True:
        try:
            # Update database with current in-memory state
            for node_id, node in connected_nodes.items():
                try:
                    await Database.nodes_collection.update_one(
                        {"_id": node_id},
                        {"$set": {
                            "isConnected": node.isConnected,
                            "isAvailable": node.isAvailable,
                            "last_heartbeat": node.last_heartbeat or datetime.utcnow()
                        }},
                        upsert=False
                    )
                except Exception as e:
                    logger.error(f"Error updating node {node_id} in database: {e}")
            
            # Detect stale connections and mark them as disconnected
            stale_threshold = datetime.utcnow() - timedelta(minutes=5)
            stale_nodes = await Database.nodes_collection.find({
                "isConnected": True,
                "last_heartbeat": {"$lt": stale_threshold}
            }).to_list(length=100)
            
            for node in stale_nodes:
                node_id = node["_id"]
                logger.info(f"Marking stale node as disconnected: {node_id}")
                await Database.nodes_collection.update_one(
                    {"_id": node_id},
                    {"$set": {"isConnected": False}}
                )
                if node_id in connected_nodes:
                    connected_nodes[node_id].isConnected = False
            
            # Load any nodes from DB that aren't in memory
            db_nodes = await Database.nodes_collection.find({
                "isConnected": True,
                "_id": {"$nin": list(connected_nodes.keys())}
            }).to_list(length=100)
            
            for node in db_nodes:
                node_id = node["_id"]
                logger.info(f"Loading node from database to memory: {node_id}")
                connected_nodes[node_id] = NodeConnection(
                    node_id=node_id,
                    capabilities=node.get("capabilities", {"cpu": {}, "gpu": []}),
                    isConnected=True,
                    isAvailable=node.get("isAvailable", False),
                    cpu_usage=node.get("cpu_usage", 0.0),
                    gpu_usage=node.get("gpu_usage", 0.0),
                    cpu_benchmark=node.get("cpu_benchmark"),
                    gpu_benchmark=node.get("gpu_benchmark"),
                    last_heartbeat=node.get("last_heartbeat")
                )
                
        except Exception as e:
            logger.error(f"Error in sync_nodes_with_db: {e}")
        
        # Sleep before next sync
        await asyncio.sleep(30)

@router.patch("/toggle-availability/{node_id}")
async def toggle_availability(
    node_id: str,
    db: Database = Depends(get_db),
    _node: str = Depends(require_node_token),
):
    try:
        # Fetch node from MongoDB
        node = await db.nodes_collection.find_one({"_id": node_id})
        if not node:
            raise HTTPException(
                status_code=404, detail=f"Node {node_id} not found")

        # Toggle availability
        new_availability = not node.get("isAvailable", False)

        # Update in MongoDB
        await db.nodes_collection.update_one(
            {"_id": node_id},
            {"$set": {"isAvailable": new_availability}}
        )

        # Update in-memory state if present
        if node_id in connected_nodes:
            connected_nodes[node_id].isAvailable = new_availability

        logger.info(f"🔁 Toggled availability for {node_id} to {new_availability}")

        return {
            "status": "success",
            "node_id": node_id,
            "isAvailable": new_availability,
            "message": f"Node availability toggled to {new_availability}"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling node availability: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to toggle availability: {str(e)}")

@router.post("/connect-node")
async def connect_node(node: NodeConnection, request: Request, db: Database = Depends(get_db)):
    try:
        # ✅ Validate GPU presence
        gpu_capabilities = node.capabilities.get("gpu", [])
        if not gpu_capabilities or (
            isinstance(gpu_capabilities, list) and
            (len(gpu_capabilities) == 0 or gpu_capabilities[0].get("name") in ["No GPU Detected", None, ""])
        ):
            return {
                "status": "rejected",
                "reason": "No valid GPU detected. Node connection refused."
            }

        node_id = None
        if node.public_key:
            existing_node = await db.nodes_collection.find_one({"public_key": node.public_key})
            if existing_node:
                node_id = existing_node["_id"]
                logger.info(f"ℹ️ Existing node found. Reusing node_id: {node_id}")
            else:
                node_id = f"node_{uuid.uuid4()}"
                logger.info(f"🆕 New node. Generated node_id: {node_id}")
        else:
            raise HTTPException(status_code=400, detail="Public key is required.")

        # ✅ Set runtime props
        node.isConnected = True
        node.last_heartbeat = datetime.utcnow()

        # ✅ Build full doc
        node_document = {
            "_id": node_id,
            "public_key": node.public_key,
            "isConnected": True,
            "isAvailable": node.isAvailable,
            "last_connected": datetime.utcnow(),
            "last_heartbeat": node.last_heartbeat,
            "has_gpu": bool(gpu_capabilities and gpu_capabilities[0].get("name") not in ["No GPU Detected", None, ""]),
        }

        await db.nodes_collection.update_one(
            {"_id": node_id},
            {"$set": node_document},
            upsert=True
        )

        # ✅ Update in-memory node cache
        connected_nodes[node_id] = node

        logger.info(f"✅ Node successfully connected: {node_id}")
        return {
            "status": "success",
            "message": "Node connected",
            "node_id": node_id,
        }

    except Exception as e:
        logger.error(f"❌ Coordinator error in /connect-node: {e}")
        raise HTTPException(status_code=500, detail=f"Coordinator failed to connect node: {str(e)}")

@router.post("/node-heartbeat/{node_id}")
async def node_heartbeat(
    node_id: str,
    status: dict = Body(...),
    db: Database = Depends(get_db),
    _node: str = Depends(require_node_token),
):
    """Endpoint for nodes to send regular heartbeats with status updates"""
    try:
        if node_id not in connected_nodes:
            # Check if node exists in database
            node_doc = await db.nodes_collection.find_one({"_id": node_id})
            if not node_doc:
                raise HTTPException(status_code=404, detail=f"Node {node_id} not registered")
            
            # Load basic node info from database
            connected_nodes[node_id] = NodeConnection(
                node_id=node_id,
                isConnected=True,
                isAvailable=node_doc.get("isAvailable", False)
            )
        
        # Update node status in memory
        node = connected_nodes[node_id]
        node.last_heartbeat = datetime.utcnow()
        node.isConnected = True
        
        # Update dynamic metrics in memory only
        if "cpu_usage" in status:
            node.cpu_usage = status["cpu_usage"]
        if "gpu_usage" in status:
            node.gpu_usage = status["gpu_usage"]
        if "capabilities" in status:
            node.capabilities = status["capabilities"]
        if "cpu_benchmark" in status:
            node.cpu_benchmark = status["cpu_benchmark"]
        if "gpu_benchmark" in status:
            node.gpu_benchmark = status["gpu_benchmark"]
        
        # Persist what outlives this process. Throughput used to be kept in
        # memory alone, so a coordinator restart reported every node as 0
        # TFLOPS until its next heartbeat -- long enough for the dashboard to
        # look wrong and for job placement to have nothing to rank on.
        persisted = {
            "last_heartbeat": node.last_heartbeat,
            "isConnected": True,
        }

        capabilities = status.get("capabilities")
        if isinstance(capabilities, dict):
            persisted["capabilities"] = capabilities
            tflops = capabilities.get("total_gpu_tflops")
            if tflops is not None:
                persisted["total_gpu_tflops"] = tflops

        # A node knows things the coordinator cannot see -- a self test never
        # becomes a task here, so without this the machine looks idle while it
        # is flat out.
        if "busy" in status:
            persisted["reported_busy"] = bool(status["busy"])

        await db.nodes_collection.update_one({"_id": node_id}, {"$set": persisted})
        
        return {"status": "success", "timestamp": node.last_heartbeat}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing heartbeat for node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process heartbeat: {str(e)}")

@router.get("/nodes")
async def get_connected_nodes(node_id: Optional[str] = None, db: Database = Depends(get_db)):
    try:
        query = {}
        if node_id:
            query["_id"] = node_id

        nodes_cursor = db.nodes_collection.find(query)
        nodes = []

        async for node in nodes_cursor:
            node_id_from_db = node["_id"]
            
            # Create base node info from database
            node_info = {
                "node_id": node_id_from_db,
                "isConnected": node.get("isConnected", False),
                "isAvailable": node.get("isAvailable", False),
                "isAuthenticated": node.get("isAuthenticated", False),  # ✅ Already here!
                "last_verified": node.get("last_verified"),  # ✅ Add this!
                "last_connected": node.get("last_connected"),
                "last_heartbeat": node.get("last_heartbeat"),
                "has_gpu": node.get("has_gpu", False)
            }

            # Add dynamic data from memory if available
            live_node = connected_nodes.get(node_id_from_db)
            if live_node:
                # Add dynamic hardware info from in-memory data
                node_info["cpu_usage"] = live_node.cpu_usage
                node_info["gpu_usage"] = live_node.gpu_usage
                node_info["cpu_benchmark"] = live_node.cpu_benchmark
                node_info["gpu_benchmark"] = live_node.gpu_benchmark
                node_info["capabilities"] = live_node.capabilities
                node_info["isConnected"] = live_node.isConnected  # Use most recent connection status
                node_info["total_gpu_tflops"] = live_node.capabilities.get("total_gpu_tflops", 0)  # ✅ Add TFLOPS

            nodes.append(node_info)

        return nodes

    except Exception as e:
        logger.error(f"Error retrieving nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve nodes: {str(e)}")

@router.get("/available-nodes")
async def get_available_nodes(db: Database = Depends(get_db)):
    try:
        query = {"isConnected": True, "isAvailable": True}

        cursor = db.nodes_collection.find(query)
        nodes = []
        
        # One aggregation rather than a query per node.
        loads = await _node_loads(db)
        running = {
            row["_id"]
            async for row in db.tasks_collection.aggregate([
                {"$match": {"status": "running"}},
                {"$group": {"_id": "$node_id"}},
            ])
        }

        async for node in cursor:
            node_id = node.pop("_id", None)
            node["node_id"] = node_id

            # Busy from either direction: a job this coordinator handed out, or
            # something the node is doing of its own accord.
            node["busy"] = node_id in running or bool(node.get("reported_busy"))
            node["queued"] = loads.get(node_id, 0)

            live_node = connected_nodes.get(node_id)
            live_tflops = (live_node.capabilities.get("total_gpu_tflops")
                           if live_node else None)
            # The stored value is the fallback while a restarted coordinator
            # waits for the next heartbeat.
            node["total_gpu_tflops"] = (
                live_tflops if live_tflops is not None
                else node.get("total_gpu_tflops", 0)
            )

            nodes.append(node)

        return nodes
    except Exception as e:
        logger.error(f"Error retrieving available nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve available nodes: {str(e)}")

@router.get("/get-connected-nodes-count")
async def get_connected_nodes_count():
    try:
        # Count the connected nodes in-memory
        in_memory_count = sum(1 for node in connected_nodes.values() if node.isConnected)
        
        logger.info(f"Connected nodes (in-memory): {in_memory_count}")

        return {"connected_nodes_count": in_memory_count}
    
    except Exception as e:
        logger.error(f"Error counting connected nodes: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to count connected nodes: {str(e)}")

async def _node_loads(db) -> Dict[str, int]:
    """How many jobs are queued or running on each node."""
    pipeline = [
        {"$match": {"status": {"$in": list(BUSY_STATUSES)}}},
        {"$group": {"_id": "$node_id", "count": {"$sum": 1}}},
    ]
    return {
        row["_id"]: row["count"]
        async for row in db.tasks_collection.aggregate(pipeline)
    }

@router.get("/generate-challenge/{node_id}")
async def generate_challenge(node_id: str, db: Database = Depends(get_db)):
    node = await db.nodes_collection.find_one({"_id": node_id})
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")

    challenge = secrets.token_hex(16)

    node_challenges[node_id] = {
        "challenge": challenge,
        "expires_at": datetime.utcnow() + timedelta(minutes=2)  # ⏰ Challenge expires in 2 minutes
    }

    logger.info(f"🔐 Generated challenge for node {node_id}: {challenge}")
    return {"challenge": challenge}

@router.post("/verify-challenge/{node_id}")
async def verify_node_challenge(node_id: str, signature_payload: dict, db: Database = Depends(get_db)):
    signature = signature_payload.get("signature")
    if not signature:
        logger.warning(f"Node {node_id} failed to provide signature.")
        raise HTTPException(status_code=400, detail="Signature is required")

    challenge_entry = node_challenges.get(node_id)
    if not challenge_entry:
        logger.warning(f"No challenge found for node {node_id}.")
        raise HTTPException(status_code=400, detail="No challenge found for node")

    if datetime.utcnow() > challenge_entry["expires_at"]:
        logger.warning(f"Challenge for node {node_id} expired.")
        raise HTTPException(status_code=400, detail="Challenge expired")

    challenge = challenge_entry["challenge"]

    node = await db.nodes_collection.find_one({"_id": node_id})
    if not node or not node.get("public_key"):
        logger.warning(f"Node {node_id} or its public key not found in database.")
        raise HTTPException(status_code=404, detail="Node or public key not found")

    public_key_pem = node["public_key"]

    if verify_signature(public_key_pem, challenge, signature):
        logger.info(f"✅ Node {node_id} verified successfully with challenge-response.")

        await db.nodes_collection.update_one(
            {"_id": node_id},
            {"$set": {
                "isAuthenticated": True,
                "last_verified": datetime.utcnow()
            }}
        )

        node_challenges.pop(node_id, None)

        token = issue_node_token(node_id)
        logger.info(f"🎟️ Issued session token for node {node_id}.")

        return {
            "status": "success",
            "message": "Node verified",
            "token": token,
            "expires_in": NODE_TOKEN_TTL,
        }

    else:
        logger.warning(f"Invalid signature for node {node_id}.")
        raise HTTPException(status_code=401, detail="Invalid signature")

@router.post("/find-node-id")
async def find_node_id(request: Request):
    data = await request.json()
    public_key = data.get("public_key")
    if not public_key:
        raise HTTPException(status_code=400, detail="Public key is required.")

    # Query your node database for matching public key
    node = await db["nodes"].find_one({"public_key": public_key})
    if not node:
        raise HTTPException(status_code=404, detail="Node not found for provided public key.")

    return {"node_id": node["_id"]}

async def cleanup_expired_challenges():
    while True:
        try:
            now = datetime.utcnow()
            expired = [node_id for node_id, data in node_challenges.items() if data["expires_at"] < now]

            for node_id in expired:
                logger.info(f"🧹 Cleaning up expired challenge for node {node_id}")
                del node_challenges[node_id]

        except Exception as e:
            logger.error(f"Error cleaning up expired challenges: {e}")

        await asyncio.sleep(60)  # Check every 60 seconds

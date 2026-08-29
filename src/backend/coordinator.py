import uuid
import time
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
from datetime import datetime, timedelta
import secrets
import asyncio
from backend.database.nodedb import db 
from backend.service.authNodeService import verify_signature 
from backend.service.tokenService import issue_node_token, read_node_token, NODE_TOKEN_TTL
from backend.service.submitterService import read_submitter_key
from backend.service.jobSpec import JobSpecError, job_schema, validate_job
from backend.service.nodePicker import (
    BUSY_STATUSES, NoNodeAvailable, pick_node, summarise_choice,
)
import logging
from motor.motor_asyncio import (
    AsyncIOMotorClient,
    AsyncIOMotorDatabase,
    AsyncIOMotorCollection,
    AsyncIOMotorGridFSBucket,
)
from pymongo import ReturnDocument
from bson import ObjectId
from fastapi.responses import Response
from backend.service.artifacts import MAX_ARTIFACT_BYTES
from backend.service import artifactCrypto

import pynvml
import os  # ✅ Import os for env vars

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("NodeDbTest")

# ✅ Database configuration (using env variable with default fallback)
MONGODB_URL = os.getenv("MONGO_URI", "mongodb://mongo_test:27017")
DB_NAME = "NodeDbTest"
MAX_RECONNECT_ATTEMPTS = 5
RECONNECT_DELAY = 5  # seconds

node_challenges = {}

# A node that claims a task and then dies must not strand it forever.
TASK_CLAIM_TIMEOUT_MINUTES = int(os.getenv("TASK_CLAIM_TIMEOUT_MINUTES", 10))
MAX_TASK_ATTEMPTS = int(os.getenv("MAX_TASK_ATTEMPTS", 3))

# Fraction of a submitted dataset withheld from the node so its returned
# model can be scored on data it never saw.
HOLDOUT_FRACTION = float(os.getenv("HOLDOUT_FRACTION", 0.2))

# How long a finished job's dataset is kept before it is deleted. Long enough
# for verification to run and for a retry to reuse the same split; short enough
# that submitted data does not accumulate indefinitely.
DATASET_RETENTION_MINUTES = int(os.getenv("DATASET_RETENTION_MINUTES", 60))

# Database connection class
class Database:
    client: AsyncIOMotorClient = None
    db: AsyncIOMotorDatabase = None
    nodes_collection: AsyncIOMotorCollection = None
    tasks_collection: AsyncIOMotorCollection = None

    @classmethod
    async def connect_db(cls):
        if cls.client is None:
            for attempt in range(MAX_RECONNECT_ATTEMPTS):
                try:
                    logger.info(f"Connecting to MongoDB (attempt {attempt+1}/{MAX_RECONNECT_ATTEMPTS}) using URL: {MONGODB_URL}")
                    cls.client = AsyncIOMotorClient(MONGODB_URL)
                    # Test connection with a simple ping
                    await cls.client.admin.command('ping')
                    logger.info("Connected to MongoDB successfully! ✅")
                    break
                except Exception as e:
                    logger.error(f"Failed to connect to MongoDB: {e}")
                    if attempt < MAX_RECONNECT_ATTEMPTS - 1:
                        logger.info(f"Retrying in {RECONNECT_DELAY} seconds...")
                        await asyncio.sleep(RECONNECT_DELAY)
                    else:
                        logger.error("Maximum reconnection attempts reached. Failed to connect to MongoDB.")
                        raise HTTPException(status_code=500, detail="Database connection failed")

            cls.db = cls.client[DB_NAME]
            cls.nodes_collection = cls.db.nodes
            cls.tasks_collection = cls.db.tasks

            # ✅ Create indices (with public_key as unique) and wrap in try/except
            try:
                await cls.nodes_collection.create_index("_id")
                await cls.nodes_collection.create_index("isAvailable")
                await cls.nodes_collection.create_index("isConnected")
                await cls.nodes_collection.create_index("public_key", unique=True)  # <-- ✅ Enforce uniqueness
                await cls.tasks_collection.create_index("node_id")
                await cls.tasks_collection.create_index("received_at")
                logger.info("✅ Database indices created")
            except Exception as e:
                logger.error(f"❌ Failed to create indexes: {e}")


    @classmethod
    async def close_db(cls):
        if cls.client:
            cls.client.close()
            cls.client = None
            logger.info("MongoDB connection closed ✅")

# Database dependency
async def get_db():
    if Database.client is None:
        await Database.connect_db()
    return Database


def authenticated_node(authorization: Optional[str] = Header(default=None)) -> str:
    """Return the node_id a valid bearer token was issued for.

    The token comes from /verify-challenge, so holding one proves the caller
    controls the private key the node registered with.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing node session token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization.split(" ", 1)[1].strip()
    token_node_id = read_node_token(token)

    if token_node_id is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired node session token. Re-verify the node.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return token_node_id


def optional_node(authorization: Optional[str] = Header(default=None)) -> Optional[str]:
    """The node a token belongs to, or None when there is no usable token.

    The raising version cannot be used where either a node or a submitter is
    acceptable, because FastAPI resolves every dependency before the endpoint
    body runs -- a missing node token would 401 a perfectly good submitter.
    """
    if not authorization:
        return None
    try:
        return authenticated_node(authorization)
    except HTTPException:
        return None


def optional_submitter(
    x_submitter_key: Optional[str] = Header(default=None),
) -> Optional[str]:
    """The submitter id proved by the X-Submitter-Key header, if any."""
    return read_submitter_key(x_submitter_key)


def require_node_token(node_id: str, caller: str = Depends(authenticated_node)) -> str:
    """Require a token issued for this exact node_id."""
    if caller != node_id:
        logger.warning(f"🚫 Token for {caller} was used against node {node_id}.")
        raise HTTPException(status_code=403, detail="Token does not grant access to this node.")
    return caller


app = FastAPI()
task_results = []


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lifecycle events
@app.on_event("startup")
async def startup_event():
    await Database.connect_db()
    asyncio.create_task(sync_nodes_with_db())
    asyncio.create_task(cleanup_expired_challenges())  # ✅ Start cleanup task
    asyncio.create_task(requeue_stale_tasks())
    asyncio.create_task(forget_finished_datasets())


@app.on_event("shutdown")
async def shutdown_event():
    await Database.close_db()


class GPUCapabilities(BaseModel):
    name: str = "No GPU"
    total_memory: Optional[int] = None
    free_memory: Optional[int] = None
    used_memory: Optional[int] = None
    load_percentage: Optional[float] = None
    temperature: Optional[float] = None


class CPUCapabilities(BaseModel):
    brand: str
    cores: int
    threads: int
    max_freq: Optional[float] = None
    min_freq: Optional[float] = None
    current_freq: Optional[float] = None


class NodeConnection(BaseModel):
    node_id: Optional[str] = None  # ✅ Coordinator generates this!
    ip: str
    country: Optional[str] = "Unknown"
    public_key: Optional[str] = None  # ✅ Provided by frontend (browser)
    capabilities: Dict = {
        "cpu": {},
        "gpu": []
    }
    isConnected: bool = False
    isAvailable: bool = False
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    cpu_benchmark: Optional[int] = None
    gpu_benchmark: Optional[int] = None
    last_heartbeat: Optional[datetime] = None

connected_nodes: Dict[str, NodeConnection] = {}

system_usage = {
    "cpu_usage": 0.0,
    "gpu_usage": 0.0,
    "last_updated": time.time()
}


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
                    ip=node.get("ip", "unknown"),
                    country=node.get("country", "Unknown"),
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


@app.patch("/toggle-availability/{node_id}")
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


@app.post("/connect-node")
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
        node.ip = request.client.host
        node.isConnected = True
        node.last_heartbeat = datetime.utcnow()

        # ✅ Build full doc
        node_document = {
            "_id": node_id,
            "ip": node.ip,
            "country": node.country,
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
            "ip": node.ip,
        }

    except Exception as e:
        logger.error(f"❌ Coordinator error in /connect-node: {e}")
        raise HTTPException(status_code=500, detail=f"Coordinator failed to connect node: {str(e)}")


@app.post("/node-heartbeat/{node_id}")
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
                ip=node_doc.get("ip", "unknown"),
                country=node_doc.get("country", "Unknown"),
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

        await db.nodes_collection.update_one({"_id": node_id}, {"$set": persisted})
        
        return {"status": "success", "timestamp": node.last_heartbeat}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing heartbeat for node {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process heartbeat: {str(e)}")


@app.get("/get-task-results")
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




@app.post("/receive-task-result")
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

@app.get("/nodes")
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
                "ip": node.get("ip", "unknown"),
                "country": node.get("country", "Unknown"),
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





@app.get("/available-nodes")
async def get_available_nodes(db: Database = Depends(get_db)):
    try:
        query = {"isConnected": True, "isAvailable": True}

        cursor = db.nodes_collection.find(query)
        nodes = []
        
        async for node in cursor:
            node_id = node.pop("_id", None)
            node["node_id"] = node_id

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



@app.get("/get-connected-nodes-count")
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

    # Check the job before it is queued. Without this a typo was accepted,
    # waited out the approval window, was claimed by a contributor, span up
    # their GPU and only then failed -- and a value that would not parse was
    # silently replaced by a default, so the job "succeeded" undertrained.
    try:
        task_data, spec_notes = validate_job(task_data)
    except JobSpecError as e:
        raise HTTPException(status_code=400, detail=str(e))

    holdout_id = None
    if dataset_id:
        try:
            dataset_id, holdout_id = await prepare_dataset_split(
                db, dataset_id, seed=int(task_data.get("holdout_seed", 0) or 0)
            )
        except Exception as e:
            logger.error(f"Could not split dataset {dataset_id}: {e}")
            raise HTTPException(status_code=400, detail=f"Dataset could not be prepared: {e}")

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


@app.post("/submit-task")
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


@app.post("/submit-task/{node_id}")
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


# Terminal states: nothing more will happen to a task in one of these.
FINISHED_STATES = ("completed", "failed", "rejected", "cancelled")


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


@app.post("/cancel-task/{task_id}")
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


@app.get("/task-cancelled/{task_id}")
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


@app.post("/retry-task/{task_id}")
async def retry_task(
    task_id: str,
    db: Database = Depends(get_db),
    submitter: Optional[str] = Depends(optional_submitter),
):
    """Queue the same job again, on the same node.

    The dataset split is reused rather than rebuilt, so a retry is scored
    against the same held-back rows as the original and the two runs can
    honestly be compared.
    """
    task = await _owned_task(db, task_id, submitter)

    if task.get("status") not in FINISHED_STATES:
        raise HTTPException(
            status_code=409,
            detail="That job has not finished yet. Cancel it first if you want to start over.",
        )

    node = await db.nodes_collection.find_one({"_id": task.get("node_id")})
    if not node or not node.get("isConnected"):
        raise HTTPException(status_code=409, detail="That node is no longer connected.")
    if not node.get("isAvailable"):
        raise HTTPException(status_code=409, detail="That node is not accepting work.")

    retry = {
        "_id": f"task_{uuid.uuid4()}",
        "node_id": task["node_id"],
        "task_data": task.get("task_data"),
        "dataset_id": task.get("dataset_id"),
        "holdout_artifact_id": task.get("holdout_artifact_id"),
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
    }


@app.get("/job-schema")
async def get_job_schema():
    """What a job may contain, so the form and the validator agree."""
    return job_schema()


@app.get("/next-task/{node_id}")
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
            "attempts": task.get("attempts", 1),
        }
    }


@app.post("/task-result/{task_id}")
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


def public_task(task: dict) -> dict:
    """A task document safe to hand to a caller."""
    clean = {k: v for k, v in task.items() if k not in INTERNAL_TASK_FIELDS}
    # Keep the fact of a dataset, which the dashboard shows, without the id.
    clean["has_holdout"] = bool(task.get("holdout_artifact_id"))
    return clean


@app.get("/tasks")
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


@app.get("/my-tasks")
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

    return [public_task(t) for t in tasks]


async def _forget_dataset(db, task: dict) -> int:
    """Delete the dataset copies a finished job no longer needs.

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


@app.post("/artifacts")
async def upload_artifact(request: Request, db: Database = Depends(get_db)):
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

    kind = request.query_params.get("kind", "dataset")
    if kind not in ("dataset", "weights"):
        raise HTTPException(status_code=400, detail=f"Unknown artifact kind: {kind}")

    summary = {}

    # A browser cannot easily produce .npz, so CSV is converted here. Parsing is
    # plain text handling -- an uploaded file still cannot execute anything.
    if request.query_params.get("format", "").lower() == "csv":
        from backend.service.artifacts import ArtifactError, pack_dataset, parse_csv_dataset
        try:
            features, labels, class_names = parse_csv_dataset(payload)
            payload = pack_dataset(features, labels)
        except ArtifactError as e:
            raise HTTPException(status_code=400, detail=str(e))

        summary = {
            "rows": int(features.shape[0]),
            "features": int(features.shape[1]),
            "classes": len(set(labels.tolist())),
            "class_names": class_names,
        }
        logger.info(
            f"Converted CSV upload: {summary['rows']} rows x {summary['features']} features, "
            f"{summary['classes']} classes"
        )

    # Through _write_artifact rather than its own upload: this path had its own
    # copy of the write, so encryption reached the split halves but not the
    # original upload the submitter sent.
    artifact_id = await _write_artifact(db, payload, kind)

    logger.info(f"Stored {kind} artifact {artifact_id} ({len(payload)} bytes)")
    return {
        "status": "success",
        "artifact_id": str(artifact_id),
        "bytes": len(payload),
        **summary,
    }


@app.get("/artifacts/{artifact_id}")
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
    try:
        object_id = ObjectId(artifact_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed artifact id.")

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
        claim = {"node_id": caller,
                 "$or": [{"dataset_id": artifact_id}, {"weights_id": artifact_id}]}
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


async def _write_artifact(db, payload: bytes, kind: str) -> str:
    bucket = AsyncIOMotorGridFSBucket(db.db, bucket_name="artifacts")

    # Encrypted before it reaches storage, so a database dump does not hand
    # over every submitter's training data.
    stored = artifactCrypto.encrypt(payload)

    artifact_id = await bucket.upload_from_stream(
        kind, stored,
        metadata={
            "kind": kind,
            "uploaded_at": datetime.utcnow(),
            "bytes": len(payload),
            "encrypted": artifactCrypto.is_enabled(),
        },
    )
    return str(artifact_id)


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

    train_x, train_y, holdout_x, holdout_y = split_holdout(
        x, y, holdout_fraction=HOLDOUT_FRACTION, seed=seed
    )

    train_id = await _write_artifact(db, pack_dataset(train_x, train_y), "dataset")
    holdout_id = await _write_artifact(db, pack_dataset(holdout_x, holdout_y), "holdout")

    logger.info(
        f"Split dataset {dataset_id}: {train_x.shape[0]} train rows to the node, "
        f"{holdout_x.shape[0]} held back for verification."
    )
    return train_id, holdout_id


@app.post("/verify-task/{task_id}")
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

    # Evaluation is CPU-bound; keep it off the event loop.
    report = await asyncio.to_thread(
        verify_training_result,
        state_dict, spec, holdout_x, holdout_y,
        task.get("metrics", {}),
    )

    await db.tasks_collection.update_one(
        {"_id": task_id},
        {"$set": {"verification": report, "verified_at": datetime.utcnow()}},
    )

    logger.info(f"Task {task_id} verification: {summarise(report)}")
    return {"task_id": task_id, **report}


@app.get("/generate-challenge/{node_id}")
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
        
@app.post("/verify-challenge/{node_id}")
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


    
@app.post("/find-node-id")
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
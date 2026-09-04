"""The pieces every route module needs: the database, who is calling, shapes.

coordinator.py was 2,586 lines holding thirty routes, the auth dependencies,
the database wrapper, the request models and the shared in-memory state. That
made simple questions expensive -- working out which endpoints were
authenticated needed a script rather than a glance.

This is the bottom of the stack. It imports nothing from the route modules, so
they can all import it.
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
from fastapi import (
    APIRouter, Body, Cookie, Depends, Header, HTTPException, Request,
)
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


# ✅ Database configuration (using env variable with default fallback)
MONGODB_URL = os.getenv("MONGO_URI", "mongodb://mongo_dev:27017")

# Set per environment; see backend/database/nodedb.py. Hardcoding this sent
# production writes to a database named Test.
DB_NAME = os.getenv("MONGO_DB", "NodeDbDev")

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
    # GitHub sign-ins, and the submitter digests each one is linked to. Holds no
    # key and no access token -- see accountService for why that matters.
    accounts_collection: AsyncIOMotorCollection = None

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
            cls.accounts_collection = cls.db.accounts

            # ✅ Create indices (with public_key as unique) and wrap in try/except
            try:
                await cls.nodes_collection.create_index("_id")
                await cls.nodes_collection.create_index("isAvailable")
                await cls.nodes_collection.create_index("isConnected")
                await cls.nodes_collection.create_index("public_key", unique=True)  # <-- ✅ Enforce uniqueness
                await cls.tasks_collection.create_index("node_id")
                await cls.tasks_collection.create_index("received_at")
                # The retention sweep asks "does any account own this digest"
                # for every finished job it looks at, on a timer, forever.
                await cls.accounts_collection.create_index("submitter_ids")
                # And it groups an owner's finished jobs by age.
                await cls.tasks_collection.create_index(
                    [("submitter_id", 1), ("finished_at", -1)])
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
    """The submitter id proved by the X-Submitter-Key header, if any.

    The key path, unchanged. A signed-in browser is resolved by
    submitter_or_session below, which falls back to this.
    """
    return read_submitter_key(x_submitter_key)


async def submitter_or_session(
    x_submitter_key: Optional[str] = Header(default=None),
    hw_session: Optional[str] = Cookie(default=None),
) -> Optional[str]:
    """Who is asking: the key they hold, or the account they signed in to.

    The key wins when both are present. Somebody who has just loaded a key file
    means to act as that key, and a session left over from an earlier sign-in
    should not quietly redirect their work somewhere else.

    A session resolves to a digest the account is linked to -- never to a key,
    because the coordinator does not hold anybody's key and should not start.
    """
    from backend.service import accountService

    by_key = read_submitter_key(x_submitter_key)
    if by_key:
        return by_key

    github_id = accountService.read_session(hw_session)
    if not github_id:
        return None

    return await accountService.primary_submitter_id(Database, github_id)


async def submitter_scope(
    x_submitter_key: Optional[str] = Header(default=None),
    hw_session: Optional[str] = Cookie(default=None),
) -> List[str]:
    """Every digest the caller may read and act on, not just the one they write as.

    Writing needs a single answer -- a job is filed under one submitter_id --
    and submitter_or_session gives it. Reading does not. Somebody who has sent
    work from a desktop and a laptop holds two keys and therefore two digests,
    and an account exists precisely so that those are one person's work rather
    than two strangers'.

    Answering with only the digest they happen to be carrying today is what
    makes an account feel decorative: sign in on the laptop and the desktop's
    jobs are still missing. So the key in the browser and every digest the
    account is linked to are all in scope.

    The key comes first, so a route that wants one id can take scope[0] and get
    the same answer submitter_or_session would have given it.
    """
    from backend.service import accountService

    scope: List[str] = []

    by_key = read_submitter_key(x_submitter_key)
    if by_key:
        scope.append(by_key)

    github_id = accountService.read_session(hw_session)
    if github_id:
        for digest in await accountService.submitter_ids(Database, github_id):
            if digest not in scope:
                scope.append(digest)

    return scope

def require_uploader(
    authorization: Optional[str] = Header(default=None),
    x_submitter_key: Optional[str] = Header(default=None),
) -> str:
    """Whoever is storing a blob: a node reporting weights, or a submitter.

    Uploading used to need nothing at all. Anyone who could reach the
    coordinator could POST half a gigabyte -- MAX_ARTIFACT_BYTES -- as many
    times as they liked, and every one of them landed in GridFS. There was no
    caller to attribute it to and so nothing to rate-limit, count or clean up
    against.

    Both real callers already prove who they are for everything else they do:
    the node sends the bearer token it got from /verify-challenge, and the
    browser holds a submitter key it creates on first use. This asks for the
    one they already have rather than introducing a third kind of credential.
    """
    if authorization and authorization.lower().startswith("bearer "):
        node_id = read_node_token(authorization.split(" ", 1)[1].strip())
        if node_id:
            return f"node:{node_id}"

    submitter = read_submitter_key(x_submitter_key)
    if submitter:
        return f"submitter:{submitter}"

    raise HTTPException(
        status_code=401,
        detail="Uploading needs either a node session token or a submitter key.",
    )

def require_node_token(node_id: str, caller: str = Depends(authenticated_node)) -> str:
    """Require a token issued for this exact node_id."""
    if caller != node_id:
        logger.warning(f"🚫 Token for {caller} was used against node {node_id}.")
        raise HTTPException(status_code=403, detail="Token does not grant access to this node.")
    return caller

task_results = []

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
    # No `ip` here on purpose. Whatever was recorded was the address the
    # connection arrived from, which -- since the setup page points nodes at the
    # dashboard -- was the dashboard's own address for every contributor who
    # followed the instructions. Nothing read it: it was never shown, and
    # nothing dials a node, because nodes poll. A field that is wrong for most
    # rows, unread by anything, and a contributor's home address when it is
    # right is better not collected.
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


# Terminal states: nothing more will happen to a task in one of these.
FINISHED_STATES = ("completed", "failed", "rejected", "cancelled")

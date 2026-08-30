"""The coordinator: the one service both sides of the network talk to.

This file used to be 2,586 lines -- thirty routes, the auth dependencies, the
database wrapper, the request models, the artifact store and the background
loops, all in one place. Simple questions were expensive to answer: working out
which endpoints were authenticated needed a script rather than a glance, and
finding where an uploaded CSV's column names ended up took three passes.

What is left here is the assembly: the app, its middleware, the background
tasks and the routers. The routes themselves live next door, grouped by what
they are about rather than by when they happened to be written.

    routes/deps.py       the database, who is calling, the request shapes
    routes/nodes.py      registering a node, heartbeats, proving identity
    routes/artifacts.py  datasets, holdouts and weights in and out of GridFS
    routes/tasks.py      submitting work, handing it out, taking results back
    routes/modelops.py   what a finished model does: sample, download, score

Nothing about the API changed: the same paths, on the same app, in the same
order. tests/test_proxy_forwarding.py and the coordinator tests both still
address it through this module.
"""

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routes import artifacts as artifacts_routes
from backend.routes import modelops as modelops_routes
from backend.routes import nodes as nodes_routes
from backend.routes import tasks as tasks_routes
from backend.routes.deps import Database

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("coordinator")

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Order matters only for readability; FastAPI matches on the path.
app.include_router(nodes_routes.router)
app.include_router(artifacts_routes.router)
app.include_router(tasks_routes.router)
app.include_router(modelops_routes.router)


@app.on_event("startup")
async def startup_event():
    await Database.connect_db()
    asyncio.create_task(nodes_routes.sync_nodes_with_db())
    asyncio.create_task(nodes_routes.cleanup_expired_challenges())
    asyncio.create_task(tasks_routes.requeue_stale_tasks())
    asyncio.create_task(artifacts_routes.forget_finished_datasets())


@app.on_event("shutdown")
async def shutdown_event():
    await Database.close_db()


# Re-exported so that `import backend.coordinator as coordinator` keeps
# reaching the things it always did. The tests address the coordinator as one
# thing, which is what it is from outside; splitting the file is not a reason
# to make every caller learn the new shape.
from bson import ObjectId                                          # noqa: E402,F401
from motor.motor_asyncio import AsyncIOMotorGridFSBucket           # noqa: E402,F401

from backend.routes.artifacts import (                             # noqa: E402,F401
    download_artifact, prepare_dataset_split, upload_artifact,
)
from backend.routes.deps import (                                  # noqa: E402,F401
    connected_nodes, get_db, require_uploader, system_usage, task_results,
)
from backend.routes.tasks import (                                 # noqa: E402,F401
    _redispatch, public_task, receive_task_result,
)

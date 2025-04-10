import uuid
import time
import numpy as np
from fastapi import Body, FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
import psutil
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
from datetime import datetime, timedelta
import httpx
from backend.utils.config import COORDINATOR_URL
import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

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

            # Create indices for performance
            await cls.nodes_collection.create_index("_id")
            await cls.nodes_collection.create_index("isAvailable")
            await cls.nodes_collection.create_index("isConnected")
            await cls.tasks_collection.create_index("node_id")
            await cls.tasks_collection.create_index("received_at")
            logger.info("✅ Database indices created")

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
    node_id: str = Field(default_factory=lambda: f"node_{uuid.uuid4()}")
    ip: str
    country: Optional[str] = "Unknown"
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
async def toggle_availability(node_id: str, db: Database = Depends(get_db)):
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
        # Validate GPU - required for connection but we won't store details
        gpu_capabilities = node.capabilities.get("gpu", [])
        if not gpu_capabilities or (
            isinstance(gpu_capabilities, list) and
            (len(gpu_capabilities) == 0 or gpu_capabilities[0].get("name") in ["No GPU Detected", None, ""])
        ):
            return {
                "status": "rejected",
                "reason": "No valid GPU detected. Node connection refused."
            }

        # Set node runtime properties
        node.ip = request.client.host
        node.isConnected = True
        node.last_heartbeat = datetime.utcnow()

        # Store only essential data in the database (not dynamic hardware info)
        node_document = {
            "_id": node.node_id,
            "ip": node.ip,
            "country": node.country,
            "isConnected": True,
            "isAvailable": node.isAvailable,
            "last_connected": datetime.utcnow(),
            "last_heartbeat": node.last_heartbeat,
            # Store only whether GPU exists, not detailed capabilities
            "has_gpu": bool(gpu_capabilities and len(gpu_capabilities) > 0 and 
                           gpu_capabilities[0].get("name") not in ["No GPU Detected", None, ""])
        }

        # Capture node_id for in-memory storage
        live_node_id = node.node_id

        # Upsert essential info in database
        await db.nodes_collection.update_one(
            {"_id": live_node_id},
            {"$set": node_document},
            upsert=True
        )

        # Keep full details in memory
        connected_nodes[live_node_id] = node

        # Log the connected node
        logger.info(f"✅ Current connected nodes: {list(connected_nodes.keys())}")

        # Forward node to coordinator (if needed)
        try:
            async with httpx.AsyncClient() as client:
                # Only send essential info to coordinator, not dynamic hardware details
                coordinator_payload = {
                    "node_id": live_node_id,
                    "ip": node.ip,
                    "country": node.country,
                    "isConnected": True,
                    "isAvailable": node.isAvailable,
                    "has_gpu": bool(gpu_capabilities and len(gpu_capabilities) > 0)
                }
                res = await client.post(f"{COORDINATOR_URL}/register-node", json=coordinator_payload)
                res.raise_for_status()
                logger.info(f"✅ Node {live_node_id} registered with coordinator successfully.")
        except Exception as e:
            logger.error(f"⚠️ Failed to register node with coordinator: {e}")

        logger.info(f"🔌 Node connected: {live_node_id}, Available: {node.isAvailable}")

        return {
            "status": "success",
            "message": "Node connected",
            "node_id": live_node_id,
            "ip": node.ip,
        }

    except Exception as e:
        logger.error(f"❌ Error connecting node: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect node: {str(e)}")


@app.post("/node-heartbeat/{node_id}")
async def node_heartbeat(
    node_id: str, 
    status: dict = Body(...), 
    db: Database = Depends(get_db)
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
        
        # Update only essential info in database
        await db.nodes_collection.update_one(
            {"_id": node_id},
            {"$set": {
                "last_heartbeat": node.last_heartbeat,
                "isConnected": True,
            }}
        )
        
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

        return results
    except Exception as e:
        logger.error(f"Error retrieving task results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve task results: {str(e)}")




@app.post("/receive-task-result")
async def receive_task_result(result: dict, db: Database = Depends(get_db)):
    try:
        logger.info(f"Task result received with status: {result.get('status', 'unknown')}")

        result.pop('logs', None)
        
        # Correct node ID handling
        if 'nodeId' in result and 'node_id' not in result:
            result['node_id'] = result.pop('nodeId')  # Rename 'nodeId' to 'node_id'
        
        # Use the preserved node_id as _id if it exists, otherwise generate a new one
        if 'node_id' in result:
            result["_id"] = result["node_id"]
        else:
            result["_id"] = str(uuid.uuid4())
            
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


@app.get("/node-performance/{node_id}")
async def get_node_performance(node_id: str):
    try:
        # Check in-memory state for dynamic performance data
        if node_id in connected_nodes:
            node = connected_nodes[node_id]
            return {
                "status": "success",
                "node_id": node_id,
                "cpu_usage": node.cpu_usage,
                "gpu_usage": node.gpu_usage,
                "cpu_benchmark": node.cpu_benchmark,
                "gpu_benchmark": node.gpu_benchmark,
                "is_connected": node.isConnected,
                "is_available": node.isAvailable,
                "capabilities": node.capabilities
            }

        # If the node is not in memory, return an error
        raise HTTPException(status_code=404, detail=f"Node {node_id} not found in memory")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting node performance for {node_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get node performance: {str(e)}")

        
            


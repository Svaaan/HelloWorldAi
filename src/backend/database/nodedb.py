import motor.motor_asyncio
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo_test:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)

db = client["NodeDbTest"]

nodes_collection = db["nodes"]
tasks_collection = db["tasks"]

async def create_indexes():
    await nodes_collection.create_index("public_key", unique=True)
    await nodes_collection.create_index("isConnected")
    await nodes_collection.create_index("isAvailable")
    await tasks_collection.create_index("node_id")
    await tasks_collection.create_index("received_at")

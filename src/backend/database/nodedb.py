import motor.motor_asyncio
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo_test:27017")
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)

db = client["NodeDbTest"]

nodes_collection = db["nodes"]
tasks_collection = db["tasks"]
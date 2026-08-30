import motor.motor_asyncio
import os

# Which server, and which database on it. Both come from the environment so
# that development and production are separated by configuration rather than by
# hoping nobody points one at the other.
#
# The database name used to be the literal "NodeDbTest", which meant a
# production deployment would have written its real node registrations and job
# history into a database called Test.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://mongo_dev:27017")
MONGO_DB = os.getenv("MONGO_DB", "NodeDbDev")

client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)

db = client[MONGO_DB]

nodes_collection = db["nodes"]
tasks_collection = db["tasks"]

async def create_indexes():
    await nodes_collection.create_index("public_key", unique=True)
    await nodes_collection.create_index("isConnected")
    await nodes_collection.create_index("isAvailable")
    await tasks_collection.create_index("node_id")
    await tasks_collection.create_index("received_at")

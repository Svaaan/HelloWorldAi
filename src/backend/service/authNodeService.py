import uuid
import logging
import httpx
import os
import binascii
from backend.shared.nodeState import node_info, get_gpu_info_list
from backend.service.systemInfoService import get_system_capabilities
from backend.service.connectionService import background_connection_handler
from backend.shared.nodeState import build_node_payload
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from backend.database.nodedb import db
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

# ✅ Check if node is already registered locally
def check_existing_node():
    if node_info.get("node_id"):
        logger.info(f"Node already has local node_id: {node_info['node_id']}")
        return True, "Node already exists, connected"
    return False, None

def validate_gpu():
    detected_gpus = get_gpu_info_list()
    if not detected_gpus or detected_gpus[0].get("name") in ["No GPU Detected", None, ""]:
        logger.warning("No valid GPU detected — connection refused.")
        return False, {"status": "rejected", "reason": "No valid GPU detected. Node connection refused."}
    return True, None

async def trigger_background_connection():
    capabilities = get_system_capabilities()
    payload = build_node_payload(capabilities)

    response = await background_connection_handler(payload, node_info)

    node_id = response.get("node_id")

    if node_id:
        existing_node = await db.nodes.find_one({"_id": node_id})

        if existing_node and not existing_node.get("public_key") and node_info.get("public_key"):
            await db.nodes.update_one(
                {"_id": node_id},
                {"$set": {"public_key": node_info["public_key"]}}
            )
            logger.info(f"✅ Saved public key for node {node_id} to database.")
        else:
            logger.info(f"ℹ️ Node {node_id} already has a public key or no public key provided.")
    else:
        logger.warning("⚠️ Missing node_id, could not save public key to database.")

    return response

def verify_signature(public_key_pem: str, challenge: str, signature_hex: str) -> bool:
    try:
        public_key = serialization.load_pem_public_key(public_key_pem.encode())
        signature_bytes = binascii.unhexlify(signature_hex)

        public_key.verify(
            signature_bytes,
            challenge.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        return True
    except InvalidSignature:
        return False
    except Exception as e:
        logger.error(f"Error verifying signature: {e}")
        return False


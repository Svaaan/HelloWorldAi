import uuid
import logging
import httpx
import os
import binascii
import textwrap
from backend.shared.nodeState import node_info, get_gpu_info_list
from backend.service.systemInfoService import get_system_capabilities
from backend.service.connectionService import background_connection_handler
from backend.shared.nodeState import build_node_payload
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes, serialization
from backend.database.nodedb import db
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

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
    # 🔐 Only reuse node_id if it matches the stored public_key in DB
    if node_info.get("node_id") and node_info.get("public_key"):
        existing = await db.nodes.find_one({
            "_id": node_info["node_id"],
            "public_key": node_info["public_key"]
        })
        if existing:
            logger.info(f"🔁 Reusing registered node ID: {node_info['node_id']}")
            return {"node_id": node_info["node_id"], "status": "already_registered"}
        else:
            logger.warning("⚠️ Node ID exists but public key mismatch — resetting for fresh registration.")
            node_info["node_id"] = None  # Force re-registration

    # 🔎 Check coordinator for public key mapping
    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{os.getenv('COORDINATOR_URL', 'http://coordinator:8100')}/find-node-id",
                json={"public_key": node_info.get("public_key")}
            )

            if res.status_code == 200:
                data = res.json()
                existing_id = data.get("node_id")
                if existing_id:
                    node_info["node_id"] = existing_id
                    logger.info(f"🔁 Found existing node ID from coordinator: {existing_id}")
                    return {"node_id": existing_id, "status": "reused"}
    except Exception as e:
        logger.warning(f"⚠️ Failed to check for existing public key on coordinator: {e}")

    # Proceed as new registration
    capabilities = get_system_capabilities()
    payload = build_node_payload(capabilities)

    response = await background_connection_handler(payload, node_info)
    if not response:
        logger.error("🚨 Coordinator registration failed after all retries.")
        return None

    node_id = response.get("node_id")

    if node_id:
        node_info["node_id"] = node_id

        # Save public key if not already saved
        existing_node = await db.nodes.find_one({"_id": node_id})
        if existing_node and not existing_node.get("public_key") and node_info.get("public_key"):
            await db.nodes.update_one(
                {"_id": node_id},
                {"$set": {"public_key": node_info["public_key"]}}
            )
            logger.info(f"✅ Saved public key for node {node_id} to database.")
    else:
        logger.warning("🚨 Missing node_id in background connection response.")

    return response


def convert_base64_to_pem(base64_key: str) -> str:
    wrapped = "\n".join(textwrap.wrap(base64_key, 64))
    return f"-----BEGIN PUBLIC KEY-----\n{wrapped}\n-----END PUBLIC KEY-----\n"

# ✅ ECDSA signature verification using converted PEM
def verify_signature(public_key_base64: str, challenge: str, signature_hex: str) -> bool:
    try:
        pem_key = convert_base64_to_pem(public_key_base64)
        public_key = serialization.load_pem_public_key(pem_key.encode())

        signature_bytes = binascii.unhexlify(signature_hex)

        # ✅ Handle raw r|s signature (from browser)
        if len(signature_bytes) == 64:
            r = int.from_bytes(signature_bytes[:32], byteorder="big")
            s = int.from_bytes(signature_bytes[32:], byteorder="big")
            der_signature = encode_dss_signature(r, s)
        else:
            # Already DER format (backend use case)
            der_signature = signature_bytes

        public_key.verify(
            der_signature,
            challenge.encode(),
            ec.ECDSA(hashes.SHA256())
        )
        return True

    except InvalidSignature:
        return False
    except Exception as e:
        logger.error(f"Error verifying signature: {e}")
        return False
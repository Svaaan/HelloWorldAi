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
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from cryptography.exceptions import InvalidSignature

logger = logging.getLogger(__name__)

COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://coordinator:8100")


async def find_node_id_by_public_key(public_key: str):
    """Ask the coordinator which node_id a public key belongs to, or None.

    The node deliberately has no database access: a contributor runs it on their
    own machine, and reaching the coordinator's MongoDB from there would mean
    exposing the database to the internet.
    """
    if not public_key:
        return None

    try:
        async with httpx.AsyncClient() as client:
            res = await client.post(
                f"{COORDINATOR_URL}/find-node-id",
                json={"public_key": public_key},
                timeout=10,
            )
    except Exception as e:
        logger.warning(f"Could not reach the coordinator to resolve the public key: {e}")
        return None

    if res.status_code == 404:
        return None
    if res.status_code != 200:
        logger.warning(f"find-node-id returned {res.status_code}: {res.text[:200]}")
        return None

    return (res.json() or {}).get("node_id")

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
    # 🔎 The coordinator owns the public_key -> node_id mapping.
    existing_id = await find_node_id_by_public_key(node_info.get("public_key"))
    if existing_id:
        if node_info.get("node_id") and node_info["node_id"] != existing_id:
            logger.warning("⚠️ Local node ID did not match the coordinator's — using the coordinator's.")
        node_info["node_id"] = existing_id
        logger.info(f"🔁 Reusing registered node ID: {existing_id}")
        return {"node_id": existing_id, "status": "already_registered"}

    # This key is unknown to the coordinator, so any local node_id is stale.
    node_info["node_id"] = None

    # Proceed as new registration
    capabilities = get_system_capabilities()
    payload = build_node_payload(capabilities)

    response = await background_connection_handler(payload, node_info)
    if not response:
        logger.error("🚨 Coordinator registration failed after all retries.")
        return None

    node_id = response.get("node_id")

    if node_id:
        # The coordinator stores the public key when it registers the node.
        node_info["node_id"] = node_id
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
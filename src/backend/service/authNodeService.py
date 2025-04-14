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

# 1️⃣ Check if node is already registered locally (node_info state)
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


# ✅ Patch your node.py trigger function!
async def trigger_background_connection():
    capabilities = get_system_capabilities()
    payload = build_node_payload(capabilities)

    response = await background_connection_handler(payload, node_info)

    # ✅ After we get the response and node_id from coordinator
    node_id = response.get("node_id")

    if node_id and node_info.get("public_key"):
        db.nodes.update_one(
            {"_id": node_id},
            {"$set": {"public_key": node_info["public_key"]}}
        )
        logger.info(f"✅ Saved public key for node {node_id} to database.")
    else:
        logger.warning("⚠️ Missing node_id or public_key, could not save to database.")

    return response


def generate_keypair():
    private_key = ec.generate_private_key(ec.SECP256R1())
    public_key = private_key.public_key()

    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )

    public_bytes = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # Save keys to local files
    key_dir = os.path.join(os.getcwd(), "keys")
    os.makedirs(key_dir, exist_ok=True)

    private_key_path = os.path.join(key_dir, "node_private_key.pem")
    public_key_path = os.path.join(key_dir, "node_public_key.pem")

    with open(private_key_path, "w") as f:
        f.write(private_bytes.decode())

    with open(public_key_path, "w") as f:
        f.write(public_bytes.decode())

    logger.info(f" Private key saved to: {private_key_path}")
    logger.info(f"Public key saved to: {public_key_path}")

    return private_bytes.decode(), public_bytes.decode()


def sign_challenge(challenge: str) -> str:
    key_dir = os.path.join(os.getcwd(), "keys")
    private_key_path = os.path.join(key_dir, "node_private_key.pem")

    # Load private key
    with open(private_key_path, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None
        )

    # Sign challenge
    signature = private_key.sign(
        challenge.encode(),
        ec.ECDSA(hashes.SHA256())
    )

    # Convert to hex string for transport
    return signature.hex()

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

async def automatic_node_verification(node_id):
    coordinator_url = os.getenv('COORDINATOR_URL', 'http://127.0.0.1:8100')

    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Request challenge
            challenge_response = await client.get(f"{coordinator_url}/generate-challenge/{node_id}", timeout=10)
            challenge_response.raise_for_status()

            challenge = challenge_response.json().get("challenge")
            if not challenge:
                logger.error("No challenge received from coordinator.")
                return

            # Step 2: Sign challenge
            signature = sign_challenge(challenge)

            # Step 3: Verify challenge
            verify_response = await client.post(
                f"{coordinator_url}/verify-challenge/{node_id}",
                json={"signature": signature},
                timeout=10
            )
            verify_response.raise_for_status()


            data = verify_response.json()
            if data.get("status") == "success":
                logger.info(f"🎉 Node {node_id} verified successfully with automatic flow.")
            else:
                logger.warning(f"⚠️ Verification failed: {data.get('message')}")

    except httpx.RequestError as e:
        logger.error(f"Error during automatic verification: {e}")
    except Exception as e:
        logger.error(f"Unexpected error during node verification: {e}")
"""Signed session tokens handed to a node once it proves ownership of its keypair.

The coordinator is the only issuer and the only verifier. A node process and the
browser both treat the token as opaque and just pass it back as a bearer token.
"""

import os
import secrets
import logging
from typing import Optional

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

logger = logging.getLogger(__name__)

TOKEN_SALT = "node-session"
NODE_TOKEN_TTL = int(os.getenv("NODE_TOKEN_TTL", 24 * 60 * 60))  # seconds

_SECRET = os.getenv("NODE_TOKEN_SECRET")
if not _SECRET:
    _SECRET = secrets.token_urlsafe(32)
    logger.warning(
        "⚠️ NODE_TOKEN_SECRET is not set — generated an ephemeral signing key. "
        "Every node session will be invalidated when this process restarts, and "
        "tokens will not validate across multiple coordinator workers. "
        "Set NODE_TOKEN_SECRET in the environment outside of local development."
    )

_serializer = URLSafeTimedSerializer(secret_key=_SECRET, salt=TOKEN_SALT)


def issue_node_token(node_id: str) -> str:
    """Mint a session token binding the caller to a single node_id."""
    return _serializer.dumps({"node_id": node_id})


def read_node_token(token: str) -> Optional[str]:
    """Return the node_id a token is good for, or None if it is invalid/expired."""
    try:
        payload = _serializer.loads(token, max_age=NODE_TOKEN_TTL)
    except SignatureExpired:
        logger.info("Rejected an expired node session token.")
        return None
    except BadSignature:
        logger.warning("Rejected a node session token with a bad signature.")
        return None
    except Exception as e:
        logger.error(f"Error reading node session token: {e}")
        return None

    if not isinstance(payload, dict):
        return None
    return payload.get("node_id")

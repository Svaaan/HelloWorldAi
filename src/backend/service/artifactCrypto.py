"""Encrypting stored artifacts at rest.

What this protects against, and what it does not
------------------------------------------------
Submitted datasets sat in GridFS as plain numbers. Anyone who could read the
database -- a backup, a dump, a compromised host, an operator browsing
collections -- could read every submitter's training data. This closes that:
blobs are encrypted before they are stored and decrypted only when served to a
party entitled to them.

It does **not** hide data from the contributor running the job. It cannot. A
GPU has to see the numbers to train on them, and the alternatives that would
avoid that -- homomorphic encryption, secure enclaves, multi-party computation
-- are either thousands of times too slow for training or need hardware
consumer graphics cards do not have. The honest position is that a node
operator can read the training data they are sent, and the product should say
so rather than imply otherwise.

So this is defence against the coordinator's own storage being read, which is a
real and separate risk, not a claim of privacy from the node.

Key handling
------------
AES-256-GCM, which authenticates as well as encrypts: a tampered blob fails to
decrypt rather than silently returning altered training data. The key comes
from ARTIFACT_ENCRYPTION_KEY and is never written to the database it protects.

With no key set, blobs are stored as they always were. That keeps development
working and existing deployments readable, but it is logged loudly once,
because a silent downgrade to plaintext is exactly the kind of thing that goes
unnoticed for a year.
"""

import base64
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# Marks a blob this module wrote. Anything without it is treated as legacy
# plaintext and passed through, so turning encryption on does not strand data
# that is already stored.
MAGIC = b"HWAIENC1"
NONCE_BYTES = 12

_warned = False


def _key() -> Optional[bytes]:
    """The configured key, or None when encryption is switched off."""
    raw = os.getenv("ARTIFACT_ENCRYPTION_KEY", "").strip()
    if not raw:
        return None

    # Accept base64 or hex so operators can paste whichever their tooling gives.
    for decode in (base64.urlsafe_b64decode, base64.b64decode, bytes.fromhex):
        try:
            key = decode(raw)
            if len(key) in (16, 24, 32):
                return key
        except Exception:
            continue

    logger.error(
        "ARTIFACT_ENCRYPTION_KEY is set but is not a 16, 24 or 32 byte "
        "base64 or hex value; storing artifacts unencrypted."
    )
    return None


def _cipher(key: bytes):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    return AESGCM(key)


def is_enabled() -> bool:
    return _key() is not None


def _warn_once():
    global _warned
    if not _warned:
        logger.warning(
            "ARTIFACT_ENCRYPTION_KEY is not set: submitted datasets are stored "
            "unencrypted and are readable by anyone who can read the database."
        )
        _warned = True


def encrypt(payload: bytes) -> bytes:
    """Encrypt a blob for storage, or return it unchanged if no key is set."""
    key = _key()
    if key is None:
        _warn_once()
        return payload

    nonce = os.urandom(NONCE_BYTES)
    ciphertext = _cipher(key).encrypt(nonce, bytes(payload), None)
    return MAGIC + nonce + ciphertext


def decrypt(blob: bytes) -> bytes:
    """Decrypt a stored blob.

    Blobs written before encryption was switched on carry no marker and are
    returned as they are, so enabling a key does not orphan existing data.
    """
    blob = bytes(blob)

    if not blob.startswith(MAGIC):
        return blob

    key = _key()
    if key is None:
        # The data is encrypted but the key is gone. Failing loudly is the only
        # honest option: returning the ciphertext would look like a corrupt
        # dataset and be debugged as one.
        raise RuntimeError(
            "This artifact is encrypted but ARTIFACT_ENCRYPTION_KEY is not set."
        )

    nonce = blob[len(MAGIC):len(MAGIC) + NONCE_BYTES]
    ciphertext = blob[len(MAGIC) + NONCE_BYTES:]

    from cryptography.exceptions import InvalidTag
    try:
        return _cipher(key).decrypt(nonce, ciphertext, None)
    except InvalidTag as e:
        raise RuntimeError(
            "This artifact could not be decrypted: wrong key, or the stored "
            "bytes have been altered."
        ) from e

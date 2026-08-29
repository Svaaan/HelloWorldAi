"""Who submitted a job, without making anybody create an account.

The problem
-----------
A node proves who it is with a keypair (see authNodeService). The other side --
the person with the data -- had no identity at all. A task recorded only the IP
it arrived from, so there was no way to ask "which jobs are mine", and no way to
decide who may download a trained model. The result was a pipeline that trained
a model and then had nowhere to hand it back to.

The approach
------------
The browser generates a random secret once and keeps it. Every request carries
it in a header; the coordinator stores only its SHA-256 digest on the task.

    submitter_id = sha256(key)

That gives ownership without accounts, passwords or email, and it means the
database never holds anything that would let an attacker who reads it claim
someone's jobs -- the digest is not usable as a credential.

The secret is a bearer credential: whoever holds it owns those jobs. It travels
in a header rather than a query string so it stays out of access logs, browser
history and Referer headers. Losing it means losing access to those jobs, which
is the honest trade for having no account to recover.
"""

import hashlib
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# 256 bits of randomness, hex encoded. Anything shorter is worth rejecting
# rather than quietly accepting a weak key.
MIN_KEY_CHARS = 32
MAX_KEY_CHARS = 256

_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


class SubmitterKeyError(ValueError):
    """The supplied key is not something we will accept."""


def submitter_id_from_key(key: str) -> str:
    """The stored identifier for a submitter secret.

    One-way: the digest is what lands in the database, so reading the database
    does not give anyone the ability to act as that submitter.
    """
    key = (key or "").strip()

    if len(key) < MIN_KEY_CHARS:
        raise SubmitterKeyError(
            f"Submitter key must be at least {MIN_KEY_CHARS} characters."
        )
    if len(key) > MAX_KEY_CHARS:
        raise SubmitterKeyError("Submitter key is too long.")
    if not _KEY_PATTERN.match(key):
        raise SubmitterKeyError("Submitter key contains unexpected characters.")

    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def read_submitter_key(header_value: Optional[str]) -> Optional[str]:
    """Turn a header into a submitter id, or None if there is no usable key.

    Returns None rather than raising for a missing header: submitting without
    a key still works, it just produces a job nobody can later claim.
    """
    if not header_value:
        return None

    try:
        return submitter_id_from_key(header_value)
    except SubmitterKeyError as e:
        logger.warning(f"Ignoring an unusable submitter key: {e}")
        return None

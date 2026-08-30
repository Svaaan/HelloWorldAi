"""How much one submitter may ask of the network.

Why this exists
---------------
A submitter key is minted by the browser on first use. Nobody approves it, it
costs nothing, and a person can have as many as they care to generate. That was
harmless while the coordinator was unreachable. The moment it has a public
address, the arithmetic is unpleasant: each job may carry a dataset of up to
MAX_ARTIFACT_BYTES (512 MB), nothing limits how many jobs are queued, and every
one of them spends a stranger's electricity and a stranger's graphics card.

Gating uploads behind a key stopped anonymous abuse. It did not stop cheap
abuse, because the key is free. What follows is the other half: a key is now
something you can exhaust.

What is counted, and what is not
--------------------------------
Submitters are limited. Nodes are not: a node uploads weights only for a job it
was given, so its usage is already bounded by work the coordinator handed it,
and throttling it would break a job that is already half done.

Counting is done from the collections that already exist rather than from a
separate ledger. There is no cache to go stale, and a restart cannot forget how
much somebody has already used. It costs an indexed count per submission, which
is nothing next to training a model.

The numbers are deliberately generous. This is a brake for the pathological
case, not a business model; the limits should be invisible to somebody using
the service as intended.
"""

import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger("quota")

# Bytes one submitter may store per rolling day. Two gigabytes is several large
# datasets, and far more than the browser upload path is comfortable with.
UPLOAD_BYTES_PER_DAY = int(os.getenv("QUOTA_UPLOAD_BYTES_PER_DAY", 2 * 1024 ** 3))

# Jobs one submitter may queue per rolling hour.
JOBS_PER_HOUR = int(os.getenv("QUOTA_JOBS_PER_HOUR", 30))

# Jobs one submitter may have unfinished at once. This is the one that protects
# contributors: without it a single person can occupy every machine on the
# network at the same moment.
ACTIVE_JOBS = int(os.getenv("QUOTA_ACTIVE_JOBS", 5))

# Everything is off when this is set, for a private deployment where the only
# submitters are people you know.
ENABLED = os.getenv("QUOTA_ENABLED", "true").lower() not in ("false", "0", "no")

FINISHED = ("completed", "failed", "rejected", "cancelled")


class QuotaExceeded(Exception):
    """Raised with a sentence meant to be shown to the person who hit it."""


def _is_submitter(caller: str) -> bool:
    """require_uploader returns 'submitter:<id>' or 'node:<id>'."""
    return bool(caller) and not str(caller).startswith("node:")


def _identity(caller: str) -> str:
    return str(caller).split(":", 1)[-1] if caller else ""


async def bytes_used_today(db, uploader: str) -> int:
    """Total stored for this uploader in the last day."""
    cutoff = datetime.utcnow() - timedelta(days=1)
    cursor = db.db["artifacts.files"].aggregate([
        {"$match": {"metadata.uploader": uploader,
                    "metadata.uploaded_at": {"$gte": cutoff}}},
        {"$group": {"_id": None, "total": {"$sum": "$metadata.bytes"}}},
    ])
    async for row in cursor:
        return int(row.get("total") or 0)
    return 0


async def check_upload(db, caller: str, incoming: int) -> None:
    """Refuse an upload that would put this submitter over the daily limit."""
    if not ENABLED or not _is_submitter(caller):
        return

    used = await bytes_used_today(db, caller)
    if used + incoming > UPLOAD_BYTES_PER_DAY:
        allowed_mb = UPLOAD_BYTES_PER_DAY / 1024 ** 2
        used_mb = used / 1024 ** 2
        logger.warning(
            f"Upload refused for {caller}: {used} + {incoming} bytes "
            f"exceeds the {UPLOAD_BYTES_PER_DAY} daily limit."
        )
        raise QuotaExceeded(
            f"That would put you over the daily upload limit of "
            f"{allowed_mb:,.0f} MB. You have stored {used_mb:,.0f} MB in the "
            f"last 24 hours. Older uploads stop counting as the day rolls "
            f"forward, or you can delete finished jobs."
        )


async def check_new_job(db, submitter: str) -> None:
    """Refuse a job that would exceed the hourly rate or the active limit."""
    if not ENABLED or not submitter:
        return

    active = await db.tasks_collection.count_documents(
        {"submitter_id": submitter, "status": {"$nin": list(FINISHED)}}
    )
    if active >= ACTIVE_JOBS:
        raise QuotaExceeded(
            f"You already have {active} jobs running or waiting, which is the "
            f"limit of {ACTIVE_JOBS}. Wait for one to finish, or cancel one, "
            f"before sending another. The limit exists so that one person "
            f"cannot occupy every machine on the network at once."
        )

    cutoff = datetime.utcnow() - timedelta(hours=1)
    recent = await db.tasks_collection.count_documents(
        {"submitter_id": submitter, "submitted_at": {"$gte": cutoff}}
    )
    if recent >= JOBS_PER_HOUR:
        raise QuotaExceeded(
            f"You have sent {recent} jobs in the last hour, which is the "
            f"limit of {JOBS_PER_HOUR}. Try again shortly."
        )


async def usage(db, submitter: str) -> dict:
    """What this submitter has used, for showing them before they hit a wall."""
    if not submitter:
        return {}

    cutoff = datetime.utcnow() - timedelta(hours=1)
    return {
        "enabled": ENABLED,
        "bytes_used_today": await bytes_used_today(db, submitter),
        "bytes_per_day": UPLOAD_BYTES_PER_DAY,
        "active_jobs": await db.tasks_collection.count_documents(
            {"submitter_id": submitter, "status": {"$nin": list(FINISHED)}}),
        "active_jobs_limit": ACTIVE_JOBS,
        "jobs_last_hour": await db.tasks_collection.count_documents(
            {"submitter_id": submitter, "submitted_at": {"$gte": cutoff}}),
        "jobs_per_hour": JOBS_PER_HOUR,
    }

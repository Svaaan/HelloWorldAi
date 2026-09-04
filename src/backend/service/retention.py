"""How long a finished job's data is kept, and whose it is.

The data behind a job used to be deleted a flat sixty minutes after it
finished, for everybody. That number was chosen for the two things that happen
immediately -- verification reads the holdout, and a retry reuses the same split
-- and it is right for those.

It is wrong for the thing people actually do. "Adjust and run" exists precisely
so a dataset can be trained again with different settings, reusing the same
withheld rows so the runs are comparable. Come back the next morning with
another idea and the answer is:

    The data for this job was deleted 60 minutes after it finished, so it
    cannot be run again. Send it as a new job with the file.

So they send the file again. And again. The loop was built; an hour-long timer
was quietly undoing it.

What changes
------------
Retention follows the owner rather than the clock alone. Somebody signed in has
an account, a workspace to manage this from, and is reachable -- their data is
kept long enough to come back to. An anonymous submitter keeps the old hour,
because there is nothing to come back to: the key that owns it may already be
gone from the browser that made it.

Neither is unbounded. Storage is real, and an account that keeps everything
forever is a bill rather than a feature, so an owner keeps data for their most
recent jobs and older ones are let go early even inside the long window. That
caps what one person can hold without counting bytes: the oldest thing goes
first, which is also what somebody would choose.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

# The old flat value, now the anonymous one. Long enough for verification and
# an immediate retry, which is all it was ever tuned for.
SHORT_MINUTES = int(os.getenv("DATASET_RETENTION_MINUTES", 60))

# A week for somebody signed in. Long enough to come back after a weekend;
# short enough that abandoned data does not sit forever.
LONG_MINUTES = int(os.getenv("DATASET_RETENTION_ACCOUNT_MINUTES", 7 * 24 * 60))

# How many of an owner's finished jobs hold onto their data at once. Beyond
# this the oldest are forgotten early, whatever the clock says.
KEEP_PER_OWNER = int(os.getenv("DATASET_KEEP_PER_OWNER", 10))

# The most retained dataset and holdout bytes this deployment will hold, across
# everybody.
#
# The per-owner cap bounds one person and bounds nothing in total: five hundred
# people each keeping ten datasets is five thousand datasets, and the machine
# this runs on has a 38 GB disk. Before retention followed the account, an hour
# made that impossible to reach; a week makes it a matter of how many people
# turn up.
#
# So there is a ceiling, and reaching it shortens everyone's window rather than
# filling the disk. A service that keeps data for a week and then falls over is
# worse than one that says a day and means it.
#
# Weights are not counted. The sweep never deletes them -- they are the thing
# somebody came for -- so counting them here would let models crowd out the
# datasets and then find nothing it was willing to drop.
STORAGE_CEILING_BYTES = int(
    os.getenv("DATASET_STORAGE_CEILING_BYTES", 8 * 1024 ** 3))

# Clear down to this fraction of the ceiling rather than just under it, so the
# next upload does not immediately trip it again and start deleting on every
# pass.
RECLAIM_TO = float(os.getenv("DATASET_RECLAIM_TO", 0.9))


def window(has_account: bool) -> timedelta:
    """How long this owner's data is kept after a job finishes."""
    return timedelta(minutes=LONG_MINUTES if has_account else SHORT_MINUTES)


def expires_at(finished_at, has_account: bool):
    """When the data behind a finished job is dropped, or None if unknown.

    Returned to the page as well as used here, so a countdown can be shown
    rather than a deletion being discovered by trying to use it.

    Marked as UTC, which is not decoration. Timestamps in this database are
    written with utcnow() and so carry no zone, and a naive datetime serialises
    without an offset -- which JavaScript then reads as *local* time. Two hours
    east of UTC that turned "40 minutes left" into "two hours ago" and the
    countdown rendered nothing at all. The week-long one survived only because
    the error was smaller than the window.
    """
    if not finished_at:
        return None

    due = finished_at + window(has_account)
    return due if due.tzinfo else due.replace(tzinfo=timezone.utc)


def should_forget(finished_at, now: datetime, *, has_account: bool,
                  rank: int) -> bool:
    """Whether this job's data goes now.

    `rank` is its position among the owner's finished jobs that still hold
    data, newest first -- 0 is the most recent. Past KEEP_PER_OWNER the data
    goes early: one person keeping an unbounded pile is the failure this
    prevents, and dropping their oldest is what they would pick themselves.
    """
    if finished_at is None:
        # Nothing to measure from. Treated as expired rather than immortal:
        # a row with no finished_at and a dataset attached is a leak.
        return True

    if rank >= KEEP_PER_OWNER:
        return True

    return now >= finished_at + window(has_account)


def describe(has_account: bool) -> str:
    """One line for the page, in hours or days rather than minutes."""
    minutes = LONG_MINUTES if has_account else SHORT_MINUTES

    if minutes >= 2 * 24 * 60:
        return f"{minutes // (24 * 60)} days"
    if minutes >= 120:
        return f"{minutes // 60} hours"
    return f"{minutes} minutes"


# --- the global ceiling -----------------------------------------------------

def over_ceiling(total_bytes: int) -> bool:
    """Whether retained data has grown past what this deployment will hold."""
    return STORAGE_CEILING_BYTES > 0 and total_bytes > STORAGE_CEILING_BYTES


def bytes_to_reclaim(total_bytes: int) -> int:
    """How much to free, or zero when there is room.

    Down to RECLAIM_TO of the ceiling rather than to the ceiling itself: coming
    to rest exactly on the line means the next upload crosses it again and
    every pass from then on deletes something.
    """
    if not over_ceiling(total_bytes):
        return 0

    target = int(STORAGE_CEILING_BYTES * RECLAIM_TO)
    return max(0, total_bytes - target)

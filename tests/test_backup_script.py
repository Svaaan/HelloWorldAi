"""backup.sh has to capture both halves, and land them somewhere safe.

Everything worth keeping lives on one machine: one server, one MongoDB, one
copy of an artifact encryption key that cannot be reissued. There was no backup
of any of it.

Two properties make this a backup rather than a file copy, and both are easy to
lose in an edit:

  * it takes the database *and* the key. Submitted datasets are encrypted with
    ARTIFACT_ENCRYPTION_KEY, which exists only in env/.env.production, and
    deploy.sh excludes env/.env.* from the sync in both directions -- so that
    file exists on exactly one disk in the world until this script runs.
    A database restored without it is a list of jobs and a pile of bytes nobody
    can read.

  * it does not write into the repository. The output holds the production
    secrets in the clear; inside a git working tree it is one `git add -A` from
    being published.

These read the script rather than running it -- a real run needs the server --
so they check for the shapes those two properties depend on.
"""

import os
import re
import stat

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BACKUP = os.path.join(ROOT, "backup.sh")


@pytest.fixture(scope="module")
def script():
    with open(BACKUP, encoding="utf-8") as fh:
        return fh.read()


def test_it_takes_the_database(script):
    assert "mongodump" in script, "no database dump"
    assert "--archive" in script, (
        "dump to an archive on stdout; a directory dump leaves files on the "
        "server to clean up and does not stream back over ssh"
    )
    assert "--gzip" in script, "the stored artifacts are the bulk of it"


def test_it_takes_the_key_that_makes_the_database_readable(script):
    """The half most easily forgotten, because it is one small file."""
    assert ".env.production" in script, (
        "backup.sh does not fetch env/.env.production. Without it the datasets "
        "in the archive cannot be decrypted, and nothing can reissue the key."
    )


def test_a_dump_that_is_obviously_empty_fails(script):
    """mongodump exits 0 for an empty database as happily as for a full one.

    That is the failure worth catching: a nightly backup that has been running
    for months and capturing nothing.
    """
    assert re.search(r"wc -c\s*<", script), "the archive size is never checked"
    assert re.search(r"-lt\s+1024", script), (
        "nothing rejects a dump too small to be a database"
    )


def test_it_refuses_to_write_into_the_repository(script):
    """The output is the production secrets in the clear."""
    assert "Refusing to write backups inside the repository" in script, (
        "no guard against BACKUP_ROOT pointing inside the working tree"
    )
    # Defaulting outside the repo is the other half: the guard only fires when
    # somebody points it at the repo deliberately.
    assert re.search(r'BACKUP_ROOT:-\$HOME', script), (
        "BACKUP_ROOT should default outside the repository"
    )


def test_the_files_are_not_world_readable(script):
    assert "chmod 700" in script and "chmod 600" in script, (
        "a directory holding NODE_TOKEN_SECRET and ARTIFACT_ENCRYPTION_KEY "
        "should not be readable by every account on the machine"
    )


def test_the_host_has_no_default(script):
    """Same reason deploy.sh requires it: backing up the wrong box silently."""
    assert re.search(r'HOST="\$\{HOST:\?', script), (
        "HOST must be required, not defaulted to whatever was written down"
    )


def test_a_restore_can_be_checked(script):
    """An untested backup is a belief.

    --verify restores into a throwaway MongoDB and counts documents, which is
    the only evidence that the archive is worth anything.
    """
    assert "--verify" in script, "no way to check the archive restores"
    assert "mongorestore" in script, "--verify does not actually restore"
    assert "countDocuments" in script, (
        "a restore that is not counted proves only that mongorestore ran"
    )


def test_the_restore_route_is_written_down(script):
    """Recovering is the one procedure nobody rehearses before needing it."""
    with open(os.path.join(ROOT, "DEPLOY.md"), encoding="utf-8") as fh:
        deploy = fh.read()

    assert "mongorestore" in deploy, "DEPLOY.md never says how to restore"
    assert "env.production" in deploy, (
        "the restore instructions skip putting the key back, which leaves the "
        "restored datasets unreadable"
    )


def test_the_docs_invoke_it_the_way_it_can_actually_be_run():
    """This repository does not record executable bits.

    deploy.sh is tracked 100644, so a fresh Unix checkout gets a file that
    `./deploy.sh` cannot execute. Nothing here is going to fix that for every
    script at once, but the instructions for this one should work as written on
    the machine somebody is standing at when they need a backup -- which is
    generally a day when something has already gone wrong.
    """
    with open(os.path.join(ROOT, "DEPLOY.md"), encoding="utf-8") as fh:
        deploy = fh.read()

    mentions = [line for line in deploy.splitlines() if "backup.sh" in line]
    assert mentions, "DEPLOY.md never shows how to run backup.sh"

    runnable = [line for line in mentions if "bash" in line or "cron" in line
                or not line.strip().startswith(("HOST", "./", "0 "))]
    assert runnable, (
        "every invocation of backup.sh in DEPLOY.md relies on an executable bit "
        "this repository does not track; write them as `bash ./backup.sh`"
    )

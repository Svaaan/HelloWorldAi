#!/bin/bash
#
# Take a backup of production, off the machine that is production.
#
#   HOST=1.2.3.4 KEY=~/.ssh/hw.key ./backup.sh
#   HOST=1.2.3.4 KEY=~/.ssh/hw.key ./backup.sh --verify
#
# Two things come back, into one dated directory:
#
#   database.archive.gz   every collection, GridFS included -- the datasets
#                         people submitted and the models that came back
#   env.production        the server's secrets
#
# They are kept together because either one alone is useless. The datasets in
# that archive are encrypted with ARTIFACT_ENCRYPTION_KEY, which lives only in
# that env file and cannot be regenerated: restore the database without it and
# you have a list of jobs and a pile of bytes nobody can read.
#
# That also means this directory is exactly as sensitive as the server. It is
# written mode 700 and it must not go anywhere the server's secrets should not
# -- above all, not into this repository.
#
# --verify restores the archive into a throwaway MongoDB and counts what came
# out. An untested backup is a belief, not a backup, and the failure mode is
# silent: mongodump exits 0 for an empty database just as happily as for a full
# one.

set -euo pipefail

HOST="${HOST:?set HOST to the server address, e.g. HOST=1.2.3.4 ./backup.sh}"
USER_NAME="${REMOTE_USER:-ubuntu}"
KEY="${KEY:-$HOME/.ssh/id_rsa}"
REMOTE_PATH="${REMOTE_PATH:-~/HelloWorldAi}"
MONGO_CONTAINER="${MONGO_CONTAINER:-mongo_prod}"
MONGO_DB="${MONGO_DB:-NodeDbProd}"

# Outside the repository by default. A backup holding the production secrets
# inside a git working tree is one `git add -A` away from being published.
BACKUP_ROOT="${BACKUP_ROOT:-$HOME/helloworldai-backups}"

VERIFY=0
for arg in "$@"; do
  case "$arg" in
    --verify) VERIFY=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# Refuse to write inside the repo even if someone points BACKUP_ROOT at it.
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
case "$(cd "$BACKUP_ROOT" 2>/dev/null && pwd || echo "$BACKUP_ROOT")" in
  "$REPO_DIR"|"$REPO_DIR"/*)
    echo "Refusing to write backups inside the repository."
    echo "These files hold the production secrets. Set BACKUP_ROOT elsewhere."
    exit 1
    ;;
esac

STAMP="$(date -u +%Y-%m-%dT%H%M%SZ)"
DEST="$BACKUP_ROOT/$STAMP"

mkdir -p "$DEST"
chmod 700 "$BACKUP_ROOT" "$DEST"

echo "Backing up $USER_NAME@$HOST -> $DEST"

# === the database ===
#
# --archive to stdout and straight down the ssh pipe, so nothing large is left
# on the server and no temporary file has to be cleaned up. --gzip because the
# stored artifacts are the bulk of it.
echo "Dumping $MONGO_DB..."
ssh -i "$KEY" "$USER_NAME@$HOST" \
  "docker exec $MONGO_CONTAINER mongodump --db=$MONGO_DB --archive --gzip --quiet" \
  > "$DEST/database.archive.gz"

DB_BYTES=$(wc -c < "$DEST/database.archive.gz")
if [ "$DB_BYTES" -lt 1024 ]; then
  echo "The dump is $DB_BYTES bytes, which is not a database."
  echo "Check that $MONGO_CONTAINER is running and $MONGO_DB is the right name."
  exit 1
fi

# === the secrets ===
#
# deploy.sh excludes env/.env.* from the sync in both directions, so this file
# exists on exactly one machine until this line runs.
echo "Fetching env/.env.production..."
scp -q -i "$KEY" "$USER_NAME@$HOST:$REMOTE_PATH/env/.env.production" \
  "$DEST/env.production"

chmod 600 "$DEST"/*

# Record what this is, so a directory found in a year explains itself.
{
  echo "HelloWorldAi backup"
  echo "taken:     $STAMP"
  echo "host:      $USER_NAME@$HOST"
  echo "database:  $MONGO_DB"
  echo
  echo "database.archive.gz  mongorestore --archive --gzip < database.archive.gz"
  echo "env.production       the key that makes the datasets in it readable"
  echo
  echo "Neither file is any use without the other. Restore instructions are in"
  echo "DEPLOY.md, under 'Getting it back'."
} > "$DEST/README.txt"
chmod 600 "$DEST/README.txt"

echo "  database.archive.gz  $(du -h "$DEST/database.archive.gz" | cut -f1)"
echo "  env.production       $(wc -l < "$DEST/env.production") lines"

# === proving it restores ===
if [ "$VERIFY" = "1" ]; then
  echo
  echo "Verifying: restoring into a throwaway MongoDB..."

  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is not available here, so the restore cannot be checked."
    exit 1
  fi

  NAME="hwai-restore-check-$$"
  docker run -d --rm --name "$NAME" mongo:7 >/dev/null

  # The container needs a moment before it accepts connections.
  for _ in $(seq 1 30); do
    if docker exec "$NAME" mongosh --quiet --eval "db.runCommand({ping:1})" \
        >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done

  docker exec -i "$NAME" mongorestore --archive --gzip --quiet < "$DEST/database.archive.gz"

  echo "Restored:"
  docker exec "$NAME" mongosh "$MONGO_DB" --quiet --eval '
    ["nodes", "tasks", "artifacts.files"].forEach(function (name) {
      print("  " + name + ": " + db.getCollection(name).countDocuments({}));
    });
  '

  docker rm -f "$NAME" >/dev/null
  echo "Restore succeeded."
fi

echo
echo "Done: $DEST"
echo "Keep this somewhere other than the server. It is the only copy of the"
echo "artifact encryption key, and there is no way to reissue it."

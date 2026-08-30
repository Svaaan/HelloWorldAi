#!/bin/bash
#
# Push this working copy to a server and restart the stack.
#
#   HOST=1.2.3.4 KEY=~/.ssh/hw.key ./deploy.sh
#
# Nothing is hardcoded any more. The previous version carried one particular
# Oracle VM's address, one particular Windows path to a key, and one particular
# home directory, which meant it worked on one machine for one server and
# quietly pointed at a box that had been retired.

set -euo pipefail

# === CONFIGURATION ===
# All overridable; HOST has no default on purpose. Deploying to whatever
# address happened to be written down a year ago is how you end up updating a
# server you thought was gone.
HOST="${HOST:?set HOST to the server address, e.g. HOST=1.2.3.4 ./deploy.sh}"
USER_NAME="${REMOTE_USER:-ubuntu}"
KEY="${KEY:-$HOME/.ssh/id_rsa}"
REMOTE_PATH="${REMOTE_PATH:-~/HelloWorldAi}"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "$0")" && pwd)/}"

# One name for the stack, always. `docker compose down` infers the project from
# the directory holding the compose file -- "docker" -- so a stack first
# started under a different name is not matched, and down reports success while
# the old containers keep running. That happened: two containers served the
# public internet for twelve months after a down that claimed to have worked.
PROJECT_NAME="${PROJECT_NAME:-helloworldai}"

echo "🚀 Deploying to $USER_NAME@$HOST ($PROJECT_NAME)"

# === STEP 1: Sync ===
# --delete so a file removed here is removed there. Without it, deleted modules
# linger on the server and can still be imported.
#
# Every env file is excluded. Secrets belong on the machine that uses them, and
# syncing a local one over the server's would replace its keys with yours --
# including the artifact encryption key, which cannot be recovered.
echo "🔄 Syncing files..."
rsync -av --delete --progress -e "ssh -i $KEY" \
  --exclude 'venv' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'node_modules' \
  --exclude '*.log' \
  --exclude 'docker/data' \
  --exclude 'env/.env.*' \
  "$PROJECT_DIR" "$USER_NAME@$HOST:$REMOTE_PATH"

# === STEP 2: Restart ===
echo "🐳 Rebuilding and restarting..."
ssh -i "$KEY" "$USER_NAME@$HOST" bash -s <<EOF
  set -euo pipefail
  cd "$REMOTE_PATH/docker"
  docker compose -p "$PROJECT_NAME" -f docker-compose.yml down
  docker compose -p "$PROJECT_NAME" -f docker-compose.yml up -d --build
  echo "--- running ---"
  docker compose -p "$PROJECT_NAME" -f docker-compose.yml ps
EOF

echo "✅ Deploy complete"

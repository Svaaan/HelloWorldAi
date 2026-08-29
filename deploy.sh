#!/bin/bash
#
# Push this working copy to the server and restart the stack.

# Stop at the first failure. Without this a broken rsync still fell through to
# the ssh block, which takes the live stack down and rebuilds it from whatever
# managed to arrive.
set -euo pipefail

# === CONFIGURATION ===
KEY_PATH="C:/ssh.key/ssh-key-2025-04-02.key"
PROJECT_DIR="C:/Users/danie/Documents/programmering/HelloWorldAi/"
REMOTE_USER="ubuntu"
REMOTE_HOST="79.76.55.71"
# rsync hands this to the remote shell, which does expand the tilde.
# shellcheck disable=SC2088
REMOTE_PATH="~/HelloWorldAi"

# === STEP 1: Print status ===
echo "🚀 Starting deploy to $REMOTE_USER@$REMOTE_HOST ..."

# === STEP 2: Sync files (excludes venv, __pycache__, .git) ===
echo "🔄 Syncing files to VM..."
rsync -av --progress -e "ssh -i $KEY_PATH" \
  --exclude 'venv' \
  --exclude '.venv' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.git' \
  --exclude 'node_modules' \
  --exclude '*.log' \
  --exclude 'docker/data' \
  --exclude 'env/.env.test' \
  "$PROJECT_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# === STEP 3: SSH into server and restart docker ===
echo "🐳 Rebuilding Docker and restarting services..."
ssh -i "$KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" << EOF
  cd ~/HelloWorldAi/docker
  docker compose -f docker-compose.yml down
  docker compose -f docker-compose.yml up -d --build
EOF

echo "✅ Deploy complete!"

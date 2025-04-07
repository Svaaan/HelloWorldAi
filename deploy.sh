#!/bin/bash

# === CONFIGURATION ===
KEY_PATH="C:/ssh.key/ssh-key-2025-04-02.key"
PROJECT_DIR="C:/Users/danie/Documents/programmering/HelloWorldAi/"
REMOTE_USER="ubuntu"
REMOTE_HOST="79.76.55.71"
REMOTE_PATH="~/HelloWorldAi"

# === STEP 1: Print status ===
echo "🚀 Starting deploy to $REMOTE_USER@$REMOTE_HOST ..."

# === STEP 2: Sync files (excludes venv, __pycache__, .git) ===
echo "🔄 Syncing files to VM..."
rsync -av --progress -e "ssh -i $KEY_PATH" \
  --exclude 'venv' \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'node_modules' \
  --exclude '*.log' \
  "$PROJECT_DIR" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_PATH"

# === STEP 3: SSH into server and restart docker ===
echo "🐳 Rebuilding Docker and restarting services..."
ssh -i "$KEY_PATH" "$REMOTE_USER@$REMOTE_HOST" << EOF
  cd ~/HelloWorldAi/docker
  docker compose -f docker-compose.yml down
  docker compose -f docker-compose.yml up -d --build
EOF

echo "✅ Deploy complete!"

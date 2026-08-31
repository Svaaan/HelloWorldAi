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

# Required, like HOST, and for the same reason: getting it wrong is quiet.
# Caddy reads DOMAIN to decide which certificate to ask for, and its Caddyfile
# falls back to `localhost` when the variable is missing. A deploy without it
# comes up healthy, all four containers green, serving a self-signed
# certificate for `localhost` that every browser refuses. Nothing in the output
# says so -- this was found by curl, after the deploy reported success.
#
# An empty value is not the same as an unset one, and is worse: Caddy's
# {$DOMAIN:localhost} only falls back when the variable is unset, so DOMAIN=""
# asks it to serve a site with no name, and it exits.
DOMAIN="${DOMAIN:?set DOMAIN to the hostname this serves, e.g. DOMAIN=example.com ./deploy.sh}"
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

echo "🚀 Deploying to $USER_NAME@$HOST ($PROJECT_NAME) as $DOMAIN"

# === STEP 1: Sync ===
# --delete so a file removed here is removed there. Without it, deleted modules
# linger on the server and can still be imported.
#
# Every env file is excluded. Secrets belong on the machine that uses them, and
# syncing a local one over the server's would replace its keys with yours --
# including the artifact encryption key, which cannot be recovered.
#
# docker/.env is excluded for a second reason: --delete removes anything on the
# server that is not here, and that file is written there rather than committed.
# A deploy quietly deleted it, which took DOMAIN with it, and the restart that
# followed served a certificate for `localhost`.
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
  --exclude 'docker/.env' \
  "$PROJECT_DIR" "$USER_NAME@$HOST:$REMOTE_PATH"

# === STEP 2: Pull and restart ===
#
# Pull, not build. CI publishes the server image on every push to main, so the
# server downloads a finished one rather than compiling a multi-gigabyte image
# while it is also meant to be serving. A build on the box is slow, needs build
# tooling in production, and leaves the stack down if it fails.
#
# If the registry is unreachable, or you are deploying a branch CI has not
# built, fall back with:  BUILD_ON_SERVER=1 ./deploy.sh
echo "🐳 Restarting..."
ssh -i "$KEY" "$USER_NAME@$HOST" bash -s <<EOF
  set -euo pipefail
  # Unquoted on purpose: REMOTE_PATH defaults to ~/HelloWorldAi, and a tilde
  # inside double quotes is a literal, so the quoted form fails to cd.
  cd $REMOTE_PATH/docker

  # Written as well as exported, so that a compose command typed by hand on the
  # box later gets the same hostname this script used. Compose reads .env from
  # the directory holding the compose file.
  #
  # No backticks anywhere in this heredoc: it is unquoted so that DOMAIN and
  # PROJECT_NAME are baked in here, which means the local shell also runs any
  # command substitution it finds -- including one written inside a comment.
  echo "DOMAIN=$DOMAIN" > .env
  export DOMAIN="$DOMAIN"

  if [ "${BUILD_ON_SERVER:-0}" = "1" ]; then
    docker compose -p "$PROJECT_NAME" -f docker-compose.yml build
  else
    docker compose -p "$PROJECT_NAME" -f docker-compose.yml pull
  fi

  docker compose -p "$PROJECT_NAME" -f docker-compose.yml up -d
  echo "--- running ---"
  docker compose -p "$PROJECT_NAME" -f docker-compose.yml ps
EOF

echo "✅ Deploy complete"

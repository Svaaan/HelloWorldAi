#!/usr/bin/env bash
#
# Set up this machine as a compute node.
#
#   bash setup-node.sh --coordinator https://coordinator.example.com
#
# Installs Docker and the NVIDIA Container Toolkit if they are missing, checks
# that a container can actually see your GPUs, then fetches and starts the node.
#
# Safe to re-run: every step checks before it acts.
#
# Note: this does NOT install the CUDA Toolkit. The node runs inside a container
# built on nvidia/cuda, which already carries the CUDA runtime. On the host you
# only need the driver, Docker, and the container toolkit.

set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/Svaaan/HelloWorldAi.git}"
REPO_BRANCH="${REPO_BRANCH:-main}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/helloworldai-node}"
COORDINATOR_URL="${COORDINATOR_URL:-}"
COMPOSE_FILE="docker/docker-compose.node.yml"
CUDA_PROBE_IMAGE="nvidia/cuda:12.2.0-base-ubuntu22.04"
START_STACK=1
ASSUME_YES=0

# --- output --------------------------------------------------------------

if [ -t 1 ]; then
    BOLD=$'\033[1m'; RED=$'\033[31m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RESET=$'\033[0m'
else
    BOLD=""; RED=""; GREEN=""; YELLOW=""; RESET=""
fi

step() { printf '\n%s==> %s%s\n' "$BOLD" "$1" "$RESET"; }
info() { printf '    %s\n' "$1"; }
ok()   { printf '    %s%s%s\n' "$GREEN" "$1" "$RESET"; }
warn() { printf '    %s%s%s\n' "$YELLOW" "$1" "$RESET"; }
die()  { printf '\n%sError: %s%s\n\n' "$RED" "$1" "$RESET" >&2; exit 1; }

have() { command -v "$1" >/dev/null 2>&1; }

confirm() {
    [ "$ASSUME_YES" -eq 1 ] && return 0
    # No TTY (piped from curl) means we cannot ask; require --yes explicitly.
    [ -t 0 ] || die "No terminal to confirm on. Re-run with --yes, or download the script and run it directly."
    printf '    %s [y/N] ' "$1"
    read -r reply
    case "$reply" in [yY]*) return 0 ;; *) return 1 ;; esac
}

usage() {
    cat <<EOF
Usage: setup-node.sh --coordinator URL [options]

  --coordinator URL   Address of the central coordinator (required)
  --dir PATH          Where to install (default: $INSTALL_DIR)
  --branch NAME       Branch to check out (default: $REPO_BRANCH)
  --repo URL          Repository to clone (default: $REPO_URL)
  --no-start          Set everything up but do not start the node
  --yes               Do not prompt before installing packages
  -h, --help          Show this help
EOF
}

# --- arguments -----------------------------------------------------------

while [ $# -gt 0 ]; do
    case "$1" in
        --coordinator) COORDINATOR_URL="${2:-}"; shift 2 ;;
        --dir)         INSTALL_DIR="${2:-}"; shift 2 ;;
        --branch)      REPO_BRANCH="${2:-}"; shift 2 ;;
        --repo)        REPO_URL="${2:-}"; shift 2 ;;
        --no-start)    START_STACK=0; shift ;;
        --yes|-y)      ASSUME_YES=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *)             usage; die "Unknown option: $1" ;;
    esac
done

[ -n "$COORDINATOR_URL" ] || { usage; die "--coordinator is required."; }

case "$COORDINATOR_URL" in
    http://*|https://*) ;;
    *) die "--coordinator must start with http:// or https:// (got '$COORDINATOR_URL')" ;;
esac

# --- privilege helper ----------------------------------------------------

if [ "$(id -u)" -eq 0 ]; then
    SUDO=""
else
    have sudo || die "sudo is required but not installed."
    SUDO="sudo"
fi

# --- preflight -----------------------------------------------------------

step "Checking this machine"

[ "$(uname -s)" = "Linux" ] || die "This script targets Linux. On Windows, run it inside Ubuntu on WSL 2."

IS_WSL=0
if grep -qiE "(microsoft|wsl)" /proc/version 2>/dev/null; then
    IS_WSL=1
    info "Running under WSL."
fi

have apt-get || die "This script supports Debian/Ubuntu (apt). Install Docker and the NVIDIA Container Toolkit manually, then re-run with --no-start removed."

if ! have nvidia-smi; then
    if [ "$IS_WSL" -eq 1 ]; then
        die "nvidia-smi not found. Install the NVIDIA driver on WINDOWS (not inside WSL), then reopen this terminal."
    fi
    die "nvidia-smi not found. Install the NVIDIA driver for your GPU, then re-run."
fi

if ! nvidia-smi >/dev/null 2>&1; then
    die "nvidia-smi is present but failed to run. Your driver install looks incomplete."
fi

GPU_LIST="$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true)"
if [ -z "$GPU_LIST" ]; then
    die "No NVIDIA GPU detected by the driver."
fi
ok "GPUs visible to the driver:"
printf '%s\n' "$GPU_LIST" | while IFS= read -r line; do info "  - $line"; done

# --- docker --------------------------------------------------------------

step "Docker"

if have docker; then
    ok "Docker already installed ($(docker --version 2>/dev/null || echo 'version unknown'))."
else
    info "Docker is not installed. The official installer at https://get.docker.com will be run."
    confirm "Install Docker now?" || die "Docker is required."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    $SUDO sh /tmp/get-docker.sh
    rm -f /tmp/get-docker.sh
    ok "Docker installed."
fi

restart_docker() {
    if have systemctl && systemctl list-units >/dev/null 2>&1; then
        $SUDO systemctl restart docker
    else
        # WSL without systemd
        $SUDO service docker restart >/dev/null 2>&1 || true
    fi
}

if ! docker info >/dev/null 2>&1; then
    info "Docker daemon is not responding; trying to start it."
    restart_docker
    sleep 3
fi

DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
    if $SUDO docker info >/dev/null 2>&1; then
        DOCKER="$SUDO docker"
        warn "Your user cannot reach the Docker socket, so sudo will be used."
        warn "To fix permanently: sudo usermod -aG docker $USER   (then log out and back in)"
    else
        die "Docker is installed but the daemon is not running. Start Docker Desktop, or: sudo service docker start"
    fi
fi

if ! $DOCKER compose version >/dev/null 2>&1; then
    die "'docker compose' (v2) is unavailable. Update Docker to a version that bundles Compose v2."
fi
ok "Docker is usable."

# --- nvidia container toolkit -------------------------------------------

step "NVIDIA Container Toolkit"
info "This is what lets a container see your GPUs. The CUDA Toolkit is not needed."

install_container_toolkit() {
    local keyring="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"

    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | $SUDO gpg --dearmor -o "$keyring"

    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed "s#deb https://#deb [signed-by=$keyring] https://#g" \
        | $SUDO tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null

    $SUDO apt-get update
    $SUDO apt-get install -y nvidia-container-toolkit
    $SUDO nvidia-ctk runtime configure --runtime=docker
    restart_docker
    sleep 3
}

gpu_visible_in_container() {
    $DOCKER run --rm --gpus all "$CUDA_PROBE_IMAGE" nvidia-smi >/dev/null 2>&1
}

if gpu_visible_in_container; then
    ok "Containers can already see your GPUs."
else
    if have nvidia-ctk; then
        info "Toolkit present but GPU passthrough is not working; reconfiguring."
        $SUDO nvidia-ctk runtime configure --runtime=docker
        restart_docker
        sleep 3
    else
        info "The toolkit will be installed from NVIDIA's package repository."
        confirm "Install the NVIDIA Container Toolkit now?" || die "The container toolkit is required."
        install_container_toolkit
    fi

    step "Verifying GPU passthrough"
    info "Pulling a small CUDA image to test with (this may take a minute)."
    if gpu_visible_in_container; then
        ok "Containers can see your GPUs."
    else
        echo ""
        warn "A container still cannot see your GPUs. Diagnostic output:"
        $DOCKER run --rm --gpus all "$CUDA_PROBE_IMAGE" nvidia-smi || true
        die "GPU passthrough is not working. The node cannot run without it."
    fi
fi

# --- source --------------------------------------------------------------

step "Node software"

if [ -d "$INSTALL_DIR/.git" ]; then
    info "Updating existing install at $INSTALL_DIR"
    git -C "$INSTALL_DIR" fetch --depth 1 origin "$REPO_BRANCH"
    git -C "$INSTALL_DIR" checkout -q "$REPO_BRANCH"
    git -C "$INSTALL_DIR" reset --hard -q "origin/$REPO_BRANCH"
elif [ -e "$INSTALL_DIR" ]; then
    die "$INSTALL_DIR exists but is not a git checkout. Move it aside or pass --dir."
else
    have git || { confirm "git is needed. Install it?" && $SUDO apt-get install -y git; }
    have git || die "git is required."
    info "Cloning into $INSTALL_DIR"
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi
ok "Source ready at $INSTALL_DIR"

[ -f "$INSTALL_DIR/$COMPOSE_FILE" ] || die "$COMPOSE_FILE is missing from the checkout. Wrong branch?"

# Written where compose reads it, and kept out of git.
ENV_FILE="$INSTALL_DIR/.env"
{
    echo "# Generated by setup-node.sh"
    echo "COORDINATOR_URL=$COORDINATOR_URL"
} > "$ENV_FILE"
chmod 600 "$ENV_FILE"
ok "Wrote $ENV_FILE"

# --- start ---------------------------------------------------------------

if [ "$START_STACK" -eq 0 ]; then
    step "Done (not started)"
    info "To start it later:"
    info "  cd $INSTALL_DIR && $DOCKER compose -f $COMPOSE_FILE up -d --build"
    exit 0
fi

step "Building and starting the node"
info "The first build compiles a CUDA image and can take several minutes."

cd "$INSTALL_DIR"
$DOCKER compose -f "$COMPOSE_FILE" up -d --build

sleep 5
if ! $DOCKER compose -f "$COMPOSE_FILE" ps --status running 2>/dev/null | grep -q node; then
    warn "The node container is not running. Recent logs:"
    $DOCKER compose -f "$COMPOSE_FILE" logs --tail 40 node || true
    die "The node failed to start."
fi

step "Node is running"
cat <<EOF

    Next: open ${BOLD}http://localhost:3000${RESET} and register this node.
    You will be given a key file -- keep it safe, it is the only proof
    this node is yours.

    Coordinator : $COORDINATOR_URL
    Install dir : $INSTALL_DIR

    Useful commands:
      cd $INSTALL_DIR
      $DOCKER compose -f $COMPOSE_FILE logs -f node     # follow logs
      $DOCKER compose -f $COMPOSE_FILE restart node     # restart
      $DOCKER compose -f $COMPOSE_FILE down             # stop

EOF

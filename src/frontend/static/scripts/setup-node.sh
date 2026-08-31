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

# Never let git stop to ask for credentials. This runs unattended on someone
# else's machine; a private repository must fail with an explanation, not sit
# on a username prompt.
export GIT_TERMINAL_PROMPT=0

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

# An option that takes a value must actually have been given one. Without this
# check `shift 2` runs out of arguments, and under `set -e` the script exits
# silently with status 1 -- no message, nothing to act on.
require_value() {
    [ "$2" -ge 2 ] || die "$1 needs a value."
}

while [ $# -gt 0 ]; do
    case "$1" in
        --coordinator) require_value "$1" $#; COORDINATOR_URL="$2"; shift 2 ;;
        --dir)         require_value "$1" $#; INSTALL_DIR="$2"; shift 2 ;;
        --branch)      require_value "$1" $#; REPO_BRANCH="$2"; shift 2 ;;
        --repo)        require_value "$1" $#; REPO_URL="$2"; shift 2 ;;
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

# The node runs in a container, where "localhost" is the container itself. A
# coordinator address like this resolves fine from a shell and then fails
# forever from inside the node, which shows up only as a quiet retry loop in
# the logs -- the install otherwise looks like it worked.
case "$COORDINATOR_URL" in
    http://localhost*|https://localhost*|http://127.0.0.1*|https://127.0.0.1*)
        die "The coordinator cannot be '$COORDINATOR_URL'.

    The node runs inside a container, and localhost there means the container,
    not this machine. Use the address other machines reach the coordinator on --
    the setup page fills the right one in for you."
        ;;
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

# `have docker` is not enough on WSL. Docker Desktop puts a shim at
# /usr/bin/docker inside every distro; with integration switched off for this
# one the command exists, runs, and prints an error telling you to switch it
# on. Testing only for the command reported that error text as the version --
#
#     Docker already installed (
#     The command 'docker' could not be found in this WSL 2 distro. ...)
#
# -- and then asked for a sudo password to restart a daemon that was never
# there. Ask the command what version it is, and believe it only if it answers.
DOCKER_VERSION="$(docker --version 2>&1 || true)"
case "$DOCKER_VERSION" in
    "Docker version"*) DOCKER_OK=1 ;;
    *)                 DOCKER_OK=0 ;;
esac

if [ "$DOCKER_OK" -eq 1 ]; then
    ok "Docker already installed ($DOCKER_VERSION)."
elif have docker; then
    if [ "$IS_WSL" -eq 1 ]; then
        die "The docker command is here but does not work in this distro.

This is Docker Desktop's shim, and integration is switched off for this WSL
distro. Nothing needs installing -- turn it on:

    Docker Desktop -> Settings -> Resources -> WSL Integration
    enable this distro, then Apply & Restart

Make sure Docker Desktop itself is running, then run this script again.

It said:
$DOCKER_VERSION"
    fi
    die "The docker command is here but does not work:

$DOCKER_VERSION"
else
    info "Docker is not installed. The official installer at https://get.docker.com will be run."
    confirm "Install Docker now?" || die "Docker is required."
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    $SUDO sh /tmp/get-docker.sh
    rm -f /tmp/get-docker.sh
    ok "Docker installed."
fi

# Never prompt for a password here. On WSL with Docker Desktop there is no
# daemon in the distro to restart, so asking for sudo stops the script dead on
# a prompt that cannot lead anywhere. -n fails immediately instead, and the
# check below then says what to actually do.
# Non-interactive sudo, for the places that are only asking a question.
SUDO_PROBE=""
[ -n "$SUDO" ] && SUDO_PROBE="$SUDO -n"

restart_docker() {
    SUDO_N="$SUDO_PROBE"

    if have systemctl && systemctl list-units >/dev/null 2>&1; then
        $SUDO_N systemctl restart docker >/dev/null 2>&1 || true
    else
        # WSL without systemd
        $SUDO_N service docker restart >/dev/null 2>&1 || true
    fi
}

if ! docker info >/dev/null 2>&1; then
    info "Docker daemon is not responding; trying to start it."
    restart_docker
    sleep 3
fi

DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
    # A probe, so it must not block. Written as a plain `sudo docker info` this
    # sat on a password prompt while only asking a question -- and on WSL with
    # Docker Desktop the answer is no anyway: the socket belongs to the user,
    # not to root, so a password would have bought nothing. -n makes it answer
    # or give up, and the advice below is what actually helps.
    if $SUDO_PROBE docker info >/dev/null 2>&1; then
        DOCKER="$SUDO docker"
        warn "Your user cannot reach the Docker socket, so sudo will be used."
        warn "To fix permanently: sudo usermod -aG docker $USER   (then log out and back in)"
    else
        if [ "$IS_WSL" -eq 1 ]; then
            die "The docker command works but no Docker daemon is running in this distro.

    If you use Docker Desktop, turn this distro on under
      Settings -> Resources -> WSL Integration, then run this again.

    Otherwise start a daemon inside WSL with:
      sudo service docker start"
        fi
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

    # Check before cloning, so a private or misspelled repository produces a
    # sentence the contributor can act on rather than a git credential error.
    if ! git ls-remote --exit-code "$REPO_URL" >/dev/null 2>&1; then
        die "Cannot read $REPO_URL.
    The repository is private, does not exist, or this machine is offline.
    A contributor cannot install from a private repository -- ask whoever
    sent you here to publish it, or to give you a --repo URL you can read."
    fi

    info "Cloning into $INSTALL_DIR"
    git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$INSTALL_DIR"
fi
ok "Source ready at $INSTALL_DIR"

[ -f "$INSTALL_DIR/$COMPOSE_FILE" ] || die "$COMPOSE_FILE is missing from the checkout. Wrong branch?"

# Beside the compose file, because that is the project directory compose
# resolves .env against. Putting it at the top of the checkout looks tidier and
# is silently never read.
ENV_FILE="$INSTALL_DIR/$(dirname "$COMPOSE_FILE")/.env"
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

# 3000 is a popular address -- another dev server, another project, or this
# project's own test stack. Docker's answer is "Bind for 0.0.0.0:3000 failed:
# port is already allocated", which arrives after the whole image has built and
# reads as a broken install rather than as a clash you can step around.
#
# So look first, and move the outside half if it is taken. Only the host port
# changes; the container still serves 3000, so nothing inside has to know.
port_taken() {
    if have ss; then
        ss -lnt 2>/dev/null | grep -q ":$1 "
    elif have netstat; then
        netstat -lnt 2>/dev/null | grep -q ":$1 "
    else
        return 1                      # cannot tell; let docker decide
    fi
}

DASHBOARD_HOST_PORT="${DASHBOARD_HOST_PORT:-3000}"
if port_taken "$DASHBOARD_HOST_PORT"; then
    original="$DASHBOARD_HOST_PORT"
    for candidate in 3100 3200 3300 3400; do
        if ! port_taken "$candidate"; then
            DASHBOARD_HOST_PORT="$candidate"
            break
        fi
    done

    if [ "$DASHBOARD_HOST_PORT" = "$original" ]; then
        die "Port $original is in use and nothing nearby is free.

Free it, or choose one yourself:
    DASHBOARD_HOST_PORT=<port> $0 $*"
    fi
    warn "Port $original is already in use; the dashboard will be on $DASHBOARD_HOST_PORT."
fi
export DASHBOARD_HOST_PORT

$DOCKER compose -f "$COMPOSE_FILE" up -d --build

sleep 5
if ! $DOCKER compose -f "$COMPOSE_FILE" ps --status running 2>/dev/null | grep -q node; then
    warn "The node container is not running. Recent logs:"
    $DOCKER compose -f "$COMPOSE_FILE" logs --tail 40 node || true
    die "The node failed to start."
fi

step "Node is running"
cat <<EOF

    Next: open ${BOLD}http://localhost:${DASHBOARD_HOST_PORT}${RESET} and register this node.
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

#!/bin/bash

echo "== AI Node Setup =="

# STEP 1: Detect if we're inside WSL
if grep -qi microsoft /proc/version; then
    echo "📦 Running inside WSL (Windows Subsystem for Linux)"
elif [[ "$(uname)" == "Linux" ]]; then
    echo "📦 Running on native Linux"
else
    echo "❌ Not running inside WSL or Linux."
    echo "ℹ️ Attempting to install WSL for Windows 10/11..."

    # STEP 2: Check if Ubuntu WSL is already installed
    if wsl.exe -l -v | grep -qi "Ubuntu"; then
        echo "✅ Ubuntu WSL already installed — skipping reinstallation."
        echo "▶️ Launching Ubuntu for you..."
        powershell.exe -Command "Start-Process wsl.exe -ArgumentList '-d Ubuntu'"
        echo "▶️ Please continue the setup inside the Ubuntu terminal window that just opened."
        echo "👉 Run the following inside Ubuntu to continue setup:"
        echo "bash <(curl -fsSL http://127.0.0.1:3000/static/scripts/setup-inside-wsl.sh)"
        echo "🆕 If this is your first time using Ubuntu, follow this order after launch:"
        echo "1. Set a UNIX username (e.g., daniel) when prompted."
        echo "2. Set a secure password (you won’t see characters as you type)."
        echo "3. Once Ubuntu shell appears (e.g., daniel@DESKTOP:~$), run the command above."
        read -p "Press enter to close..."
        exit 0
    else
        echo "📥 Installing Ubuntu via WSL..."
        wsl --install -d Ubuntu
        echo "✅ WSL installation triggered. Please reboot and launch Ubuntu from Start Menu."
        echo "👉 After launch, run this in Ubuntu to continue setup:"
        echo "bash <(curl -fsSL http://127.0.0.1:3000/static/scripts/setup-inside-wsl.sh)"
        echo "🆕 If this is your first time using Ubuntu, follow this order after launch:"
        echo "1. Set a UNIX username (e.g., daniel) when prompted."
        echo "2. Set a secure password (you won’t see characters as you type)."
        echo "3. Once Ubuntu shell appears, run: bash <(curl -fsSL http://127.0.0.1:3000/static/scripts/setup-inside-wsl.sh)."
        read -p "Press enter to close..."
        exit 0
    fi
fi

# === SETUP CONTINUES HERE ONLY IF INSIDE WSL OR NATIVE LINUX ===
echo "🔄 Updating system packages..."
sudo apt update -y && sudo apt upgrade -y

# Docker
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# NVIDIA Container Toolkit
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "🎯 Installing NVIDIA Container Toolkit..."
    sudo apt install -y nvidia-container-toolkit
    sudo systemctl restart docker
    echo "✅ Toolkit installed"
else
    echo "✅ NVIDIA Container Toolkit already installed"
fi

# wget
if ! command -v wget &> /dev/null; then
    echo "📥 Installing wget..."
    sudo apt install -y wget
fi

# GPU Test
echo ""
echo "== GPU System Readiness Check =="
if ! docker info &> /dev/null; then
    echo "⚠️  Your user is likely not in the docker group. Run the following and restart Ubuntu:"
    echo "    sudo usermod -aG docker \$USER"
    echo "    exit"
    echo "Then relaunch Ubuntu."
else
    docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi && echo "✅ Docker can access GPU" || echo "❌ Docker can't access GPU"
fi

# CUDA Compiler
echo ""
echo "— CUDA Compiler Check (nvcc) —"
if command -v nvcc &> /dev/null; then
    nvcc --version
    echo "✅ CUDA compiler (nvcc) is available"
else
    echo "❌ CUDA compiler not found — some dev features may not work"
fi

echo ""
echo "✅ Setup complete!"
echo "📎 Please reboot or re-login if needed."
read -p "Press enter to close..."

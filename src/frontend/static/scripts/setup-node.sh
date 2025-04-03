#!/bin/bash

echo "== AI Node Setup =="

# Detect WSL or Linux
if grep -qi microsoft /proc/version; then
    echo "📦 Running inside WSL (Windows Subsystem for Linux)"
elif [[ "$(uname)" == "Linux" ]]; then
    echo "📦 Running on native Linux"
else
    echo "❌ Unsupported OS. This script only works on Ubuntu or WSL2."
    echo "➡️ Please follow the manual setup steps at /setup"
    read -p "Press enter to close..."
    exit 1
fi

# Update system
echo "🔄 Updating system packages..."
sudo apt update -y && sudo apt upgrade -y

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    curl -fsSL https://get.docker.com | sh
    sudo usermod -aG docker $USER
    echo "✅ Docker installed"
else
    echo "✅ Docker already installed"
fi

# Install NVIDIA Container Toolkit
if ! dpkg -l | grep -q nvidia-container-toolkit; then
    echo "🎯 Installing NVIDIA Container Toolkit..."
    sudo apt install -y nvidia-container-toolkit
    sudo systemctl restart docker
    echo "✅ Toolkit installed"
else
    echo "✅ NVIDIA Container Toolkit already installed"
fi

# Install wget if needed
if ! command -v wget &> /dev/null; then
    echo "📥 Installing wget..."
    sudo apt install -y wget
fi

# Test GPU access
echo "🚀 Testing GPU access inside Docker..."
docker run --rm --gpus all nvidia/cuda:12.2.0-base-ubuntu22.04 nvidia-smi || echo "⚠️ GPU not detected. You may need to reboot."

echo ""
echo "✅ Setup complete!"
echo "📎 If this was the first time installing Docker or GPU tools, please restart your computer or run:"
echo "    sudo reboot"
echo ""
read -p "Press enter to close this window..."

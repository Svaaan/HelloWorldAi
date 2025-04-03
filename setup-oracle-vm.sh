#!/bin/bash

# 🚀 Update system
echo "🛠️ Updating system..."
sudo apt update && sudo apt upgrade -y

# 🐳 Install Docker
echo "📦 Installing Docker..."
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# 🧱 Install Docker Compose
echo "📦 Installing Docker Compose..."
sudo apt install docker-compose -y

# 🔄 Restart shell group
newgrp docker

# ✅ Clone your project (optional)
# git clone https://github.com/yourusername/yourrepo.git
# cd yourrepo

echo "✅ Done! Now you can run:"
echo "   docker-compose up --build -d"

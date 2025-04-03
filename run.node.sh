#!/bin/bash

echo "🚀 Starting HelloworldAI Node with NVIDIA GPU support..."

docker run -d \
  --gpus all \
  --name helloworldai-node \
  --restart=always \
  -p 9100:9100 \
  -e COORDINATOR_BASE=https://your-vps-ip:8100 \
  helloworldai-node

echo "✅ Node is running in the background and connected to the coordinator!"

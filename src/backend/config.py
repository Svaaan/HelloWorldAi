# backend/config.py

import os
from dotenv import load_dotenv

# === Load the correct .env file ===
ENVIRONMENT = os.getenv("ENV", "local")

if ENVIRONMENT == "production":
    env_file = ".env.production"
else:
    env_file = ".env.local"

load_dotenv(dotenv_path=env_file, override=True)

# === Shared Config Variables ===
USE_DOCKER = os.getenv("USE_DOCKER", "false").lower() == "true"

# URLs
COORDINATOR_URL = os.getenv("COORDINATOR_URL", "http://127.0.0.1:8100")
NODE_URL = os.getenv("NODE_URL", "http://127.0.0.1:9100")

# Ports
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", 3000))
COORDINATOR_PORT = int(os.getenv("COORDINATOR_PORT", 8100))
NODE_PORT = int(os.getenv("NODE_PORT", 9100))

# Optional logs to check loading
print(f"✅ Loaded config: ENVIRONMENT={ENVIRONMENT}")
print(f"✅ USE_DOCKER={USE_DOCKER}")
print(f"✅ COORDINATOR_URL={COORDINATOR_URL}")
print(f"✅ NODE_URL={NODE_URL}")
print(f"✅ Ports: Dashboard={DASHBOARD_PORT}, Coordinator={COORDINATOR_PORT}, Node={NODE_PORT}")

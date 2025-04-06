import os
import logging
from dotenv import load_dotenv

# === Global logging configuration ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# === Load the correct .env file based on ENV variable ===
ENVIRONMENT = os.getenv("ENV", "local")

if ENVIRONMENT == "production":
    env_file = ".env.production"
elif ENVIRONMENT == "test":
    env_file = ".env.test"
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

# === Print once, safely ===
if not globals().get("CONFIG_LOGGED"):
    logger.info(f"✅ Loaded config: ENVIRONMENT={ENVIRONMENT}")
    logger.info(f"✅ USE_DOCKER={USE_DOCKER}")
    logger.info(f"✅ COORDINATOR_URL={COORDINATOR_URL}")
    logger.info(f"✅ NODE_URL={NODE_URL}")
    logger.info(f"✅ Ports: Dashboard={DASHBOARD_PORT}, Coordinator={COORDINATOR_PORT}, Node={NODE_PORT}")
    globals()["CONFIG_LOGGED"] = True

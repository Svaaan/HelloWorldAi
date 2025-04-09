# connectionService.py

import requests
import os
import logging

logger = logging.getLogger(__name__)

def background_connection_handler(payload, node_info):
    # Clean payload before sending
    payload.pop("cpu_usage", None)
    payload.pop("gpu_usage", None)
    payload.pop("cpu_benchmark", None)
    payload.pop("gpu_benchmark", None)

    coordinator_url = os.getenv('COORDINATOR_URL', 'http://127.0.0.1:8100/connect-node')
    logger.info(f"📡 Attempting to connect to coordinator at {coordinator_url}")
    try:
        res = requests.post(f"{coordinator_url}/connect-node", json=payload, timeout=10)
        if res.status_code == 200:
            node_info["connected"] = True
            logger.info(f"✅ Node '{node_info['node_id']}' connected successfully!")
        else:
            logger.error(f"❌ Connection failed. Status: {res.status_code}, Response: {res.text}")
    except requests.exceptions.RequestException as e:
        logger.error(f"🚨 Connection error to coordinator: {e}")



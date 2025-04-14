import logging
import httpx
import os
import asyncio


# Initialize logger
logger = logging.getLogger(__name__)

async def background_connection_handler(payload, node_info, max_retries=5, delay=2):
    payload.pop("cpu_usage", None)
    payload.pop("gpu_usage", None)
    payload.pop("cpu_benchmark", None)
    payload.pop("gpu_benchmark", None)

    coordinator_url = os.getenv('COORDINATOR_URL', 'http://127.0.0.1:8100')
    logging.info(f"📡 Attempting to register node at coordinator {coordinator_url}")
    logging.info(f"📦 Payload being sent: {payload}")

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(f"{coordinator_url}/connect-node", json=payload, timeout=10)
                res.raise_for_status()

                data = res.json()

                if data.get("status") == "success" and data.get("node_id"):
                    node_info["connected"] = True
                    node_info["node_id"] = data["node_id"]  # ✅ Here we assign node_id from coordinator!
                    logging.info(f"✅ Node registered successfully! Node ID: {data['node_id']}")
                    return data

                logging.warning(f"❌ Registration failed. Response: {data}")

        except httpx.RequestError as e:
            logging.warning(f"🚨 Attempt {attempt}: Registration error to coordinator: {e}")
        except httpx.HTTPStatusError as e:
            logging.warning(f"🚨 Attempt {attempt}: HTTP status error: {e.response.text}")

        if attempt < max_retries:
            wait_time = delay * attempt
            logging.info(f"🔄 Retrying in {wait_time} seconds...")
            await asyncio.sleep(wait_time)

    logging.error(f"❌ All {max_retries} attempts failed to register node.")
    return None


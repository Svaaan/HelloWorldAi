import logging
import httpx
import os
import asyncio
import uuid  # ✅ Import uuid to generate node_id

# Initialize logger
logger = logging.getLogger(__name__)

async def background_connection_handler(payload, node_info, max_retries=5, delay=2):
    # Clean up dynamic fields before sending
    payload.pop("cpu_usage", None)
    payload.pop("gpu_usage", None)
    payload.pop("cpu_benchmark", None)
    payload.pop("gpu_benchmark", None)

    # ✅ Ensure node_id is included in the payload!
    payload["node_id"] = node_info.get("node_id") or f"node_{uuid.uuid4()}"  # Generate if missing

    coordinator_url = os.getenv('COORDINATOR_URL', 'http://127.0.0.1:8100')
    logging.info(f"📡 Attempting to register node at coordinator {coordinator_url}")
    logging.info(f"📦 Payload being sent: {payload}")  # ✅ Add log for full payload debug

    for attempt in range(1, max_retries + 1):
        try:
            async with httpx.AsyncClient() as client:
                res = await client.post(f"{coordinator_url}/connect-node", json=payload, timeout=10)
                res.raise_for_status()  # Will raise if non-200

                data = res.json()

                if data.get("status") == "success" and data.get("node_id"):
                    node_info["connected"] = True
                    node_info["node_id"] = data["node_id"]  # ✅ Save node_id back to node_info!
                    logging.info(f"✅ Node registered successfully! Node ID: {data['node_id']}")
                    return data  # ✅ Return response to caller

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
    return None  # ✅ Return None if all attempts fail

import requests
from backend.node import node_info

def register():
    payload = {
        "node_id": node_info["node_id"],
        "ip": node_info["ip"],
        "port": node_info["port"],
        "capabilities": node_info["capabilities"]
    }

    print("Sending payload:", payload)

    try:
        # Send the POST request to register the node
        res = requests.post("http://127.0.0.1:8100/register-node", json=payload)

        print("Response Status:", res.status_code)
        print("Response Text:", res.text)

        if res.status_code == 200:
            print("✅ Node registered successfully!")
            node_info["registered"] = True
        elif res.status_code == 405:
            print("❌ Method Not Allowed: Check if the route is correct.")
        elif res.status_code == 404:
            print("❌ Not Found: The endpoint is not found. Please verify the URL.")
        else:
            print(f"❌ Registration failed with status {res.status_code}: {res.text}")
    except requests.exceptions.ConnectionError as e:
        print(f"⚠️ Connection error: {e}")
    except requests.exceptions.Timeout as e:
        print(f"⚠️ Timeout error: {e}")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")

if __name__ == "__main__":
    register()

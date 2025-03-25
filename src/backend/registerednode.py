import requests
from backend.node import node_info

def register():
    payload = {
        "node_id": node_info["node_id"],
        "ip": node_info["ip"],
        "port": node_info["port"],
        "capabilities": node_info["capabilities"]
    }

    try:
        res = requests.post("http://127.0.0.1:8100/register-node", json=payload)
        if res.status_code == 200:
            print("✅ Node registered successfully!")
            node_info["registered"] = True
        else:
            print("❌ Registration failed:", res.text)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    register()

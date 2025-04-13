# src/backend/service/gpus-json.py

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GPU_DB_PATH = os.path.join(BASE_DIR, 'gpu-db.json')

# Load the GPU database once
with open(GPU_DB_PATH, 'r', encoding='utf-8') as f:
    gpu_db = json.load(f)

def get_cuda_cores(gpu_name: str):
    gpu_name_clean = gpu_name.lower().replace("nvidia", "").replace("geforce", "").strip()

    for gpu in gpu_db:
        db_name_clean = gpu['name'].lower().replace("nvidia", "").replace("geforce", "").strip()
        if gpu_name_clean in db_name_clean or db_name_clean in gpu_name_clean:
            return gpu.get('cuda_cores')

    return None

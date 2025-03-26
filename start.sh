#!/bin/bash

# Ensure the environment is activated if using a virtual environment
# Example: If you're using a venv, uncomment the following line
# source /path/to/your/venv/bin/activate

# Install dependencies (optional step if not already done in Render's build step)
pip install -r requirements.txt

export PYTHONPATH=$PYTHONPATH:$(pwd)/src
# Run the FastAPI app using Uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8000

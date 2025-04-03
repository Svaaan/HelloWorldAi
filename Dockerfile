# Use NVIDIA CUDA runtime image with Ubuntu and Python preinstalled
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory inside the container
WORKDIR /app

# Install Python + tools (Ubuntu base doesn't include them by default)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3.11-dev \
    curl \
    && ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install Python deps early for cache reuse
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./src ./src
COPY ./static ./static
COPY ./templates ./templates
COPY app.py .

# Environment variables
ENV USE_DOCKER=true
ENV PYTHONPATH=/app/src

# Run FastAPI app via Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "3000"]

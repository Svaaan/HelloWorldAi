# Use base image with NVIDIA CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3.11-dev \
    curl \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip \
 && rm -rf /var/lib/apt/lists/*

# Copy only requirements for layer caching
COPY requirements.txt .

# Install Python dependencies (will be cached unless requirements.txt changes)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ✅ Copy source code after installing deps
COPY ./src ./src
COPY ./src/frontend/static ./static
COPY ./src/frontend/template ./templates
COPY ./src/app.py .

# Set environment variables
ENV USE_DOCKER=true
ENV PYTHONPATH=/app/src

# ✅ No hardcoded app
CMD ["sleep", "infinity"]

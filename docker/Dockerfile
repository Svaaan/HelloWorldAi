#  Use base image with NVIDIA CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

#  Set working directory
WORKDIR /app

#  Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-venv \
    python3.11-dev \
    curl \
 && ln -sf /usr/bin/python3.11 /usr/bin/python \
 && ln -sf /usr/bin/pip3 /usr/bin/pip \
 && rm -rf /var/lib/apt/lists/*

#  Install MongoDB tools (Optional, if you need mongo cli inside container)
# RUN curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-archive-keyring.gpg \
#     && echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-archive-keyring.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-6.0.list \
#     && apt-get update && apt-get install -y mongodb-mongosh \
#     && rm -rf /var/lib/apt/lists/*

#  Copy only requirements file first (for caching)
COPY ./requirements.txt ./requirements.txt

#  Install dependencies (cached if requirements.txt is unchanged)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code (your app files)
COPY ./src ./src
COPY ./src/frontend/static ./static
COPY ./src/frontend/template ./templates
COPY ./src/app.py ./app.py

# Optional: Copy .env file if needed
# COPY .env .env

# ✅ Set environment variables
ENV USE_DOCKER=true
ENV PYTHONPATH=/app/src

# ✅ No hardcoded app entrypoint — Docker Compose controls this
CMD ["sleep", "infinity"]

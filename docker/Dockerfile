#  Use base image with NVIDIA CUDA support
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

#  Set working directory
WORKDIR /app

#  Install system-level dependencies.
#  One interpreter only. `python` used to point at python3.11 while
#  python3-pip installed into python3.10, so every package -- torch
#  included -- landed where `python` could not see it. Services still
#  started, because uvicorn is a 3.10 console script, but any
#  `python -m ...` found an empty interpreter.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    curl \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

#  Install MongoDB tools (Optional, if you need mongo cli inside container)
# RUN curl -fsSL https://www.mongodb.org/static/pgp/server-6.0.asc | gpg --dearmor -o /usr/share/keyrings/mongodb-archive-keyring.gpg \
#     && echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-archive-keyring.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | tee /etc/apt/sources.list.d/mongodb-org-6.0.list \
#     && apt-get update && apt-get install -y mongodb-mongosh \
#     && rm -rf /var/lib/apt/lists/*

#  Copy only requirements file first (for caching)
COPY ./requirements.txt ./requirements.txt

#  Install dependencies (cached if requirements.txt is unchanged)
RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements.txt

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

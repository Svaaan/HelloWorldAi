# Two images, because two very different machines run this code.
#
# Everything used to build from nvidia/cuda:12.2.0-runtime-ubuntu22.04, which
# made every image about 12 GB. That is the right base for exactly one of the
# three services. The coordinator and the dashboard never touch a GPU: the
# coordinator imports torch only to rebuild a returned model and score it on
# the holdout, and that runs on the CPU. They were carrying a CUDA runtime and
# a CUDA build of torch -- several gigabytes each -- to do arithmetic they do
# on the processor anyway.
#
# Build a specific one with --target:
#
#   docker build --target server -f docker/Dockerfile .
#   docker build --target node   -f docker/Dockerfile .
#
# The compose files name their target, so `docker compose build` picks the
# right one per service.


# --------------------------------------------------------------------------
# server: the coordinator and the dashboard. No GPU, no CUDA.
# --------------------------------------------------------------------------
FROM python:3.11-slim AS server

WORKDIR /app

# curl is used by the container healthchecks and by hand when debugging.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

ARG REQUIREMENTS=requirements.txt
COPY ./${REQUIREMENTS} ./requirements-install.txt

# torch first, from the index that publishes CPU-only wheels. The default
# wheel on PyPI bundles the CUDA libraries and is most of the image size;
# 2.6.0+cpu satisfies the `torch==2.6.0` pin that follows, so the second
# install leaves it alone.
RUN python -m pip install --no-cache-dir --upgrade pip \
 && python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu torch \
 && python -m pip install --no-cache-dir -r requirements-install.txt

COPY ./src ./src
COPY ./src/frontend/static ./static
COPY ./src/frontend/template ./templates
COPY ./src/app.py ./app.py

ENV USE_DOCKER=true
ENV PYTHONPATH=/app/src

# No entrypoint: compose says which service this container is.
CMD ["sleep", "infinity"]


# --------------------------------------------------------------------------
# node: a contributor's agent. This one does need the GPU.
# --------------------------------------------------------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04 AS node

WORKDIR /app

# One interpreter only. `python` used to point at python3.11 while python3-pip
# installed into python3.10, so every package -- torch included -- landed where
# `python` could not see it. Services still started, because uvicorn is a 3.10
# console script, but any `python -m ...` found an empty interpreter.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    curl \
 && ln -sf /usr/bin/python3 /usr/bin/python \
 && rm -rf /var/lib/apt/lists/*

# A contributor's node needs a fraction of the project. Installing the rest on
# their machine cost several gigabytes and many minutes for packages node.py
# never imports.
ARG REQUIREMENTS=requirements-node.txt
COPY ./${REQUIREMENTS} ./requirements-install.txt

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir -r requirements-install.txt

COPY ./src ./src
COPY ./src/frontend/static ./static
COPY ./src/frontend/template ./templates
COPY ./src/app.py ./app.py

ENV USE_DOCKER=true
ENV PYTHONPATH=/app/src

CMD ["sleep", "infinity"]

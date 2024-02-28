ARG IMAGE=python:3.9-slim-bullseye
FROM ${IMAGE} as base

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ninja-build \
    libopenblas-dev \
    build-essential && \
    # Clean up cache to reduce layer size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . /app

RUN python3 -m pip install --upgrade pip && \
    pip install pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context llama-cpp-python

## Choose and download model
#RUN python3 -m pip install huggingface-hub==0.20.1
#
## Create the models directory
#RUN mkdir -p /app/models
#
## Download the model
#RUN huggingface-cli download lmstudio-ai/gemma-2b-it-GGUF gemma-2b-it-q4_k_m.gguf --local-dir ./models --local-dir-use-symlinks False

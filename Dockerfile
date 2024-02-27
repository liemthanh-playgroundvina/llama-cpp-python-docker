ARG IMAGE=python:3.9-slim-bullseye
FROM ${IMAGE}
ARG IMAGE

RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ninja-build \
    libopenblas-dev \
    build-essential

WORKDIR /app

COPY . .

# Install depencencies
RUN python3 -m pip install --upgrade pip pytest cmake scikit-build setuptools fastapi uvicorn sse-starlette pydantic-settings starlette-context

# Install llama-cpp-python (build with cuda)
RUN python3 -m pip install llama-cpp-python

RUN make deps && make build && make clean

## Choose and download model
#RUN python3 -m pip install huggingface-hub==0.20.1
#
## Create the models directory
#RUN mkdir -p /app/models
#
## Download the model
#RUN huggingface-cli download lmstudio-ai/gemma-2b-it-GGUF gemma-2b-it-q4_k_m.gguf --local-dir ./models --local-dir-use-symlinks False

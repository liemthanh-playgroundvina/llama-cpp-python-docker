ARG BASE_IMAGE=ubuntu:20.04

FROM ${BASE_IMAGE} AS build

RUN apt-get update -y && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends ca-certificates git build-essential \
    && update-ca-certificates

RUN git config --global http.sslVerify false

RUN git clone https://github.com/ggerganov/llama.cpp

WORKDIR /llama.cpp

RUN make

FROM ${BASE_IMAGE} AS runtime

WORKDIR /app

COPY --from=build /llama.cpp/server /app/server
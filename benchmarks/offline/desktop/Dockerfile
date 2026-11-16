# AI-12 desktop benchmark image: llama.cpp built from a pinned release, plus Python
# for the stdlib-only harness. Model weights are never baked in; they live in a
# named volume mounted at /models.
#
#   docker build -t ai12-bench benchmarks/offline/desktop
#   docker volume create ai12_models
FROM python:3.12-slim

ARG LLAMA_CPP_TAG=v1.0.0
ARG LLAMA_CPP_COMMIT=b1b770d90d94cad9e9ea5b993bdbc383a522fb28

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${LLAMA_CPP_TAG}" https://github.com/ggml-org/llama.cpp /src/llama.cpp \
 && test "$(git -C /src/llama.cpp rev-parse HEAD)" = "${LLAMA_CPP_COMMIT}" \
 && cmake -S /src/llama.cpp -B /src/build -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF \
 && cmake --build /src/build -j --target llama-server llama-bench llama-cli \
 && mkdir -p /opt/llama \
 && cp -r /src/build/bin /opt/llama/bin \
 && rm -rf /src/build

WORKDIR /work

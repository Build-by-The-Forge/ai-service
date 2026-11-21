#!/usr/bin/env bash
# Build llama.cpp v1.0.0 on Linux or WSL without root or Docker: cmake and ninja come
# from PyPI into a private folder. Needs gcc/g++, git, python3 with pip.
# Used for the desktop results because Docker Desktop was unavailable on the 8 GB
# laptop; desktop/Dockerfile is the equivalent containerised build.
#   bash benchmarks/offline/desktop/build_llama_cpp.sh   -> $AI12_ROOT/build/bin
# The server's embedded web UI is fetched from a moving remote bundle at build time and is
# unused here (--no-webui), so it is not fetched: the build is then reproducible offline.
set -euo pipefail
ROOT=${AI12_ROOT:-$HOME/ai12}
mkdir -p "$ROOT"
cd "$ROOT"
if [ ! -x pytools/bin/cmake ]; then
  python3 -m pip install --quiet --target "$ROOT/pytools" --break-system-packages cmake==4.1.2 ninja==1.13.0
fi
export PATH=$ROOT/pytools/bin:$PATH PYTHONPATH=$ROOT/pytools
if [ ! -d llama.cpp ]; then
  git clone --quiet --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp llama.cpp
fi
test "$(git -C llama.cpp rev-parse HEAD)" = b1b770d90d94cad9e9ea5b993bdbc383a522fb28
cmake -S llama.cpp -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DGGML_NATIVE=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build build -j 4 --target llama-server llama-bench llama-cli
ls -la build/bin

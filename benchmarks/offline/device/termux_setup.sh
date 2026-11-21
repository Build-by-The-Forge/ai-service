#!/data/data/com.termux/files/usr/bin/bash
# Run INSIDE Termux on the phone. Builds llama.cpp v1.0.0 natively (no NDK, no root)
# and prepares ~/ai12. Re-runnable.
#
#   bash termux_setup.sh            CPU build (every phone)
#   bash termux_setup.sh vulkan     also a Vulkan build (Mali/Adreno/PowerVR GPUs)
#   bash termux_setup.sh opencl     also an OpenCL build (Adreno GPUs only)
set -euo pipefail
ROOT=$HOME/ai12
mkdir -p "$ROOT"/models "$ROOT"/runs
pkg update -y
pkg install -y git cmake clang ninja python
cd "$ROOT"
[ -d llama.cpp ] || git clone --depth 1 --branch v1.0.0 https://github.com/ggml-org/llama.cpp
test "$(git -C llama.cpp rev-parse HEAD)" = b1b770d90d94cad9e9ea5b993bdbc383a522fb28

build() {  # build <dir> <extra cmake flags...>
  local dir=$1; shift
  cmake -S llama.cpp -B "$dir" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DBUILD_SHARED_LIBS=OFF "$@"
  # -j 2 keeps a 4 GB phone responsive and stops the build being OOM-killed.
  cmake --build "$dir" -j 2 --target llama-server llama-bench llama-cli
}

build build-cpu
case "${1:-}" in
  vulkan)
    pkg install -y vulkan-headers vulkan-loader-android shaderc
    build build-vulkan -DGGML_VULKAN=ON ;;
  opencl)
    pkg install -y opencl-headers ocl-icd
    build build-opencl -DGGML_OPENCL=ON ;;
esac

termux-setup-storage || true   # grants ~/storage/shared so results can be pulled with adb
mkdir -p "$HOME/storage/shared/ai12" 2>/dev/null || true
echo "built: $(ls -d "$ROOT"/build-*/bin)"

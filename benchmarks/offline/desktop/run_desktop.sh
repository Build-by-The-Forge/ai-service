#!/usr/bin/env bash
# Desktop run of one registered llama.cpp model (Linux or WSL).
#
#   bash benchmarks/offline/desktop/run_desktop.sh <model-id> [threads]
#
# Fetches the pinned file into $AI12_ROOT/models (sha256-checked), runs bench.py with
# the binaries from build_llama_cpp.sh, writes runs/desktop/<model-id>.json, and
# deletes the weights afterwards unless AI12_KEEP=1.
set -euo pipefail
MODEL_ID=$1
THREADS=${2:-4}
ROOT=${AI12_ROOT:-$HOME/ai12}
HERE=$(cd "$(dirname "$0")/.." && pwd)
FILE=$(python3 -I -c "import json,sys; m=[m for m in json.load(open(sys.argv[1]))['models'] if m['id']==sys.argv[2]][0]; assert m['runtime']=='llama.cpp'; print(m['file'])" "$HERE/models.json" "$MODEL_ID")
python3 -I "$HERE/fetch_model.py" "$MODEL_ID" --dest "$ROOT/models"
python3 "$HERE/bench.py" --model-id "$MODEL_ID" --gguf "$ROOT/models/$FILE" \
  --bin-dir "$ROOT/build/bin" --threads "$THREADS" --ctx 4096 --reps 3 \
  --sampled-repeats 3 --label desktop --out "$HERE/runs/desktop/$MODEL_ID.json"
[ "${AI12_KEEP:-0}" = 1 ] || rm -f "$ROOT/models/$FILE"

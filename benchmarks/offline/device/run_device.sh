#!/data/data/com.termux/files/usr/bin/bash
# Run INSIDE Termux. One model, one backend: the task-set run, then a thermal soak.
#
#   bash run_device.sh <model-id> <gguf-file> [cpu|vulkan|opencl] [threads]
#
# Writes ~/ai12/runs/<label>/<model-id>.<backend>.json and the soak CSV, then copies
# them to shared storage (/sdcard/ai12) for `adb pull`.
set -euo pipefail
MODEL_ID=$1
GGUF=$2
BACKEND=${3:-cpu}
THREADS=${4:-4}
ROOT=$HOME/ai12
BIN=$ROOT/build-$BACKEND/bin
HERE=$(cd "$(dirname "$0")/.." && pwd)    # benchmarks/offline
LABEL=$(getprop ro.product.manufacturer)-$(getprop ro.product.model)
LABEL=${LABEL// /_}
OUT=$ROOT/runs/$LABEL
mkdir -p "$OUT"

NGL=()
[ "$BACKEND" != cpu ] && NGL=(--server-arg=-ngl --server-arg=99)

# 1. Task set: footprint, cold start, TTFT, tokens/s, quality. Fewer sampled repeats
#    than the desktop run to keep the phone's session short.
python "$HERE/bench.py" --model-id "$MODEL_ID" --gguf "$GGUF" --bin-dir "$BIN" \
  --threads "$THREADS" --ctx 4096 --reps 3 --sampled-repeats 1 --label "$LABEL" \
  --out "$OUT/$MODEL_ID.$BACKEND.json" "${NGL[@]}"

# 2. Thermal soak: llama-bench tg128 back to back for 10 minutes. The tokens/s curve
#    shows throttling; the host monitor records temperatures at the same time.
SOAK=$OUT/$MODEL_ID.$BACKEND.soak.csv
echo "epoch_s,tg128_tps" > "$SOAK"
NGL_BENCH=()
[ "$BACKEND" != cpu ] && NGL_BENCH=(-ngl 99)
END=$(( $(date +%s) + 600 ))
while [ "$(date +%s)" -lt "$END" ]; do
  TPS=$("$BIN/llama-bench" -m "$GGUF" -t "$THREADS" -p 0 -n 128 -r 1 -o csv "${NGL_BENCH[@]}" \
        | awk -F, 'NR==1{for(i=1;i<=NF;i++) if($i=="\"avg_ts\"") c=i} NR==2{gsub(/"/,"",$c); print $c}')
  echo "$(date +%s),$TPS" >> "$SOAK"
done

cp "$OUT"/"$MODEL_ID".* "$HOME/storage/shared/ai12/" 2>/dev/null || true
echo "done: $OUT"

#!/usr/bin/env bash
# Run on the PC while run_device.sh runs on the phone. Samples battery and thermal
# state over adb every 5 s into a CSV until stopped with Ctrl-C.
#
#   bash host_monitor.sh <out.csv> [adb-serial]
#
# Use wireless debugging (Settings > Developer options > Wireless debugging, then
# `adb pair` / `adb connect`): over USB the phone charges and battery drain is meaningless.
set -euo pipefail
OUT=$1
ADB_BIN=${ADB_BIN:-adb}   # set to the full path when adb is not on PATH
ADB=("$ADB_BIN")
[ -n "${2:-}" ] && ADB=("$ADB_BIN" -s "$2")
echo "epoch_s,battery_level_pct,battery_temp_c,plugged,current_now_ua,thermal_status,max_skin_or_cpu_c,mem_available_kb" > "$OUT"
while true; do
  BAT=$("${ADB[@]}" shell dumpsys battery)
  LEVEL=$(awk -F': ' '/  level:/{print $2; exit}' <<<"$BAT")
  TEMP=$(awk -F': ' '/  temperature:/{printf "%.1f", $2/10; exit}' <<<"$BAT")
  PLUGGED=$(grep -Eq '(AC|USB|Wireless) powered: true' <<<"$BAT" && echo 1 || echo 0)
  CURRENT=$("${ADB[@]}" shell cat /sys/class/power_supply/battery/current_now 2>/dev/null | tr -d '\r' || true)
  THERMAL=$("${ADB[@]}" shell dumpsys thermalservice)
  STATUS=$(awk -F': ' '/Thermal Status:/{print $2; exit}' <<<"$THERMAL" | tr -d '\r')
  # Highest CPU or SKIN temperature among the "Current temperatures from HAL" block.
  MAXT=$(awk '/Current temperatures from HAL/{f=1;next} /Current cooling devices/{f=0}
              f && /mType=(0|3)/ {match($0,/mValue=[0-9.]+/); v=substr($0,RSTART+7,RLENGTH-7); if(v>m)m=v}
              END{print m+0}' <<<"$THERMAL")
  # System-wide headroom: how close the phone is to the low-memory killer.
  AVAIL=$("${ADB[@]}" shell cat /proc/meminfo | awk '/MemAvailable:/{print $2; exit}')
  echo "$(date +%s),$LEVEL,$TEMP,$PLUGGED,${CURRENT:-},${STATUS:-},$MAXT,${AVAIL:-}" >> "$OUT"
  sleep 5
done

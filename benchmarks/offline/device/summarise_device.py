"""Attach battery, thermal and throttling figures to a device run file.

    python benchmarks/offline/device/summarise_device.py RUN.json MONITOR.csv SOAK.csv

Reads the host_monitor.sh CSV for the window covered by the run and the soak, and
the run_device.sh soak CSV, then writes a "device_monitor" block into RUN.json.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any


def _floats(rows: list[dict[str, str]], key: str) -> list[float]:
    out = []
    for row in rows:
        try:
            out.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def summarise(monitor: list[dict[str, str]], soak: list[dict[str, str]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    tps = _floats(soak, "tg128_tps")
    times = _floats(soak, "epoch_s")
    if tps:
        head = tps[: max(1, len(tps) // 10)]
        tail = tps[-max(1, len(tps) // 10) :]
        result["soak_minutes"] = round((times[-1] - times[0]) / 60, 1) if len(times) > 1 else 0
        result["soak_first_tps"] = round(sum(head) / len(head), 2)
        result["soak_last_tps"] = round(sum(tail) / len(tail), 2)
        # 1.0 = no throttling; 0.6 = sustained speed is 60% of the cold speed.
        result["sustained_ratio"] = round(result["soak_last_tps"] / result["soak_first_tps"], 2)
    level = _floats(monitor, "battery_level_pct")
    epoch = _floats(monitor, "epoch_s")
    plugged = any(r.get("plugged") == "1" for r in monitor)
    result["charging_during_run"] = plugged
    if len(level) > 1 and len(epoch) > 1 and not plugged:
        hours = (epoch[-1] - epoch[0]) / 3600
        result["battery_drop_pct"] = level[0] - level[-1]
        result["battery_drop_pct_per_hour"] = round((level[0] - level[-1]) / hours, 1)
    for key, name in (
        ("battery_temp_c", "max_battery_temp_c"),
        ("max_skin_or_cpu_c", "max_skin_or_cpu_c"),
        ("thermal_status", "worst_thermal_status"),
    ):
        values = _floats(monitor, key)
        if values:
            result[name] = max(values)
    available = _floats(monitor, "mem_available_kb")
    if available:
        result["min_mem_available_mb"] = round(min(available) / 1024)
    return result


def window(rows: list[dict[str, str]], start: float, end: float) -> list[dict[str, str]]:
    return [r for r in rows if r.get("epoch_s") and start <= float(r["epoch_s"]) <= end]


def main() -> int:
    run_path, monitor_path, soak_path = map(Path, sys.argv[1:4])
    run = json.loads(run_path.read_text(encoding="utf-8"))
    soak = list(csv.DictReader(soak_path.open(encoding="utf-8")))
    monitor = list(csv.DictReader(monitor_path.open(encoding="utf-8")))
    soak_times = _floats(soak, "epoch_s")
    total = sum(r["total_s"] for r in run["tasks"]) + run["cold_start_s"]
    start = (soak_times[0] if soak_times else 0) - total - 120  # task run precedes the soak
    end = soak_times[-1] if soak_times else float("inf")
    run["device_monitor"] = summarise(window(monitor, start, end), soak)
    run_path.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(run["device_monitor"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

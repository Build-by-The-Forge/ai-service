"""benchmarks/offline/device/summarise_device.py: battery, thermal and throttling."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "offline_device", ROOT / "benchmarks" / "offline" / "device" / "summarise_device.py"
)
device = importlib.util.module_from_spec(_spec)
sys.modules["offline_device"] = device
_spec.loader.exec_module(device)


def monitor_row(t, level, plugged="0", temp="35.0", status="0", skin="40", avail="900000"):
    return {
        "epoch_s": str(t),
        "battery_level_pct": str(level),
        "battery_temp_c": temp,
        "plugged": plugged,
        "thermal_status": status,
        "max_skin_or_cpu_c": skin,
        "mem_available_kb": avail,
    }


def test_drain_throttling_and_peaks():
    monitor = [
        monitor_row(0, 80),
        monitor_row(1800, 77, temp="41.5", status="2", skin="47", avail="300000"),
        monitor_row(3600, 74),
    ]
    soak = [
        {"epoch_s": str(t), "tg128_tps": str(v)}
        for t, v in zip(range(0, 600, 60), range(20, 10, -1))
    ]
    s = device.summarise(monitor, soak)
    assert s["battery_drop_pct_per_hour"] == 6.0
    assert s["max_battery_temp_c"] == 41.5 and s["worst_thermal_status"] == 2
    assert s["min_mem_available_mb"] == 293
    assert s["soak_first_tps"] == 20 and s["soak_last_tps"] == 11
    assert s["sustained_ratio"] == 0.55


def test_no_battery_figure_while_charging():
    s = device.summarise([monitor_row(0, 80, plugged="1"), monitor_row(3600, 81)], [])
    assert s["charging_during_run"] is True and "battery_drop_pct_per_hour" not in s


def test_window_keeps_rows_inside_the_run():
    rows = [monitor_row(t, 50) for t in (0, 10, 20, 30)]
    assert [r["epoch_s"] for r in device.window(rows, 5, 25)] == ["10", "20"]

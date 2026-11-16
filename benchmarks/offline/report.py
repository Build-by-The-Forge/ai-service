"""Turn AI-12 run files into comparison tables and a native-speaker review sheet.

    python benchmarks/offline/report.py benchmarks/offline/runs/desktop \
        --tables benchmarks/offline/runs/desktop/TABLES.md \
        --review benchmarks/offline/runs/desktop/human_review.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def pct(value: Any) -> str:
    return "-" if value is None else f"{100 * value:.0f}%"


def num(value: Any, digits: int = 1) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def load_runs(folder: Path) -> list[dict[str, Any]]:
    runs = []
    for path in sorted(folder.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") == 1:
            runs.append(data)
    return runs


def tables(runs: list[dict[str, Any]]) -> str:
    lines = [
        "### Footprint and speed",
        "",
        "| Model | Runtime | Download MB | Peak RSS MB | Cold start s | TTFT median s | "
        "Decode tok/s | Prefill tok/s | llama-bench tg128 | llama-bench pp512 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        s, b = r["summary"], r.get("llama_bench") or {}
        lines.append(
            f"| {r['model_id']} | {r['runtime']} | {r['download_bytes'] / 1e6:.0f} | "
            f"{num(r['memory'].get('peak_hwm_mb') or r['memory'].get('peak_rss_sampled_mb'), 0)} | "
            f"{num(r['cold_start_s'], 2)} | {num(s['ttft_median_s'], 2)} | "
            f"{num(s['decode_tps_median'])} | {num(s['prefill_tps_median'], 0)} | "
            f"{num((b.get('tg128') or {}).get('avg_ts'))} | {num((b.get('pp512') or {}).get('avg_ts'), 0)} |"
        )  # fmt: skip
    lines += [
        "",
        "### Structured output and tools (free = instruction only; schema = grammar-constrained)",
        "",
        "| Model | JSON strict (free) | JSON parseable (free) | Schema-valid (free) | "
        "Schema-valid (schema) | Extraction correct (free / schema) | Tool call correct | "
        "Tool name correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        s = r["summary"]
        f, c = s.get("free", {}), s.get("schema", {})
        lines.append(
            f"| {r['model_id']} | {pct(f.get('json_strict_rate'))} | "
            f"{pct(f.get('json_parseable_rate'))} | {pct(f.get('schema_ok_rate'))} | "
            f"{pct(c.get('schema_ok_rate'))} | {pct(f.get('extract_correct_rate'))} / "
            f"{pct(c.get('extract_correct_rate'))} | {pct(s.get('tool_call_accuracy'))} | "
            f"{pct(s.get('tool_name_accuracy'))} |"
        )
    lines += [
        "",
        "### Grounded answers in English (10 answerable, 6 unanswerable)",
        "",
        "free/schema: 3 passages per question, greedy. long: all 10 passages (context handling). "
        "sampled: temperature 0.7, repeated (consistency = share of sampled runs whose "
        "correctness matches the greedy run).",
        "",
        "| Model | Mode | Answer correct | Faithful (correct + only supporting citations) | "
        "Citation precision | False abstention | Abstained when unanswerable | "
        "Hallucinated when unanswerable |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        for mode in ("free", "schema", "long", "sampled"):
            m = r["summary"].get(mode, {})
            lines.append(
                f"| {r['model_id']} | {mode} | {pct(m.get('en_answer_accuracy'))} | "
                f"{pct(m.get('en_faithful_rate'))} | {pct(m.get('en_citation_precision'))} | "
                f"{pct(m.get('en_false_abstention_rate'))} | {pct(m.get('en_abstention_rate'))} | "
                f"{pct(m.get('en_hallucination_rate'))} |"
            )
    lines += [
        "",
        "| Model | Consistency (sampled vs greedy) | Long-context TTFT median s | "
        "Long-context prompt tokens |",
        "|---|---:|---:|---:|",
    ]
    for r in runs:
        s = r["summary"]
        lines.append(
            f"| {r['model_id']} | {pct(s.get('sampled_consistency'))} | "
            f"{num(s.get('long_ttft_median_s'), 2)} | {num(s.get('long_prompt_tokens_median'), 0)} |"
        )
    lines += [
        "",
        "### Nigerian languages (unreviewed prompts; 3 grounded tasks + 1 explanation each)",
        "",
        "| Model | Yoruba grounded (free / schema) | Hausa grounded | Igbo grounded | "
        "Yoruba explain: language share / terms | Hausa explain | Igbo explain | "
        "English explain terms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        s = r["summary"]
        f, c = s.get("free", {}), s.get("schema", {})
        cells = [
            f"{pct(f.get(f'{lang}_grounded_correct'))} / {pct(c.get(f'{lang}_grounded_correct'))}"
            for lang in ("yo", "ha", "ig")
        ]
        explains = [
            f"{pct(s.get(f'{lang}_explain_language_share'))} / "
            f"{pct(s.get(f'{lang}_explain_term_recall'))}"
            for lang in ("yo", "ha", "ig")
        ]
        lines.append(
            f"| {r['model_id']} | {' | '.join(cells)} | {' | '.join(explains)} | "
            f"{pct(s.get('en_explain_term_recall'))} |"
        )
    devices = [r for r in runs if r.get("device_monitor")]
    if devices:
        lines += [
            "",
            "### Device battery and thermal (device_monitor)",
            "",
            "| Model | Device | Battery %/h | Max battery C | Max skin/CPU C | "
            "Worst thermal status | Sustained ratio | Min MemAvailable MB |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in devices:
            d = r["device_monitor"]
            lines.append(
                f"| {r['model_id']} | {r['label']} | {num(d.get('battery_drop_pct_per_hour'))} | "
                f"{num(d.get('max_battery_temp_c'))} | {num(d.get('max_skin_or_cpu_c'))} | "
                f"{num(d.get('worst_thermal_status'), 0)} | {num(d.get('sustained_ratio'), 2)} | "
                f"{num(d.get('min_mem_available_mb'), 0)} |"
            )
    hw = runs[0]["hardware"] if runs else {}
    settings = runs[0]["settings"] if runs else {}
    lines += [
        "",
        f"Hardware: {hw.get('cpu')}, {hw.get('logical_cpus')} logical CPUs visible, "
        f"{hw.get('mem_total_mb')} MB RAM visible, {hw.get('system')} {hw.get('kernel')}. "
        f"Threads {settings.get('threads')}, context {settings.get('ctx')}, temperature 0.",
    ]
    return "\n".join(lines) + "\n"


def review_sheet(runs: list[dict[str, Any]], path: Path) -> int:
    rows = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "model_id",
                "task_id",
                "lang",
                "mode",
                "output",
                "fluency_0_3",
                "accuracy_0_3",
                "notes",
            ]
        )
        for r in runs:
            for t in r["tasks"]:
                if t["lang"] != "en" or t["type"] == "explain":
                    writer.writerow([r["model_id"], t["id"], t["lang"], t["mode"], t["output"]])
                    rows += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--review", type=Path)
    args = parser.parse_args()
    runs = load_runs(args.folder)
    args.tables.write_text(tables(runs), encoding="utf-8")
    if args.review:
        print(f"{review_sheet(runs, args.review)} rows for native-speaker review")
    print(f"{len(runs)} runs -> {args.tables}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

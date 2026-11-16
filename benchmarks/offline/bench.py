"""AI-12 offline benchmark runner for llama.cpp (llama-server + llama-bench).

Runs one model through the task set and writes one JSON result file. Standard
library only, Linux /proc for memory, so it runs unchanged in the desktop
container and in Termux on an Android phone (see device/DEVICE_PROCEDURE.md).

    python benchmarks/offline/bench.py --model-id qwen3.5-0.8b-q4km \
        --gguf /models/Qwen3.5-0.8B-Q4_K_M.gguf --bin-dir /opt/llama/bin \
        --out benchmarks/offline/runs/desktop/qwen3.5-0.8b-q4km.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, BinaryIO

sys.path.insert(0, str(Path(__file__).resolve().parent))

import scoring  # noqa: E402

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"

GROUNDED_SYSTEM = (
    "You are BUILD's offline study assistant. Answer the student's question using ONLY the "
    "course passages provided. Reply with one JSON object and nothing else, in exactly this "
    'form: {"answer": "<answer>", "citations": ["<passage id>"], "answerable": true}. '
    "Cite the id of every passage you used. If the passages do not contain the answer, reply "
    '{"answer": "", "citations": [], "answerable": false}. Never answer from general knowledge.'
)
EXTRACT_SYSTEM = (
    "Extract information from the course passage. Reply with one JSON object and nothing "
    "else. It must match this JSON Schema: {schema}"
)
EXPLAIN_SYSTEM = (
    "You are BUILD's offline study assistant. Explain using only the course passage. "
    "Use at most {limit} words. Write your whole reply in {language}."
)
TOOL_SYSTEM = (
    "You are BUILD's offline study assistant on the student's phone. Call a tool when one is "
    "needed to fulfil the request; otherwise reply briefly without calling a tool."
)
MAX_TOKENS = {"grounded": 256, "json_extract": 320, "explain": 400, "tool_call": 256}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def render_passages(ids: list[str], passages: dict[str, dict[str, Any]]) -> str:
    blocks = []
    for pid in ids:
        p = passages[pid]
        blocks.append(f"[{pid}] ({p['course']}: {p['title']})\n{p['text']}")
    return "\n\n".join(blocks)


def build_messages(
    task: dict[str, Any], passages: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    kind = task["type"]
    language = scoring.LANGUAGE_NAMES[task.get("lang", "en")]
    if kind == "grounded":
        system = GROUNDED_SYSTEM
        if task.get("lang", "en") != "en":
            system += f" Write the answer text in {language}."
        user = (
            f"Passages:\n{render_passages(task['passages'], passages)}\n\n"
            f"Question: {task['question']}"
        )
    elif kind == "json_extract":
        system = EXTRACT_SYSTEM.format(schema=json.dumps(task["schema"]))
        user = f"{render_passages(task['passages'], passages)}\n\nTask: {task['instruction']}"
    elif kind == "explain":
        system = EXPLAIN_SYSTEM.format(limit=task["max_words"], language=language)
        user = f"{render_passages(task['passages'], passages)}\n\n{task['question']}"
    else:
        system, user = TOOL_SYSTEM, task["message"]
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


class MemorySampler(threading.Thread):
    """Samples VmRSS of a process; VmHWM (kernel peak) is read at the end."""

    def __init__(self, pid: int, interval: float = 0.1) -> None:
        super().__init__(daemon=True)
        self.pid, self.interval = pid, interval
        self.peak_rss_kb = 0
        self.stop_event = threading.Event()

    def status_kb(self, field: str) -> int | None:
        try:
            for line in Path(f"/proc/{self.pid}/status").read_text().splitlines():
                if line.startswith(field + ":"):
                    return int(line.split()[1])
        except (OSError, ValueError):
            return None
        return None

    def run(self) -> None:
        while not self.stop_event.is_set():
            rss = self.status_kb("VmRSS")
            if rss:
                self.peak_rss_kb = max(self.peak_rss_kb, rss)
            time.sleep(self.interval)


def http_json(url: str, body: dict[str, Any] | None = None, timeout: float = 600) -> Any:
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def chat(base: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Streamed chat completion; returns text, tool calls, TTFT and server timings."""
    payload = dict(payload, stream=True, timings_per_token=False)
    request = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    first: float | None = None
    text: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    timings: dict[str, Any] = {}
    finish = None
    with urllib.request.urlopen(request, timeout=900) as response:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            timings = chunk.get("timings", timings)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                finish = choice.get("finish_reason") or finish
                got = (
                    delta.get("content")
                    or delta.get("reasoning_content")
                    or delta.get("tool_calls")
                )
                if got and first is None:
                    first = time.perf_counter()
                if delta.get("content"):
                    text.append(delta["content"])
                for call in delta.get("tool_calls") or []:
                    slot = calls.setdefault(
                        call.get("index", 0), {"function": {"name": "", "arguments": ""}}
                    )
                    fn = call.get("function", {})
                    slot["function"]["name"] += fn.get("name") or ""
                    slot["function"]["arguments"] += fn.get("arguments") or ""
    ended = time.perf_counter()
    return {
        "text": "".join(text),
        "tool_calls": [calls[i] for i in sorted(calls)],
        "finish_reason": finish,
        "ttft_s": round((first or ended) - started, 4),
        "total_s": round(ended - started, 4),
        "timings": timings,
    }


def score(task: dict[str, Any], reply: dict[str, Any]) -> dict[str, Any]:
    kind = task["type"]
    if kind == "grounded":
        return scoring.score_grounded(task, reply["text"])
    if kind == "json_extract":
        return scoring.score_json_extract(task, reply["text"])
    if kind == "explain":
        return scoring.score_explain(task, reply["text"])
    return scoring.score_tool_call(task, reply["tool_calls"])


def long_context(task: dict[str, Any], all_ids: list[str]) -> dict[str, Any]:
    """The same question with every passage in the pack, in a fixed per-task order."""
    order = sorted(all_ids, key=lambda pid: hashlib.sha256((task["id"] + pid).encode()).digest())
    return dict(task, passages=order)


def plan(
    tasks: list[dict[str, Any]], all_ids: list[str], sampled_repeats: int
) -> list[tuple[dict[str, Any], str, int]]:
    """(task variant, mode, repeat) triples.

    free: instruction only, greedy. schema: grammar-constrained JSON, greedy.
    long: English grounded tasks with all passages (context handling).
    sampled: English grounded tasks at temperature 0.7 (consistency across runs).
    """
    runs: list[tuple[dict[str, Any], str, int]] = []
    for task in tasks:
        runs.append((task, "free", 0))
        if task["type"] in ("grounded", "json_extract"):
            runs.append((task, "schema", 0))
        if task["type"] == "grounded" and task.get("lang") == "en" and not task.get("dataset"):
            runs.append((long_context(task, all_ids), "long", 0))
            runs.extend((task, "sampled", i + 1) for i in range(sampled_repeats))
    return runs


def run_one(
    base: str,
    task: dict[str, Any],
    mode: str,
    repeat: int,
    passages: dict[str, dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    sampled = mode == "sampled"
    payload: dict[str, Any] = {
        "messages": build_messages(task, passages),
        "temperature": 0.7 if sampled else 0,
        "top_k": 20 if sampled else 1,
        "top_p": 0.8 if sampled else 1.0,
        "seed": 1000 + repeat if sampled else 42,
        "max_tokens": MAX_TOKENS[task["type"]],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if task["type"] == "tool_call":
        payload["tools"] = tools
    if mode == "schema":
        schema = scoring.GROUNDED_SCHEMA if task["type"] == "grounded" else task["schema"]
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "reply", "schema": schema},
        }
    reply = chat(base, payload)
    t = reply["timings"]
    return {
        "id": task["id"],
        "type": task["type"],
        "lang": task.get("lang", "en"),
        "dataset": task.get("dataset", "A"),
        "mode": mode,
        "repeat": repeat,
        "answerable": task.get("answerable"),
        "output": reply["text"],
        "tool_calls": reply["tool_calls"],
        "finish_reason": reply["finish_reason"],
        "ttft_s": reply["ttft_s"],
        "total_s": reply["total_s"],
        "prompt_tokens": t.get("prompt_n"),
        "completion_tokens": t.get("predicted_n"),
        "prefill_tps": _rounded(t.get("prompt_per_second")),
        "decode_tps": _rounded(t.get("predicted_per_second")),
        "metrics": score(task, reply),
    }


def redact(row: dict[str, Any]) -> dict[str, Any]:
    metrics = {k: v for k, v in row["metrics"].items() if k not in ("cited",)}
    return dict(row, output="[redacted: dataset B]", tool_calls=[], metrics=metrics)


def _rounded(value: Any) -> float | None:
    return round(float(value), 2) if value else None


def load_extra(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Dataset B: human-verified tasks over local PDFs (pdf_eval.py). Never committed.

    Only tasks whose review status is "verified" are run; drafts are skipped.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    passages = {p["id"]: p for p in data["passages"]}
    tasks = [dict(t, dataset="B") for t in data["tasks"] if t.get("review") == "verified"]
    return tasks, passages


def rate(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [r["metrics"].get(key) for r in rows if r["metrics"].get(key) is not None]
    if not values:
        return None
    return round(sum(float(v) for v in values) / len(values), 3)


def consistency(rows: list[dict[str, Any]]) -> float | None:
    """Mean share of sampled runs whose correctness matches the greedy run, per task."""
    greedy = {r["id"]: r["metrics"]["correct"] for r in rows if r["mode"] == "free"}
    agreement: dict[str, list[bool]] = {}
    for r in rows:
        if r["mode"] == "sampled" and r["id"] in greedy:
            agreement.setdefault(r["id"], []).append(r["metrics"]["correct"] == greedy[r["id"]])
    if not agreement:
        return None
    return round(statistics.mean(sum(v) / len(v) for v in agreement.values()), 3)


def summarise(all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [r for r in all_rows if r.get("dataset", "A") == "A"]

    def pick(**match: Any) -> list[dict[str, Any]]:
        return [r for r in rows if all(r.get(k) == v for k, v in match.items())]

    summary: dict[str, Any] = {}
    extra = [r for r in all_rows if r.get("dataset") == "B"]
    if extra:
        summary["dataset_b"] = {
            "rows": len(extra),
            "correct": _share(extra, lambda m: m["correct"]),
            "faithful": _share(extra, lambda m: m["faithful"]),
        }
    summary["sampled_consistency"] = consistency(rows)
    for mode in ("free", "schema", "long", "sampled"):
        g = [r for r in pick(type="grounded", mode=mode) if r["lang"] == "en"]
        answerable = [r for r in g if r["answerable"]]
        unanswerable = [r for r in g if not r["answerable"]]
        json_rows = pick(mode=mode, type="grounded") + pick(mode=mode, type="json_extract")
        summary[mode] = {
            "json_strict_rate": _share(json_rows, lambda m: m["json"] == "strict"),
            "json_parseable_rate": _share(json_rows, lambda m: m["json"] != "invalid"),
            "schema_ok_rate": _share(json_rows, lambda m: m.get("schema_ok") is True),
            "extract_correct_rate": _share(
                pick(mode=mode, type="json_extract"), lambda m: m["correct"]
            ),
            "en_answer_accuracy": _share(answerable, lambda m: m["correct"]),
            "en_faithful_rate": _share(answerable, lambda m: m["faithful"]),
            "en_citation_precision": rate(answerable, "citation_precision"),
            "en_false_abstention_rate": _share(
                answerable, lambda m: m.get("false_abstention") is True
            ),
            "en_abstention_rate": _share(unanswerable, lambda m: m["correct"]),
            "en_hallucination_rate": _share(unanswerable, lambda m: m.get("hallucinated") is True),
        }
        for lang in ("yo", "ha", "ig"):
            lr = [r for r in pick(type="grounded", mode=mode) if r["lang"] == lang]
            summary[mode][f"{lang}_grounded_correct"] = _share(lr, lambda m: m["correct"])
    tools = pick(type="tool_call")
    explains = pick(type="explain")
    long_ttft = [r["ttft_s"] for r in rows if r["mode"] == "long"]
    long_prompt = [r["prompt_tokens"] for r in rows if r["mode"] == "long" and r["prompt_tokens"]]
    summary["long_ttft_median_s"] = round(statistics.median(long_ttft), 3) if long_ttft else None
    summary["long_prompt_tokens_median"] = statistics.median(long_prompt) if long_prompt else None
    rows = [r for r in rows if r["mode"] in ("free", "schema")]  # speed: comparable prompts
    summary["tool_call_accuracy"] = _share(tools, lambda m: m["correct"])
    summary["tool_name_accuracy"] = _share(tools, lambda m: m["name_ok"])
    summary["en_explain_term_recall"] = rate(
        [r for r in explains if r["lang"] == "en"], "term_recall"
    )
    for lang in ("yo", "ha", "ig"):
        lr = [r for r in explains if r["lang"] == lang]
        summary[f"{lang}_explain_language_share"] = rate(lr, "language_share")
        summary[f"{lang}_explain_term_recall"] = rate(lr, "term_recall")
    ttft = [r["ttft_s"] for r in rows if r["type"] != "tool_call"]
    decode = [r["decode_tps"] for r in rows if r.get("decode_tps")]
    prefill = [r["prefill_tps"] for r in rows if r.get("prefill_tps")]
    summary["ttft_median_s"] = round(statistics.median(ttft), 3) if ttft else None
    summary["ttft_p90_s"] = round(_quantile(ttft, 0.9), 3) if ttft else None
    summary["decode_tps_median"] = round(statistics.median(decode), 2) if decode else None
    summary["prefill_tps_median"] = round(statistics.median(prefill), 2) if prefill else None
    summary["hit_token_limit"] = sum(1 for r in rows if r.get("finish_reason") == "length")
    return summary


def _share(rows: list[dict[str, Any]], test: Any) -> float | None:
    if not rows:
        return None
    return round(sum(1 for r in rows if test(r["metrics"])) / len(rows), 3)


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(q * len(ordered)))]


def hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"machine": platform.machine(), "system": platform.system()}
    info["kernel"] = platform.release()
    try:
        cpuinfo = Path("/proc/cpuinfo").read_text()
        names = [
            ln.split(":", 1)[1].strip()
            for ln in cpuinfo.splitlines()
            if ":" in ln and (ln.startswith("model name") or ln.startswith("Hardware"))
        ]
        info["cpu"] = names[0] if names else None
        info["logical_cpus"] = os.cpu_count()
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                info["mem_total_mb"] = int(line.split()[1]) // 1024
    except OSError:
        pass
    return info


def run_llama_bench(bin_dir: Path, gguf: Path, threads: int, reps: int) -> dict[str, Any]:
    command = [
        str(bin_dir / "llama-bench"), "-m", str(gguf), "-t", str(threads),
        "-p", "512", "-n", "128", "-r", str(reps), "-o", "json",
    ]  # fmt: skip
    output = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    results = {}
    for row in json.loads(output):
        test = f"pp{row['n_prompt']}" if row["n_prompt"] else f"tg{row['n_gen']}"
        results[test] = {"avg_ts": round(row["avg_ts"], 2), "stddev_ts": round(row["stddev_ts"], 2)}
        results["build"] = row.get("build_commit")
        results["backend"] = row.get("backends") or row.get("backend")
    return results


def start_server(
    args: argparse.Namespace,
) -> tuple[subprocess.Popen[bytes], float, BinaryIO]:
    """Start llama-server; cold start = spawn until /health reports the model loaded."""
    command = [
        str(args.bin_dir / "llama-server"), "-m", str(args.gguf), "-t", str(args.threads),
        "-c", str(args.ctx), "--port", str(args.port), "--host", "127.0.0.1", "-np", "1",
        "--jinja", "--no-webui", "--seed", "42",
    ] + args.server_arg  # fmt: skip
    started = time.perf_counter()
    log = open(args.out.with_suffix(".server.log"), "wb")
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{args.port}"
    deadline = started + args.load_timeout
    while time.perf_counter() < deadline:
        if process.poll() is not None:
            log.close()
            raise RuntimeError(f"llama-server exited with {process.returncode}; see log")
        try:
            if http_json(f"{base}/health", timeout=2).get("status") == "ok":
                return process, time.perf_counter() - started, log
        except (urllib.error.URLError, ConnectionError, TimeoutError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    process.kill()
    log.close()
    raise RuntimeError(f"llama-server did not load within {args.load_timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--gguf", type=Path, required=True)
    parser.add_argument("--bin-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--reps", type=int, default=3, help="llama-bench repetitions")
    parser.add_argument("--only", default="", help="comma-separated task ids")
    parser.add_argument("--skip-llama-bench", action="store_true")
    parser.add_argument("--load-timeout", type=float, default=600)
    parser.add_argument("--sampled-repeats", type=int, default=3, help="consistency runs")
    parser.add_argument("--extra-tasks", type=Path, help="local dataset B file (pdf_eval.py)")
    parser.add_argument("--server-arg", action="append", default=[], help="extra server flag")
    parser.add_argument("--label", default="desktop", help="device or host label for the report")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    passages = {p["id"]: p for p in load_jsonl(TASKS / "passages.jsonl")}
    tasks = load_jsonl(TASKS / "tasks.jsonl")
    tools = json.loads((TASKS / "tools.json").read_text(encoding="utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        tasks = [t for t in tasks if t["id"] in wanted]
    pack_ids = sorted(passages)
    if args.extra_tasks:
        extra_tasks, extra_passages = load_extra(args.extra_tasks)
        tasks += extra_tasks
        passages.update(extra_passages)

    process, cold_start, process_log = start_server(args)
    sampler = MemorySampler(process.pid)
    sampler.start()
    base = f"http://127.0.0.1:{args.port}"
    rss_after_load = sampler.status_kb("VmRSS")
    rows: list[dict[str, Any]] = []
    try:
        props = http_json(f"{base}/props")
        for task, mode, repeat in plan(tasks, pack_ids, args.sampled_repeats):
            rows.append(run_one(base, task, mode, repeat, passages, tools))
            row = rows[-1]
            print(f"{task['id']:4} {mode:7} ok={row['metrics'].get('correct')} "
                  f"ttft={row['ttft_s']}s decode={row['decode_tps']} tok/s", flush=True)  # fmt: skip
        peak_hwm = sampler.status_kb("VmHWM")
    finally:
        sampler.stop_event.set()
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
        process_log.close()

    speed = (
        {}
        if args.skip_llama_bench
        else run_llama_bench(args.bin_dir, args.gguf, args.threads, args.reps)
    )
    result = {
        "schema": 1,
        "model_id": args.model_id,
        "file": args.gguf.name,
        "download_bytes": args.gguf.stat().st_size,
        "runtime": "llama.cpp llama-server",
        "label": args.label,
        "hardware": hardware(),
        "settings": {
            "threads": args.threads,
            "ctx": args.ctx,
            "temperature": 0,
            "thinking": "disabled via chat_template_kwargs",
            "sampled_repeats": args.sampled_repeats,
            "server_args": args.server_arg,
            "server_build": props.get("build_info"),
            "chat_template_has_tools": "tools" in (props.get("chat_template") or ""),
        },
        "memory": {
            "rss_after_load_mb": round((rss_after_load or 0) / 1024, 1),
            "peak_rss_sampled_mb": round(sampler.peak_rss_kb / 1024, 1),
            "peak_hwm_mb": round((peak_hwm or 0) / 1024, 1),
        },
        "cold_start_s": round(cold_start, 3),
        "llama_bench": speed,
        "summary": summarise(rows),
        # Dataset B text comes from local PDFs that must not leave the machine: keep scores only.
        "tasks": [r if r["dataset"] == "A" else redact(r) for r in rows],
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("memory", "cold_start_s", "llama_bench")}, indent=1))
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

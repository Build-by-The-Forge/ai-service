"""AI-12 runner for Google AI Edge LiteRT-LM (.litertlm models, e.g. Gemma 4 E2B).

Same task set, plan, scorers and result format as bench.py, so the report can show
both runtimes. The runtimes differ (engine, kernels, quantisation, chat template),
so rows from the two are compared as different deployment paths, not as the same
model measured twice.

    python benchmarks/offline/litert_bench.py --model-id gemma-4-e2b-litertlm \
        --model /models/gemma-4-E2B-it.litertlm --out runs/desktop/gemma-4-e2b-litertlm.json

Needs `pip install litert-lm-api==0.19.2` (Linux x86_64/aarch64, macOS, Windows,
Android wheels exist).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench  # noqa: E402
import scoring  # noqa: E402

TASKS = bench.TASKS


def spec_tool_class(lm: Any) -> Any:
    """A litert_lm.Tool that only exposes a tools.json description (nothing is executed)."""

    class SpecTool(lm.Tool):
        def __init__(self, spec: dict[str, Any]) -> None:
            self.spec = spec

        def get_tool_description(self) -> dict[str, Any]:
            return self.spec

        def execute(self, param: Any) -> Any:  # never called: automatic_tool_calling=False
            return {"error": "not executed in the benchmark"}

    return SpecTool


def message_text(message: Any) -> str:
    content = dict(message).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(c.get("text", "") for c in content if isinstance(c, dict))
    return ""


def peak_rss_mb() -> float | None:
    """Peak RSS of this process (KiB from ru_maxrss on Linux and Android)."""
    try:
        import resource
    except ImportError:  # Windows
        return None
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


def run_one(
    lm: Any,
    engine: Any,
    task: dict[str, Any],
    mode: str,
    repeat: int,
    passages: dict[str, dict[str, Any]],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    messages = bench.build_messages(task, passages)
    sampled = mode == "sampled"
    sampler = (
        lm.SamplerConfig(top_k=20, top_p=0.8, temperature=0.7, seed=1000 + repeat)
        if sampled
        else lm.SamplerConfig(top_k=1, top_p=1.0, temperature=1.0, seed=42)
    )
    constrained = None
    kwargs: dict[str, Any] = {}
    if mode == "schema":
        schema = scoring.GROUNDED_SCHEMA if task["type"] == "grounded" else task["schema"]
        kwargs["response_format"] = lm.ResponseFormat.json(schema)
        constrained = lm.ConstrainedDecodingConfig(
            enable=True, provider=lm.LiteRtLmConstraintProviderType.LL_GUIDANCE
        )
    error = None
    conversation = None
    try:
        conversation = engine.create_conversation(
            messages=[messages[0]],
            tools=(
                [spec_tool_class(lm)(t) for t in tools] if task["type"] == "tool_call" else None
            ),
            automatic_tool_calling=False,
            sampler_config=sampler,
            max_output_tokens=bench.MAX_TOKENS[task["type"]],
            thinking_config=lm.ThinkingConfig(enable_thinking=False),
            constrained_decoding_config=constrained,
        )
    except Exception as exc:  # a runtime refusal is a result, not a crash
        error = f"{type(exc).__name__}: {exc}"
    started = time.perf_counter()
    first: float | None = None
    text: list[str] = []
    calls: list[dict[str, Any]] = []
    try:
        if conversation is None:
            raise RuntimeError(error)
        for chunk in conversation.send_message_async(messages[1]["content"], **kwargs):
            piece = message_text(chunk)
            if (piece or chunk.tool_calls) and first is None:
                first = time.perf_counter()
            text.append(piece)
            for call in chunk.tool_calls:
                spec = call.to_json()
                spec["function"]["arguments"] = json.dumps(spec["function"]["arguments"])
                calls.append(spec)
    except Exception as exc:  # a runtime refusal is a result, not a crash
        error = error or f"{type(exc).__name__}: {exc}"
    ended = time.perf_counter()
    info = (
        conversation.get_benchmark_info()
        if conversation is not None
        else SimpleNamespace(
            last_prefill_token_count=None,
            last_prefill_tokens_per_second=None,
            last_decode_token_count=None,
            last_decode_tokens_per_second=None,
        )
    )
    if conversation is not None:
        conversation.close()
    reply = {"text": "".join(text), "tool_calls": calls}
    return {
        "id": task["id"],
        "type": task["type"],
        "lang": task.get("lang", "en"),
        "dataset": task.get("dataset", "A"),
        "mode": mode,
        "repeat": repeat,
        "answerable": task.get("answerable"),
        "output": reply["text"],
        "tool_calls": calls,
        "finish_reason": None,
        "error": error,
        "ttft_s": round((first or ended) - started, 4),
        "total_s": round(ended - started, 4),
        "prompt_tokens": info.last_prefill_token_count,
        "completion_tokens": info.last_decode_token_count,
        "prefill_tps": bench._rounded(info.last_prefill_tokens_per_second),
        "decode_tps": bench._rounded(info.last_decode_tokens_per_second),
        "metrics": bench.score(task, reply),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--sampled-repeats", type=int, default=3)
    parser.add_argument("--only", default="")
    parser.add_argument("--label", default="desktop")
    parser.add_argument("--cache-dir", default=None, help="compiled-artifact cache")
    args = parser.parse_args()
    import litert_lm as lm

    passages = {p["id"]: p for p in bench.load_jsonl(TASKS / "passages.jsonl")}
    tasks = bench.load_jsonl(TASKS / "tasks.jsonl")
    tools = json.loads((TASKS / "tools.json").read_text(encoding="utf-8"))
    if args.only:
        tasks = [t for t in tasks if t["id"] in set(args.only.split(","))]
    backend = (
        lm.Backend.CPU(thread_count=args.threads) if args.backend == "cpu" else lm.Backend.GPU()
    )
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)  # the runtime requires it to exist
    rss_before = peak_rss_mb()
    started = time.perf_counter()
    engine = lm.Engine(
        str(args.model),
        backend=backend,
        max_num_tokens=args.ctx,
        cache_dir=args.cache_dir,
        enable_benchmark=True,
    )
    cold_start = time.perf_counter() - started
    rss_after_load = peak_rss_mb()
    rows = []
    with engine:
        for task, mode, repeat in bench.plan(tasks, sorted(passages), args.sampled_repeats):
            rows.append(run_one(lm, engine, task, mode, repeat, passages, tools))
            r = rows[-1]
            print(f"{task['id']:4} {mode:7} ok={r['metrics'].get('correct')} ttft={r['ttft_s']}s "
                  f"decode={r['decode_tps']} tok/s {r['error'] or ''}", flush=True)  # fmt: skip
    result = {
        "schema": 1,
        "model_id": args.model_id,
        "file": args.model.name,
        "download_bytes": args.model.stat().st_size,
        "runtime": f"LiteRT-LM {args.backend}",
        "label": args.label,
        "hardware": bench.hardware(),
        "settings": {
            "threads": args.threads,
            "ctx": args.ctx,
            "temperature": "greedy (top_k=1)",
            "thinking": "disabled via ThinkingConfig",
            "sampled_repeats": args.sampled_repeats,
            "litert_lm_api": getattr(lm, "__version__", "0.19.2"),
        },
        "memory": {
            "python_baseline_mb": rss_before,
            "rss_after_load_mb": rss_after_load,
            # Peak of the whole Python process (runtime + harness); LiteRT-LM runs
            # in-process, unlike llama-server.
            "peak_hwm_mb": peak_rss_mb(),
        },
        "cold_start_s": round(cold_start, 3),
        "llama_bench": {},
        "summary": bench.summarise(rows),
        "tasks": [r if r["dataset"] == "A" else bench.redact(r) for r in rows],
        "errors": sum(1 for r in rows if r["error"]),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("memory", "cold_start_s", "errors")}, indent=1))
    print(json.dumps(result["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

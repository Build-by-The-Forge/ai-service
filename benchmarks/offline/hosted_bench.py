"""Hosted-provider mode: the AI-12 task set against an OpenAI-compatible hosted API.

Built for the future provider evaluation (Fireworks, Gemini's OpenAI-compatible
endpoint, ...). It is safe by construction and self-tested against a fake server only.

    # Dry run (the default): prints the plan and the worst-case cost, sends nothing.
    python benchmarks/offline/hosted_bench.py --provider-id fireworks-x \
        --model some/model --base-url https://api.example.com/v1 \
        --prices prices.json --out runs/hosted/x.json

    # A real run needs ALL of these, and a verified price table:
    #   --confirm-spend --max-usd 2.00 --max-requests 150 --max-tokens-per-request 512
    #   --key-env PROVIDER_API_KEY   (or --key-file /path/outside/the/repo)

Guarantees (each has a test in tests/unit/test_offline_benchmark_hosted.py):
- Nothing is sent without --confirm-spend and all three caps (USD, requests, tokens).
- A price table that is not marked verified (with source and date) blocks a real run.
  Prices are an input file, never hardcoded.
- The run stops the moment the next call could exceed a cap, and records why.
- Cost is estimated before every call (input tokens over-estimated, output at the cap).
- The API key comes from an environment variable or a file outside the repository,
  is never printed, logged or written to a result, and is scrubbed from error text.
- Only datasets marked synthetic are sent. Dataset B (real PDFs) is refused.
- Real runs need https; plain http is allowed only for localhost (the fake server).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bench  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
CHARS_PER_TOKEN = 3.0  # deliberately pessimistic (English averages about 4)


class Refusal(Exception):
    """A safety rule stopped the run before anything was sent."""


def load_dataset_marker(tasks_dir: Path) -> dict[str, Any]:
    marker = tasks_dir / "DATASET.json"
    if not marker.exists():
        raise Refusal(f"{tasks_dir} has no DATASET.json; only marked synthetic data may be sent")
    data = json.loads(marker.read_text(encoding="utf-8"))
    if data.get("synthetic") is not True or data.get("third_party_material") is not False:
        raise Refusal("dataset is not marked synthetic with no third-party material")
    return data


def load_prices(path: Path | None, model: str) -> dict[str, Any]:
    if path is None:
        raise Refusal("--prices is required (a price table is an input, never hardcoded)")
    table = json.loads(path.read_text(encoding="utf-8"))
    entry = (table.get("models") or {}).get(model)
    if not entry or entry.get("input_per_mtok") is None or entry.get("output_per_mtok") is None:
        raise Refusal(f"price table has no input/output prices for {model}")
    return {
        "verified": table.get("verified") is True
        and bool(table.get("source"))
        and bool(table.get("verified_on")),
        "source": table.get("source"),
        "verified_on": table.get("verified_on"),
        "input_per_mtok": float(entry["input_per_mtok"]),
        "output_per_mtok": float(entry["output_per_mtok"]),
    }


def read_key(env_name: str | None, key_file: Path | None) -> str:
    if bool(env_name) == bool(key_file):
        raise Refusal("give the key through exactly one of --key-env or --key-file")
    if env_name:
        key = os.environ.get(env_name, "")
    else:
        assert key_file is not None
        resolved = key_file.resolve()
        if REPO_ROOT in resolved.parents:
            raise Refusal("the key file must be outside the repository")
        key = resolved.read_text(encoding="utf-8").strip()
    if not key:
        raise Refusal("the API key is empty or not set")
    return key


def check_base_url(url: str, real: bool) -> None:
    parsed = urllib.parse.urlparse(url)
    local = parsed.hostname in LOCAL_HOSTS
    if parsed.scheme != "https" and not local:
        raise Refusal("a hosted provider must be reached over https")
    if real and local and not os.environ.get("AI12_ALLOW_LOCAL_REAL_RUN"):
        raise Refusal("a real run against localhost is for tests only")


def estimate_cost(prompt_chars: int, max_out: int, prices: dict[str, Any]) -> float:
    """Worst-case cost of one call: pessimistic input tokens, output at the cap."""
    tokens_in = prompt_chars / CHARS_PER_TOKEN
    return (tokens_in * prices["input_per_mtok"] + max_out * prices["output_per_mtok"]) / 1e6


def scrub(text: str, key: str | None) -> str:
    return text.replace(key, "[redacted-key]") if key else text


def post(
    base_url: str, key: str, payload: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body, time.perf_counter() - started


def build_payload(
    model: str,
    task: dict[str, Any],
    mode: str,
    passages: dict[str, Any],
    max_tokens: int,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": bench.build_messages(task, passages),
        "temperature": 0,
        "max_tokens": min(bench.MAX_TOKENS[task["type"]], max_tokens),
    }
    if task["type"] == "tool_call":
        payload["tools"] = tools
    if mode == "schema":
        schema = bench.scoring.GROUNDED_SCHEMA if task["type"] == "grounded" else task["schema"]
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": "reply", "schema": schema},
        }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--provider-id", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--prices", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tasks-dir", type=Path, default=bench.TASKS)
    parser.add_argument("--modes", default="free,schema", help="free, schema, long")
    parser.add_argument("--only", default="")
    parser.add_argument("--confirm-spend", action="store_true")
    parser.add_argument("--max-usd", type=float)
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--max-tokens-per-request", type=int)
    parser.add_argument("--key-env")
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args(argv)

    try:
        marker = load_dataset_marker(args.tasks_dir)
        real = args.confirm_spend
        check_base_url(args.base_url, real)
        prices = load_prices(args.prices, args.model)
        caps = (args.max_usd, args.max_requests, args.max_tokens_per_request)
        if real:
            if any(c is None or c <= 0 for c in caps):
                raise Refusal(
                    "a real run needs --max-usd, --max-requests and "
                    "--max-tokens-per-request, all above zero"
                )
            if not prices["verified"]:
                raise Refusal(
                    "the price table is not marked verified (verified, source, "
                    "verified_on); a real run is blocked"
                )
            key: str | None = read_key(args.key_env, args.key_file)
        else:
            key = None
    except Refusal as refusal:
        print(f"REFUSED: {refusal}", file=sys.stderr)
        return 2

    passages = {p["id"]: p for p in bench.load_jsonl(args.tasks_dir / "passages.jsonl")}
    tasks = bench.load_jsonl(args.tasks_dir / "tasks.jsonl")
    tools = json.loads((args.tasks_dir / "tools.json").read_text(encoding="utf-8"))
    if args.only:
        tasks = [t for t in tasks if t["id"] in set(args.only.split(","))]
    modes = set(args.modes.split(","))
    steps = [s for s in bench.plan(tasks, sorted(passages), 0) if s[1] in modes]
    cap_tokens = args.max_tokens_per_request or 512
    estimates = [
        estimate_cost(
            sum(len(m["content"]) for m in bench.build_messages(t, passages)),
            min(bench.MAX_TOKENS[t["type"]], cap_tokens),
            prices,
        )
        for t, _, _ in steps
    ]
    print(
        f"{len(steps)} requests planned; worst-case cost ${sum(estimates):.4f} "
        f"(prices verified: {prices['verified']})",
        flush=True,
    )
    if not real:
        print("DRY RUN: nothing was sent. Add --confirm-spend and the caps for a real run.")
        return 0

    assert key is not None
    spent = 0.0
    rows: list[dict[str, Any]] = []
    stop_reason = "completed"
    for (task, mode, repeat), estimate in zip(steps, estimates):
        if len(rows) >= args.max_requests:
            stop_reason = f"max_requests ({args.max_requests}) reached"
            break
        if spent + estimate > args.max_usd:
            stop_reason = (
                f"max_usd (${args.max_usd}) would be exceeded by the next call "
                f"(spent ${spent:.4f} + worst case ${estimate:.4f})"
            )
            break
        payload = build_payload(args.model, task, mode, passages, cap_tokens, tools)
        error = None
        reply: dict[str, Any] = {"text": "", "tool_calls": []}
        usage: dict[str, Any] = {}
        latency = 0.0
        try:
            body, latency = post(args.base_url, key, payload, args.timeout)
            message = body["choices"][0]["message"]
            reply = {
                "text": message.get("content") or "",
                "tool_calls": message.get("tool_calls") or [],
            }
            usage = body.get("usage") or {}
        except Exception as exc:  # a failed call still costs a request and is recorded
            error = scrub(f"{type(exc).__name__}: {exc}", key)[:300]
        tokens_in = usage.get("prompt_tokens")
        tokens_out = usage.get("completion_tokens")
        if tokens_in is None or tokens_out is None:
            cost, cost_basis = estimate, "worst-case estimate (no usage reported)"
        else:
            cost = (
                tokens_in * prices["input_per_mtok"] + tokens_out * prices["output_per_mtok"]
            ) / 1e6
            cost_basis = "reported usage"
        spent += cost
        metrics = bench.score(task, reply)
        rows.append(
            {
                "id": task["id"],
                "type": task["type"],
                "lang": task.get("lang", "en"),
                "mode": mode,
                "answerable": task.get("answerable"),
                "model": args.model,
                "prompt_tokens": tokens_in,
                "completion_tokens": tokens_out,
                "cost_usd": round(cost, 6),
                "cost_basis": cost_basis,
                "latency_s": round(latency, 3),
                "error": error,
                "ttft_s": round(latency, 3),
                "total_s": round(latency, 3),
                "decode_tps": None,
                "prefill_tps": None,
                "finish_reason": None,
                "schema_valid": metrics.get("schema_ok"),
                "citation_faithful": metrics.get("faithful"),
                "abstained": metrics.get("answerable_pred") is False,
                "correct": metrics.get("correct"),
                "output": scrub(reply["text"], key),
                "metrics": metrics,
            }
        )
    result = {
        "schema": 1,
        "provider_id": args.provider_id,
        "model": args.model,
        "dataset": marker,
        "prices": {k: v for k, v in prices.items() if k != "verified"} | {"verified": True},
        "caps": {
            "max_usd": args.max_usd,
            "max_requests": args.max_requests,
            "max_tokens_per_request": cap_tokens,
        },
        "requests": len(rows),
        "spent_usd": round(spent, 6),
        "stop_reason": stop_reason,
        "summary": bench.summarise([dict(r, dataset="A") for r in rows]),
        "tasks": rows,
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(rows)} requests, ${spent:.4f} spent, stopped: {stop_reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

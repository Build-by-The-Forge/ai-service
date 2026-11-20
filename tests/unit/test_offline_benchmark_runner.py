"""benchmarks/offline/bench.py against a fake llama-server (no model, no download)."""

import importlib.util
import json
import os
import socket
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks" / "offline"
sys.path.insert(0, str(BENCH))
_spec = importlib.util.spec_from_file_location("offline_bench", BENCH / "bench.py")
bench = importlib.util.module_from_spec(_spec)
sys.modules["offline_bench"] = bench
_spec.loader.exec_module(bench)

FAKE_SERVER = textwrap.dedent('''
    #!/usr/bin/env python3
    """Speaks the slice of the llama-server API that bench.py uses."""
    import json, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer

    port = int(sys.argv[sys.argv.index("--port") + 1])
    TIMINGS = {"prompt_n": 50, "predicted_n": 10,
               "prompt_per_second": 500.0, "predicted_per_second": 25.0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_json(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self.send_json({"status": "ok"})
            else:
                self.send_json({"build_info": "fake", "chat_template": "{{ tools }}"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if body.get("tools"):
                delta = {"tool_calls": [{"index": 0, "function": {
                    "name": "calculate", "arguments": "{\\"expression\\": \\"0.75*120+18\\"}"}}]}
            else:
                # Always abstains: right on unanswerable tasks, wrong on answerable ones.
                delta = {"content": json.dumps(
                    {"answer": "", "citations": [], "answerable": False})}
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for chunk in (
                {"choices": [{"delta": delta, "finish_reason": None}]},
                {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": TIMINGS},
            ):
                self.wfile.write(("data: " + json.dumps(chunk) + "\\n\\n").encode())
            self.wfile.write(b"data: [DONE]\\n\\n")

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
''').lstrip()

FAKE_BENCH = textwrap.dedent("""
    #!/usr/bin/env python3
    import json
    print(json.dumps([
        {"n_prompt": 512, "n_gen": 0, "avg_ts": 100.0, "stddev_ts": 1.0, "build_commit": "x"},
        {"n_prompt": 0, "n_gen": 128, "avg_ts": 20.0, "stddev_ts": 0.5, "build_commit": "x"},
    ]))
""").lstrip()


def task(task_id):
    return next(t for t in bench.load_jsonl(BENCH / "tasks" / "tasks.jsonl") if t["id"] == task_id)


def test_plan_adds_long_and_sampled_runs_only_to_english_grounded_tasks():
    runs = bench.plan([task("G01"), task("NY1"), task("T01"), task("J01")], ["P01", "P02"], 2)
    modes = [(t["id"], mode) for t, mode, _ in runs]
    assert modes == [
        ("G01", "free"), ("G01", "schema"), ("G01", "long"), ("G01", "sampled"),
        ("G01", "sampled"), ("NY1", "free"), ("NY1", "schema"), ("T01", "free"),
        ("J01", "free"), ("J01", "schema"),
    ]  # fmt: skip
    long_variant = runs[2][0]
    assert sorted(long_variant["passages"]) == ["P01", "P02"]


def test_dataset_b_tasks_get_no_long_or_sampled_runs():
    runs = bench.plan([dict(task("G01"), dataset="B")], ["P01"], 3)
    assert [mode for _, mode, _ in runs] == ["free", "schema"]


def row(task_id, mode, correct, dataset="A"):
    return {"id": task_id, "mode": mode, "dataset": dataset, "metrics": {"correct": correct}}


def test_consistency_compares_sampled_runs_with_the_greedy_run():
    rows = [row("G01", "free", True), row("G01", "sampled", True), row("G01", "sampled", False)]
    rows += [row("G02", "free", False), row("G02", "sampled", False)]
    assert bench.consistency(rows) == 0.75


def test_redact_drops_text_but_keeps_scores():
    r = dict(row("B01", "free", True, "B"), output="private page text", tool_calls=[1])
    r["metrics"]["cited"] = ["DOC-p3"]
    out = bench.redact(r)
    assert out["output"] == "[redacted: dataset B]" and out["tool_calls"] == []
    assert out["metrics"] == {"correct": True}


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.skipif(os.name == "nt", reason="needs /proc and executable scripts (Linux CI)")
def test_end_to_end_against_fake_server(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("llama-server", FAKE_SERVER), ("llama-bench", FAKE_BENCH)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    gguf = tmp_path / "fake.gguf"
    gguf.write_bytes(b"GGUF" + bytes(1000))
    out = tmp_path / "run.json"
    subprocess.run(
        [
            sys.executable, str(BENCH / "bench.py"), "--model-id", "fake", "--gguf", str(gguf),
            "--bin-dir", str(bin_dir), "--out", str(out), "--port", str(free_port()),
            "--sampled-repeats", "1", "--reps", "1", "--load-timeout", "30",
        ],
        check=True, capture_output=True, timeout=300,
    )  # fmt: skip
    result = json.loads(out.read_text())
    s = result["summary"]
    assert result["download_bytes"] == 1004
    assert result["memory"]["peak_hwm_mb"] > 0
    assert result["llama_bench"]["tg128"]["avg_ts"] == 20.0
    # The fake always abstains.
    assert s["free"]["en_abstention_rate"] == 1.0
    assert s["free"]["en_answer_accuracy"] == 0.0
    assert s["free"]["en_false_abstention_rate"] == 1.0
    assert s["free"]["json_strict_rate"] > 0
    assert s["sampled_consistency"] == 1.0
    assert s["decode_tps_median"] == 25.0
    assert result["reload_warm_s"] > 0 and result["unload_s"] >= 0
    # Only T03 expects calculate(0.75*120+18); T05 expects no call.
    assert s["tool_call_accuracy"] == round(1 / 6, 3)

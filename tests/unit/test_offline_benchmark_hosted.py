"""benchmarks/offline/hosted_bench.py: spend caps, key handling and refusals.

Nothing here touches a real provider: a fake OpenAI-compatible server runs on
localhost, and every other host is blocked at the urllib level.
"""

import importlib.util
import json
import shutil
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "benchmarks" / "offline"
sys.path.insert(0, str(BENCH))
_spec = importlib.util.spec_from_file_location("offline_hosted", BENCH / "hosted_bench.py")
hosted = importlib.util.module_from_spec(_spec)
sys.modules["offline_hosted"] = hosted
_spec.loader.exec_module(hosted)

KEY = "sk-test-SECRET-7f3a9c"
MODEL = "vendor/model-x"


@pytest.fixture(autouse=True)
def no_real_network(monkeypatch):
    """Any request to a host other than localhost fails the test."""
    real = urllib.request.urlopen

    def guarded(request, *args, **kwargs):
        host = urlparse(getattr(request, "full_url", request)).hostname
        assert host in ("127.0.0.1", "localhost"), f"network access to {host}"
        return real(request, *args, **kwargs)

    monkeypatch.setattr(urllib.request, "urlopen", guarded)


class Fake:
    def __init__(self, echo_key=False):
        self.requests = []
        self.auth = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.requests.append(body)
                outer.auth.append(self.headers.get("Authorization"))
                text = {"answer": "", "citations": [], "answerable": False}
                content = json.dumps(text)
                if outer.echo_key:
                    content = "echo " + self.headers["Authorization"].split()[-1]
                data = json.dumps(
                    {
                        "choices": [{"message": {"content": content}}],
                        "usage": {"prompt_tokens": 100, "completion_tokens": 10},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.echo_key = echo_key
        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake():
    server = Fake()
    yield server
    server.close()


def prices(tmp_path, verified=True):
    path = tmp_path / "prices.json"
    path.write_text(
        json.dumps(
            {
                "verified": verified,
                "source": "https://example.test/pricing" if verified else None,
                "verified_on": "2026-11-20" if verified else None,
                "models": {MODEL: {"input_per_mtok": 1.0, "output_per_mtok": 2.0}},
            }
        )
    )
    return path


def run(tmp_path, fake, *extra, price_path=None, monkeypatch=None):
    out = tmp_path / "out.json"
    argv = [
        "--provider-id", "fake", "--model", MODEL, "--base-url", fake.url,
        "--prices", str(price_path or prices(tmp_path)), "--out", str(out), *extra,
    ]  # fmt: skip
    return hosted.main(argv), out


REAL = ["--confirm-spend", "--max-usd", "5", "--max-requests", "1000",
        "--max-tokens-per-request", "256", "--key-env", "FAKE_KEY"]  # fmt: skip


@pytest.fixture(autouse=True)
def key_env(monkeypatch):
    monkeypatch.setenv("FAKE_KEY", KEY)
    monkeypatch.setenv("AI12_ALLOW_LOCAL_REAL_RUN", "1")


def test_dry_run_is_the_default_and_sends_nothing(tmp_path, fake, capsys):
    code, out = run(tmp_path, fake)
    assert code == 0 and not out.exists() and fake.requests == []
    assert "DRY RUN" in capsys.readouterr().out


def test_a_real_run_is_impossible_without_the_flag(tmp_path, fake):
    caps_only = [a for a in REAL if a != "--confirm-spend"]
    code, _ = run(tmp_path, fake, *caps_only)
    assert code == 0 and fake.requests == []


@pytest.mark.parametrize("missing", ["--max-usd", "--max-requests", "--max-tokens-per-request"])
def test_a_real_run_needs_every_cap(tmp_path, fake, missing, capsys):
    args = list(REAL)
    i = args.index(missing)
    del args[i : i + 2]
    code, _ = run(tmp_path, fake, *args)
    assert code == 2 and fake.requests == []
    assert "all above zero" in capsys.readouterr().err


def test_an_unverified_price_table_blocks_a_real_run(tmp_path, fake, capsys):
    code, _ = run(tmp_path, fake, *REAL, price_path=prices(tmp_path, verified=False))
    assert code == 2 and fake.requests == []
    assert "not marked verified" in capsys.readouterr().err


def test_max_requests_stops_the_run_and_says_why(tmp_path, fake):
    args = [a if a != "1000" else "3" for a in REAL]
    code, out = run(tmp_path, fake, *args)
    result = json.loads(out.read_text())
    assert code == 0 and len(fake.requests) == 3 and result["requests"] == 3
    assert "max_requests" in result["stop_reason"]


def test_max_usd_stops_before_the_cap_is_exceeded(tmp_path, fake):
    # Each call costs 100 in + 10 out tokens = $0.00012; a worst case (about $0.0005) fits
    # three or four calls under $0.0015, never more.
    args = [a if a != "5" else "0.0015" for a in REAL]
    code, out = run(tmp_path, fake, *args)
    result = json.loads(out.read_text())
    assert code == 0 and 1 <= result["requests"] < 10
    assert result["spent_usd"] <= 0.0015
    assert "max_usd" in result["stop_reason"]


def test_result_records_tokens_cost_latency_and_quality(tmp_path, fake):
    code, out = run(tmp_path, fake, *REAL, "--only", "G01,A01")
    result = json.loads(out.read_text())
    row = result["tasks"][0]
    assert result["stop_reason"] == "completed" and result["requests"] == 4
    assert row["model"] == MODEL and row["prompt_tokens"] == 100 and row["completion_tokens"] == 10
    assert row["cost_usd"] == pytest.approx((100 * 1.0 + 10 * 2.0) / 1e6)
    assert {"latency_s", "schema_valid", "citation_faithful", "abstained", "correct"} <= set(row)
    assert result["prices"]["verified"] is True and result["caps"]["max_usd"] == 5


def test_the_key_is_sent_as_a_header_but_never_appears_anywhere_else(tmp_path, capsys):
    server = Fake(echo_key=True)  # the provider echoes the key back in its reply
    try:
        code, out = run(tmp_path, server, *REAL, "--only", "G01")
    finally:
        server.close()
    assert code == 0 and server.auth[0] == f"Bearer {KEY}"
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err
    assert KEY not in out.read_text()
    assert "[redacted-key]" in out.read_text()


def test_key_file_inside_the_repository_is_refused(tmp_path, fake, capsys):
    inside = ROOT / "benchmarks" / "offline" / "local" / "key.txt"
    inside.parent.mkdir(exist_ok=True)
    inside.write_text(KEY)
    try:
        args = [a for a in REAL if a not in ("--key-env", "FAKE_KEY")]
        code, _ = run(tmp_path, fake, *args, "--key-file", str(inside))
    finally:
        inside.unlink()
    assert code == 2 and fake.requests == []
    assert "outside the repository" in capsys.readouterr().err


def test_key_file_outside_the_repository_works(tmp_path, fake):
    key_file = tmp_path / "key.txt"
    key_file.write_text(KEY + "\n")
    args = [a for a in REAL if a not in ("--key-env", "FAKE_KEY")]
    code, out = run(tmp_path, fake, *args, "--key-file", str(key_file), "--only", "G01")
    assert code == 0 and fake.auth[0] == f"Bearer {KEY}" and KEY not in out.read_text()


def test_missing_key_is_refused(tmp_path, fake, monkeypatch, capsys):
    monkeypatch.delenv("FAKE_KEY")
    code, _ = run(tmp_path, fake, *REAL)
    assert code == 2 and fake.requests == []


def test_datasets_not_marked_synthetic_are_refused(tmp_path, fake, capsys):
    copy = tmp_path / "tasks"
    shutil.copytree(BENCH / "tasks", copy)
    marker = json.loads((copy / "DATASET.json").read_text())
    marker["third_party_material"] = True
    (copy / "DATASET.json").write_text(json.dumps(marker))
    code, _ = run(tmp_path, fake, *REAL, "--tasks-dir", str(copy))
    assert code == 2 and fake.requests == []
    (copy / "DATASET.json").unlink()
    code, _ = run(tmp_path, fake, *REAL, "--tasks-dir", str(copy))
    assert code == 2 and "no DATASET.json" in capsys.readouterr().err


def test_a_hosted_provider_must_use_https(tmp_path, fake, capsys):
    out = tmp_path / "o.json"
    argv = ["--provider-id", "p", "--model", MODEL, "--base-url", "http://api.example.test/v1",
            "--prices", str(prices(tmp_path)), "--out", str(out)]  # fmt: skip
    assert hosted.main(argv) == 2
    assert "https" in capsys.readouterr().err


def test_a_real_run_against_localhost_needs_the_test_override(tmp_path, fake, monkeypatch):
    monkeypatch.delenv("AI12_ALLOW_LOCAL_REAL_RUN")
    code, _ = run(tmp_path, fake, *REAL)
    assert code == 2 and fake.requests == []


def test_prices_are_an_input_not_hardcoded(tmp_path, fake, capsys):
    argv = ["--provider-id", "p", "--model", MODEL, "--base-url", fake.url,
            "--out", str(tmp_path / "o.json")]  # fmt: skip
    assert hosted.main(argv) == 2
    assert "--prices is required" in capsys.readouterr().err

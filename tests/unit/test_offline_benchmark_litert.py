"""benchmarks/offline/litert_bench.py against a fake litert_lm module (no runtime, no model)."""

import json
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "benchmarks" / "offline" / "litert_bench.py"

FAKE_LITERT = textwrap.dedent('''
    """Just enough of litert_lm 0.19 for litert_bench.py."""
    import json
    from types import SimpleNamespace

    class Tool:
        """Base class: the real runtime only accepts Tool subclasses."""

    class ConstrainedDecodingConfig:
        def __init__(self, enable=True, provider=None):
            self.enable, self.provider = enable, provider

    class LiteRtLmConstraintProviderType:
        LL_GUIDANCE = 1

    class SamplerConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class ThinkingConfig:
        def __init__(self, enable_thinking):
            self.enable_thinking = enable_thinking

    class ResponseFormat:
        @classmethod
        def json(cls, schema):
            return ("json", schema)

    class Backend:
        @staticmethod
        def CPU(thread_count=None):
            return ("cpu", thread_count)

        @staticmethod
        def GPU():
            return ("gpu", None)

    class Call:
        def __init__(self, name, arguments):
            self.name, self.arguments = name, arguments

        def to_json(self):
            return {"type": "function", "function": {"name": self.name, "arguments": self.arguments}}

    class Chunk(dict):
        def __init__(self, text="", tool_calls=()):
            super().__init__(content=[{"type": "text", "text": text}])
            self.tool_calls = list(tool_calls)

    class Conversation:
        def __init__(self, tools, constrained):
            self.tools, self.constrained = tools, constrained

        def send_message_async(self, text, response_format=None):
            # The real runtime rejects response_format without this configuration.
            assert response_format is None or (
                self.constrained and self.constrained.provider == 1)
            if self.tools:
                names = [t.get_tool_description()["function"]["name"] for t in self.tools]
                assert "calculate" in names
                yield Chunk(tool_calls=[Call("calculate", {"expression": "0.75*120+18"})])
                return
            reply = {"answer": "", "citations": [], "answerable": False}
            text = json.dumps(reply)
            yield Chunk(text[:10])
            yield Chunk(text[10:])

        def get_benchmark_info(self):
            return SimpleNamespace(
                last_prefill_token_count=40, last_prefill_tokens_per_second=300.0,
                last_decode_token_count=12, last_decode_tokens_per_second=15.0)

        def close(self):
            pass

    class Engine:
        def __init__(self, path, backend, max_num_tokens, cache_dir, enable_benchmark):
            assert enable_benchmark and backend == ("cpu", 4)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def create_conversation(self, messages, tools, automatic_tool_calling, **kwargs):
            assert automatic_tool_calling is False
            assert messages[0]["role"] == "system"
            assert all(isinstance(t, Tool) for t in tools or [])
            return Conversation(tools, kwargs.get("constrained_decoding_config"))
''').lstrip()


def test_litert_runner_end_to_end_with_fake_runtime(tmp_path):
    (tmp_path / "litert_lm.py").write_text(FAKE_LITERT)
    model = tmp_path / "m.litertlm"
    model.write_bytes(bytes(2048))
    out = tmp_path / "run.json"
    env = {"PYTHONPATH": str(tmp_path), "SYSTEMROOT": "C:\\Windows", "PATH": ""}
    subprocess.run(
        [
            sys.executable, str(RUNNER), "--model-id", "fake-litert", "--model", str(model),
            "--out", str(out), "--sampled-repeats", "1",
        ],
        check=True, capture_output=True, timeout=300, env=env,
    )  # fmt: skip
    result = json.loads(out.read_text(encoding="utf-8"))
    s = result["summary"]
    assert result["runtime"] == "LiteRT-LM cpu"
    assert result["errors"] == 0  # schema mode and tools both reached the fake runtime
    assert result["download_bytes"] == 2048
    assert s["free"]["en_abstention_rate"] == 1.0
    assert s["free"]["json_strict_rate"] > 0  # streamed chunks are joined before parsing
    assert s["decode_tps_median"] == 15.0
    assert s["tool_call_accuracy"] == round(1 / 6, 3)
    assert {r["mode"] for r in result["tasks"]} == {"free", "schema", "long", "sampled"}

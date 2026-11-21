"""The AI-12 benchmark workflow is manual, credential-free and read-only."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "offline-benchmark.yml"
TEXT = WORKFLOW.read_text(encoding="utf-8")


def test_only_triggered_by_hand_or_a_bench_branch():
    on_block = TEXT.split("\non:\n", 1)[1].split("\npermissions:", 1)[0]
    assert "workflow_dispatch:" in on_block
    assert 'push:\n    branches: ["bench/**"]' in on_block
    for trigger in ("pull_request", "schedule:", "workflow_run", "tags:"):
        assert trigger not in on_block


def test_ci_model_list_names_registered_models():
    registry = json.loads((ROOT / "benchmarks/offline/models.json").read_text(encoding="utf-8"))
    ids = {m["id"] for m in registry["models"]}
    lines = (ROOT / "benchmarks/offline/ci_models.txt").read_text(encoding="utf-8").splitlines()
    listed = [x.strip() for x in lines if x.strip() and not x.startswith("#")]
    assert listed and set(listed) <= ids


def test_uses_no_secrets_and_only_reads_the_repository():
    assert "secrets." not in TEXT
    assert "permissions:\n  contents: read" in TEXT
    assert not re.search(r"^\s+(id-token|pull-requests|contents): write", TEXT, re.M)


def test_bash_steps_keep_failures_visible():
    # `shell: bash` gives -e -o pipefail; a bare `run:` would not (see the CI pipe-masking note).
    for step in re.split(r"\n\s+- ", TEXT):
        if re.search(r"^\s+run:", step, re.M):
            assert "shell: bash" in step, step.splitlines()[0]


def test_uploads_results_not_weights():
    upload = TEXT.split("actions/upload-artifact", 1)[1]
    assert ".gguf" not in upload and ".litertlm" not in upload
    assert "runs/desktop/${{ matrix.id }}.json" in upload

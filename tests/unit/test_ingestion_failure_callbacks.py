"""#54: every ingestion failure code is in one closed, documented set.

The backend maps each code to a user-safe message, so a code must never be
invented ad hoc: FAILURE_CODES lists them all, and this scan checks that every
code raised in app/ is one of them.
"""

import re
from pathlib import Path

from app.services.tasks.ingestion_outcomes import FAILURE_CODES

APP = Path(__file__).resolve().parents[2] / "app"


def _codes_raised_in_app() -> set[str]:
    codes: set[str] = set()
    for path in APP.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        # A failure built with a literal code, on the same line or the next.
        codes.update(re.findall(r'IngestionError\(\s*"([A-Z][A-Z_]+)"', source))
        # A code the worker reports itself.
        if path.name == "worker.py":
            codes.update(re.findall(r'error_code="([A-Z][A-Z_]+)"', source))
            codes.update(re.findall(r'"(AI_[A-Z_]+|UNREGISTERED_HANDLER)"', source))
        # A module constant naming a code (PIPELINE_UNAVAILABLE, REFERENCE_REJECTED).
        if path.parent.name in ("tasks", "ingestion"):
            codes.update(re.findall(r'^[A-Z_]+ = "([A-Z][A-Z_]+)"$', source, flags=re.M))
    return codes


def test_every_failure_code_raised_in_app_is_in_the_closed_set():
    raised = _codes_raised_in_app()

    assert raised, "the scan found no codes"
    assert (
        raised <= FAILURE_CODES
    ), f"codes missing from FAILURE_CODES: {sorted(raised - FAILURE_CODES)}"

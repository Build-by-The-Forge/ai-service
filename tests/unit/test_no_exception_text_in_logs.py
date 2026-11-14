"""#52's rule, kept: an exception is logged by type and place, never by its text.

A regex sweep missed log calls whose exception variable had an unexpected name
(model_err) or whose call spanned lines. This guard reads app/'s syntax tree:
inside every ``except ... as <name>`` block, a logging call may not pass the
exception itself, str() or repr() of it, an f-string holding it, or a
traceback (exc_info, .exception). The safe forms are describe(<name>),
type(<name>).__name__ and fixed metadata attributes (code, status_code...).
The few sites reviewed and kept are listed with their reasons.
"""

import ast
import logging
import types
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[2] / "app"
LOG_METHODS = frozenset(
    {"debug", "info", "warning", "warn", "error", "critical", "exception", "log"}
)
# Attributes that are fixed metadata, never an exception's message.
SAFE_ATTRIBUTES = frozenset(
    {"__name__", "code", "error_code", "status_code", "retryable", "provider", "status"}
)
# Reviewed in #52 and kept, by file and the expression logged.
REVIEWED = {
    # Library import failures at startup: the library's and the system's text.
    ("app/services/ocr/paddle.py", "exc"),
    ("app/services/ocr/pdf_parser.py", "exc"),
    ("app/services/ocr/paddle.py", "exc_info"),
    ("app/services/ocr/pdf_parser.py", "exc_info"),
    # Offline evaluation tooling over local ground-truth files.
    ("app/core/documents/metrics.py", "f'Failed to parse ground truth JSON {gt_path}: {e}'"),
    # Signature rejections: fixed reasons written for the log, naming headers
    # or context keys, never their values (service_signature_v1).
    ("app/core/middleware/service_auth.py", "rejected.reason"),
}


def _uses(node: ast.AST, name: str) -> bool:
    """Whether the expression reaches the exception other than through a
    safe accessor."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "describe":
            return False
        if node.func.id == "type" and node.args and isinstance(node.args[0], ast.Name):
            return False
    if isinstance(node, ast.Attribute):
        chain, base = [], node
        while isinstance(base, ast.Attribute):
            chain.append(base.attr)
            base = base.value
        if isinstance(base, ast.Name) and base.id == name:
            return chain[-1] not in SAFE_ATTRIBUTES  # the first attribute read
    if isinstance(node, ast.Name):
        return node.id == name
    return any(_uses(child, name) for child in ast.iter_child_nodes(node))


def violations(source: str, path: str) -> list[tuple[str, str]]:
    found = []
    for handler in ast.walk(ast.parse(source)):
        if not isinstance(handler, ast.ExceptHandler) or not handler.name:
            continue
        for call in ast.walk(handler):
            if not (
                isinstance(call, ast.Call)
                and isinstance(call.func, ast.Attribute)
                and call.func.attr in LOG_METHODS
            ):
                continue
            if call.func.attr == "exception" or any(
                k.arg == "exc_info"
                and not (isinstance(k.value, ast.Constant) and not k.value.value)
                for k in call.keywords
            ):
                found.append((path, "exc_info"))
            for arg in [*call.args, *(k.value for k in call.keywords if k.arg != "exc_info")]:
                if _uses(arg, handler.name):
                    found.append((path, ast.unparse(arg)))
    return found


def test_the_guard_catches_every_form_and_allows_the_safe_ones():
    source = """
try:
    pass
except Exception as model_err:
    logger.warning("a: %s", model_err)
    logger.warning("b: %s", str(model_err))
    logger.warning(f"c: {model_err}")
    logger.error("d", exc_info=True)
    logger.exception("e")
    logger.warning("f", extra={"error": repr(model_err)})
    logger.warning("ok: %s", describe(model_err))
    logger.warning("ok", extra={"type": type(model_err).__name__, "code": model_err.code})
"""
    found = [what for _, what in violations(source, "x.py")]

    assert found == [
        "model_err",
        "str(model_err)",
        "f'c: {model_err}'",
        "exc_info",
        "exc_info",
        "{'error': repr(model_err)}",
    ]


def test_no_exception_text_reaches_a_log_in_app():
    found = []
    for path in sorted(APP.rglob("*.py")):
        relative = path.relative_to(APP.parent).as_posix()
        found += [
            v for v in violations(path.read_text(encoding="utf-8"), relative) if v not in REVIEWED
        ]

    assert not found, f"exception text logged (use describe(exc)): {found}"


# --- the database log lines, by behaviour -----------------------------------------------------

SENTINEL = "postgresql://build:secret-7d1e@db/build"


@pytest.mark.asyncio
async def test_a_failed_database_start_attempt_is_logged_without_its_text(monkeypatch, caplog):
    import app.db.session as session

    class Engine:
        def begin(self):
            raise ConnectionError(f"could not connect to {SENTINEL}")

    async def no_wait(seconds):
        return None

    monkeypatch.setattr(session, "_build_engine", lambda: Engine())
    # Only this module's reference: its retry waits are skipped, the loop's aren't.
    monkeypatch.setattr(session, "asyncio", types.SimpleNamespace(sleep=no_wait))

    with caplog.at_level(logging.DEBUG), pytest.raises(ConnectionError):
        await session.initialize_database()

    assert caplog.records, "nothing was logged"
    assert all(SENTINEL not in r.getMessage() for r in caplog.records)


@pytest.mark.asyncio
async def test_a_failed_engine_dispose_is_logged_without_its_text(monkeypatch, caplog):
    import app.db.session as session

    class Engine:
        async def dispose(self):
            raise ConnectionError(f"lost {SENTINEL}")

    monkeypatch.setattr(session, "engine", Engine())

    with caplog.at_level(logging.WARNING):
        await session.reset_engine()

    [line] = [r.getMessage() for r in caplog.records if "dispose" in r.getMessage()]
    assert "ConnectionError at " in line and SENTINEL not in line

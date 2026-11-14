"""#54: every ingestion failure code is in one closed set, however it is raised.

The backend maps each code to a user-safe message, so no code may appear that
FAILURE_CODES doesn't list. A literal-only scan missed a code raised through an
alias (StageReporter: ``error = Transient... if retryable else Permanent...``,
then ``error("CALLBACK_DELIVERY_FAILED", ...)``). This scan reads the syntax
tree: it resolves class aliases, conditional aliases and module constants, adds
the worker's own codes, and fails on any construction whose code it cannot
resolve, so a dynamic code can't slip past the set either. The runtime tests
then check the codes the stage reporter and the worker actually emit.
"""

import ast
from pathlib import Path

import pytest

from app.core.callbacks.dispatcher import InMemoryCallbackDispatcher, backend_callback_body
from app.core.callbacks.events import CallbackDeliveryResult, CallbackEventType
from app.core.execution_context import AITask
from app.services.tasks.handlers import TaskHandler, TaskHandlerRegistry
from app.services.tasks.ingestion_outcomes import (
    FAILURE_CODES,
    PIPELINE_UNAVAILABLE,
    IngestionFailure,
    StageReporter,
)
from app.services.tasks.task_records import TaskRecord
from app.services.tasks.transport import InMemoryTaskTransport
from app.services.tasks.worker import TaskWorker

APP = Path(__file__).resolve().parents[2] / "app"
FAILURE_CLASSES = frozenset(
    {
        "IngestionFailure",
        "TransientIngestionError",
        "PermanentIngestionError",
        "PipelineUnavailableError",
    }
)


def _name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _string_constants(tree: ast.Module) -> dict[str, str]:
    return {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _aliases(tree: ast.Module) -> set[str]:
    """The failure classes, and every name bound to one of them, directly or
    through a conditional expression, to a fixed point."""
    aliases = set(FAILURE_CLASSES)
    changed = True
    while changed:
        changed = False
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
                continue
            target = node.targets[0]
            values = (
                [node.value.body, node.value.orelse]
                if isinstance(node.value, ast.IfExp)
                else [node.value]
            )
            if (
                isinstance(target, ast.Name)
                and target.id not in aliases
                and all(_name(v) in aliases for v in values)
            ):
                aliases.add(target.id)
                changed = True
    return aliases


def scan(source: str, constants: dict[str, str]) -> tuple[set[str], list[int]]:
    """The codes the source constructs failures with, and the line of every
    construction whose code isn't a literal or a known constant."""
    tree = ast.parse(source)
    known = {**constants, **_string_constants(tree)}
    aliases = _aliases(tree)
    codes: set[str] = set()
    unresolved: list[int] = []

    def resolve(node: ast.expr | None, line: int) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            codes.add(node.value)
        elif isinstance(node, ast.Name) and node.id in known:
            codes.add(known[node.id])
        else:
            unresolved.append(line)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or _name(node.func) not in aliases:
            continue
        if _name(node.func) == "PipelineUnavailableError":
            codes.add(known.get("PIPELINE_UNAVAILABLE", PIPELINE_UNAVAILABLE))
            continue
        code = (
            node.args[0]
            if node.args
            else next((k.value for k in node.keywords if k.arg == "code"), None)
        )
        resolve(code, node.lineno)
    return codes, unresolved


def worker_codes(source: str) -> set[str]:
    """Codes the worker reports itself: error_code= keywords, the code argument
    of _fail_terminally, and literals assigned to a variable named code."""
    codes: set[str] = set()

    def literals(node: ast.expr) -> list[str]:
        branches = [node.body, node.orelse] if isinstance(node, ast.IfExp) else [node]
        return [
            b.value for b in branches if isinstance(b, ast.Constant) and isinstance(b.value, str)
        ]

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "error_code":
                    codes.update(literals(keyword.value))
            if _name(node.func) == "_fail_terminally" and len(node.args) >= 4:
                codes.update(literals(node.args[3]))
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "code" for t in node.targets
        ):
            codes.update(literals(node.value))
    return codes


def _app_modules() -> list[tuple[Path, str]]:
    return [(path, path.read_text(encoding="utf-8")) for path in sorted(APP.rglob("*.py"))]


def test_the_scan_resolves_aliases_constants_and_flags_dynamic_codes():
    source = """
NAMED = "NAMED_CODE"

def aliased(retryable):
    error = TransientIngestionError if retryable else PermanentIngestionError
    raise error("ALIASED_CODE", "message")

def through_a_constant():
    raise PermanentIngestionError(NAMED, "message")

def by_keyword():
    raise TransientIngestionError(code="KEYWORD_CODE", message="message")

def dynamic(code):
    raise TransientIngestionError(code, "message")
"""
    codes, unresolved = scan(source, {})

    assert codes == {"ALIASED_CODE", "NAMED_CODE", "KEYWORD_CODE"}
    assert unresolved == [15]  # the dynamic code is reported, not skipped


def test_every_failure_code_raised_in_app_is_in_the_closed_set():
    modules = _app_modules()
    constants: dict[str, str] = {}
    for _, source in modules:
        constants.update(_string_constants(ast.parse(source)))

    raised: set[str] = set()
    unresolved: list[str] = []
    for path, source in modules:
        codes, lines = scan(source, constants)
        raised |= codes
        unresolved += [f"{path.relative_to(APP.parent)}:{line}" for line in lines]
        if path.name == "worker.py":
            raised |= worker_codes(source)

    assert "CALLBACK_DELIVERY_FAILED" in raised  # the alias case the literal scan missed
    assert not unresolved, f"failures built with a code the set can't check: {unresolved}"
    assert (
        raised <= FAILURE_CODES
    ), f"codes missing from FAILURE_CODES: {sorted(raised - FAILURE_CODES)}"


# --- the codes actually emitted ---------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("retryable", [True, False])
async def test_a_stage_that_cannot_be_delivered_fails_with_a_listed_code(retryable):
    async def deliver(stage, metrics):
        return False, retryable

    with pytest.raises(IngestionFailure) as caught:
        await StageReporter(deliver).report("extracted", {})

    assert caught.value.code == "CALLBACK_DELIVERY_FAILED"
    assert caught.value.code in FAILURE_CODES
    assert caught.value.retryable is retryable


@pytest.mark.asyncio
async def test_a_stage_out_of_order_fails_with_a_listed_code():
    async def deliver(stage, metrics):  # pragma: no cover - refused before delivery
        return True, False

    with pytest.raises(IngestionFailure) as caught:
        await StageReporter(deliver).report("indexed", {})

    assert caught.value.code == "STAGE_OUT_OF_ORDER" and caught.value.code in FAILURE_CODES


class ReportsAStage(TaskHandler):
    async def handle(self, ctx, payload):
        await ctx.stage_reporter.report("extracted", {"characters": 1, "pages": 1})
        return {}


class RefusesStages(InMemoryCallbackDispatcher):
    async def dispatch(self, event):
        if event.event_type == CallbackEventType.STAGE_COMPLETED:
            return CallbackDeliveryResult(
                success=False, status_code=422, retryable=False, event_id=event.event_id
            )
        return await super().dispatch(event)


@pytest.mark.asyncio
async def test_the_worker_reports_an_undelivered_stage_with_a_listed_code():
    transport, registry, dispatcher = (
        InMemoryTaskTransport(),
        TaskHandlerRegistry(),
        RefusesStages(),
    )
    registry.register("content.ingestion", ReportsAStage())
    worker = TaskWorker(transport=transport, registry=registry, callback_dispatcher=dispatcher)
    task = AITask(
        task_id="task_1",
        task_type="content.ingestion",
        execution_id="exec_1",
        tenant_id="6f1c2a7e-4b3d-4c8e-9a1f-2d3e4f5a6b7c",
        job_id="job_1",
        asset_id="asset_1",
        run_number=1,
        attempt_count=1,
        correlation_id="corr_1",
        payload={},
        max_attempts=1,
    )
    await worker.task_records.create(
        TaskRecord(
            external_task_id=task.task_id,
            tenant_id=task.tenant_id,
            job_id=task.job_id,
            run_number=task.run_number,
            attempt_count=task.attempt_count,
        )
    )

    await worker.process_single_task("msg-1", task)

    [failed] = [e for e in dispatcher.delivered_events if e.event_type == CallbackEventType.FAILED]
    body = backend_callback_body(failed)
    assert body is not None and body["error_code"] == "CALLBACK_DELIVERY_FAILED"
    assert body["error_code"] in FAILURE_CODES and body["is_retryable"] is False

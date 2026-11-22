"""The versioned task set: original, synthetic course-style tasks.

Everything in a suite is invented for evaluation: the course, its passages, the
students' practice rows. No real student or course material is in it. A suite
file is pinned by its digest, so a result names exactly the tasks it ran.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.core.agentic.contracts import SuppliedData
from app.core.agentic.memory_ports import CorpusPassage

Category = Literal[
    "tool_selection",
    "argument_accuracy",
    "loop_detection",
    "abstention",
    "budget_exhaustion",
    "cancellation",
    "permission_denial",
    "replay_idempotency",
]
CATEGORIES: tuple[str, ...] = (
    "tool_selection",
    "argument_accuracy",
    "loop_detection",
    "abstention",
    "budget_exhaustion",
    "cancellation",
    "permission_denial",
    "replay_idempotency",
)
TENANT = "eval-tenant"
USER = "eval-student"
DEFAULT_SUITE = Path(__file__).resolve().parents[4] / "data" / "agent_eval" / "v1" / "tasks.json"


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Expect(_M):
    state: str
    code: str | None = None
    # The tools the model attempted, in order (a refused attempt counts).
    tools: tuple[str, ...] = ()
    # Calls that must have been made with at least these arguments.
    args: tuple[dict[str, Any], ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    proposals: int = 0
    cited: tuple[str, ...] = ()
    max_steps: int = 8


class Inject(_M):
    cancel_at_step: int | None = None
    # The process dies after this many checkpoint saves; the run is resumed.
    crash_after_saves: int | None = Field(default=None, ge=1)


class EvalTask(_M):
    id: str
    category: Category
    goal: str
    task_type: str = "grounded_explanation"
    allowed_tools: tuple[str, ...]
    permissions: tuple[str, ...] = (
        "content:read",
        "assessment:generate",
        "progress:read",
        "progress:propose",
        "plan:read",
        "plan:propose",
    )
    denied_assets: tuple[str, ...] = ()
    official_attempt_active: bool = False
    budget: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    completion: dict[str, Any] = Field(default_factory=dict)
    supplied: dict[str, Any] = Field(default_factory=dict)
    # The reference trajectory: the decisions a good model makes, by step.
    script: tuple[dict[str, Any], ...]
    inject: Inject = Field(default_factory=Inject)
    expect: Expect


class Suite(_M):
    version: int
    name: str
    sha256: str
    corpus: tuple[CorpusPassage, ...]
    tasks: tuple[EvalTask, ...]

    def supplied_for(self, task: EvalTask) -> SuppliedData:
        return SuppliedData.model_validate(task.supplied)


def load_suite(path: Path | None = None) -> Suite:
    raw = (path or DEFAULT_SUITE).read_bytes()
    data = json.loads(raw)
    corpus = tuple(
        CorpusPassage(
            tenant_id=TENANT,
            chunk_id=p["chunk_id"],
            document_id=p["document_id"],
            text=p["text"],
            page=p.get("page"),
        )
        for p in data["corpus"]
    )
    tasks = tuple(EvalTask.model_validate(t) for t in data["tasks"])
    ids = [t.id for t in tasks]
    if len(set(ids)) != len(ids):
        raise ValueError("task ids are unique")
    return Suite(
        version=int(data["version"]),
        name=str(data["name"]),
        sha256=hashlib.sha256(raw).hexdigest(),
        corpus=corpus,
        tasks=tasks,
    )

"""Tools reach data only through service boundaries, never the database, a
network client or a provider; and the retrieval door is the chat evidence path."""

import ast
from pathlib import Path

from app.core.agentic.contracts import CallerContext
from app.core.agentic.ports import Ports
from app.core.agentic.registry import ToolRegistry
from app.core.agentic.tools import register_study_tools
from app.services.agentic.ports import RetrievalEvidencePort
from tests.rag_support import institutional

APP = Path(__file__).resolve().parents[2] / "app"
PACKAGES = [APP / "core" / "agentic", APP / "services" / "agentic"]
BANNED = {
    "sqlalchemy",
    "asyncpg",
    "psycopg",
    "psycopg2",
    "redis",
    "httpx",
    "requests",
    "aiohttp",
    "boto3",
    "google",
    "openai",
    "anthropic",
    "fastapi",
}
BANNED_APP = ("app.db", "app.api", "app.core.models.providers")
PROVIDERS = ("gemini", "fireworks", "openai", "anthropic")


def _imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def test_no_agentic_module_imports_a_database_network_client_route_or_provider():
    files = [f for package in PACKAGES for f in package.rglob("*.py")]
    assert files
    for path in files:
        for module in _imports(path):
            assert module.split(".")[0] not in BANNED, (path.name, module)
            assert not module.startswith(BANNED_APP), (path.name, module)


def test_no_agentic_module_names_a_provider():
    for package in PACKAGES:
        for path in package.rglob("*.py"):
            text = path.read_text(encoding="utf-8").lower()
            for provider in PROVIDERS:
                assert provider not in text, (path.name, provider)


def test_the_portable_core_imports_no_server_service_or_config():
    """Only the services adapter package may touch app.services or app.core.config."""
    for path in (APP / "core" / "agentic").rglob("*.py"):
        for module in _imports(path):
            if module.startswith("app."):
                # safe_log is stdlib-only: the one shared helper (exceptions by type and place).
                assert module.startswith("app.core.agentic") or module == "app.core.safe_log"
                assert "config" not in module


def test_nothing_mounts_an_agentic_route():
    main = (APP / "main.py").read_text(encoding="utf-8")
    assert "agentic" not in main
    for path in (APP / "api").rglob("*.py"):
        assert "agentic" not in path.read_text(encoding="utf-8"), path.name


async def test_the_retrieval_door_is_the_chat_evidence_path(in_memory_retrieval, in_memory_scopes):
    """Tenant, scope and denied documents apply through the real engine."""
    text = "Conduction moves heat through a solid. " * 6
    for doc in ("asset-a", "asset-b"):
        await in_memory_retrieval.index_document(
            tenant_id="t1", document_id=doc, document_title=doc, text=text
        )
    await in_memory_retrieval.index_document(
        tenant_id="t2", document_id="asset-c", document_title="c", text=text
    )
    await institutional(in_memory_scopes, "t1", "asset-a", "asset-b")
    await institutional(in_memory_scopes, "t2", "asset-c")

    registry = register_study_tools(
        ToolRegistry(Ports(retrieval=RetrievalEvidencePort(in_memory_retrieval)))
    )

    def context(denied):
        return CallerContext(
            tenant_id="t1",
            user_id="u1",
            permissions=("content:read",),
            capabilities=("agentic_workflows",),
            denied_assets=denied,
        )

    async def search(ctx):
        return await registry.execute(
            "search_course_material",
            {"query": "conduction heat"},
            ctx,
            call_id="c",
            execution_id="e",
        )

    everything = await search(context(()))
    docs = {p["document_id"] for p in everything.data["passages"]}
    assert docs == {"asset-a", "asset-b"}  # never the other tenant's asset-c
    assert everything.data["passages"][0]["run_number"] == 1

    denied = await search(context(("asset-a",)))
    assert {p["document_id"] for p in denied.data["passages"]} == {"asset-b"}
    assert all(s["asset_id"] != "asset-a" for s in denied.data["sources_used"])

    no_scope = await search(context(None))
    assert no_scope.error_code == "MISSING_SCOPE"

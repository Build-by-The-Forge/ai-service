"""Every model call in the app goes through the gateway (AI-01 decision P3-5).

The legacy model layer (LLMProvider, OpenAILLM, DeterministicLLM) is removed,
and this keeps it removed. Its modules can't be imported, a provider SDK is
imported only by its own gateway adapter (listed below), and the only model
interface in app/ is GatewayLLM.
The gateway's own adapters (app/core/models/providers) call providers over
HTTP behind its approval check, and are the one approved path.
"""

import ast
import importlib
from pathlib import Path

import pytest

from app.core.models.budget import BudgetTracker
from app.core.models.gateway import ModelGateway
from app.services.ai.gateway_llm import GatewayLLM, ModelCaller

APP = Path(__file__).resolve().parents[2] / "app"
SDKS = {"openai", "anthropic", "google", "vertexai", "cohere", "mistralai"}
INTERFACE = {"generate", "generate_json", "for_capability"}
# The gateway adapters that may import a provider SDK, and which SDK. Each is
# reached only through the gateway, behind its approval check.
SDK_ADAPTERS = {("app/core/models/providers/gemini.py", "google")}


@pytest.mark.parametrize(
    "module", ["app.services.ai.llm", "app.services.ai.openai", "app.services.ai.deterministic"]
)
def test_the_legacy_model_layer_is_gone(module):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module)


def _modules() -> list[tuple[str, ast.Module]]:
    return [
        (path.relative_to(APP.parent).as_posix(), ast.parse(path.read_text(encoding="utf-8")))
        for path in sorted(APP.rglob("*.py"))
    ]


def test_a_provider_sdk_is_imported_only_by_its_gateway_adapter():
    imported = set()
    for path, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            else:
                continue
            imported |= {(path, name.split(".")[0]) for name in names if name.split(".")[0] in SDKS}

    # Nowhere else, and the listed adapters really do (so the list can't go stale).
    assert imported == SDK_ADAPTERS


def test_gateway_llm_is_the_only_model_interface_in_app():
    """A class with the model interface's methods would be a second way to
    call a model: only GatewayLLM may have them (ModelCaller itself declares
    them)."""
    implementers = []
    for path, tree in _modules():
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {
                    item.name
                    for item in node.body
                    if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef)
                }
                if methods & {"generate_json", "for_capability"}:
                    implementers.append(f"{path}:{node.name}")

    assert implementers == [
        "app/services/ai/gateway_llm.py:ModelCaller",
        "app/services/ai/gateway_llm.py:GatewayLLM",
    ]


def test_gateway_llm_is_a_model_caller():
    caller: ModelCaller = GatewayLLM(
        ModelGateway(providers=[], approved_providers=set(), budget_tracker=BudgetTracker())
    )
    assert INTERFACE <= set(dir(caller))

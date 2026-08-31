from collections.abc import Callable, Coroutine
from typing import Any

from app.core.tools.contracts import (
    ToolContext,
    ToolDefinition,
    ToolExecutionPolicy,
    ToolSideEffect,
)

ToolHandler = Callable[[ToolContext, dict[str, Any]], Coroutine[Any, Any, Any]]


class ToolRegistry:
    """
    Centralized registry for tool definitions and their execution handlers.
    Provides policy-aware and permission-filtered capability resolution.
    """

    def __init__(self):
        self._tools: dict[str, tuple[ToolDefinition, ToolHandler]] = {}

    def register(self, definition: ToolDefinition, handler: ToolHandler) -> None:
        self._tools[definition.name] = (definition, handler)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> tuple[ToolDefinition, ToolHandler] | None:
        return self._tools.get(name)

    def list_available(
        self,
        context: ToolContext,
        policy: ToolExecutionPolicy | None = None,
    ) -> list[ToolDefinition]:
        """
        Returns the subset of registered tools that the caller is authorized and permitted to execute.
        """
        available: list[ToolDefinition] = []

        for definition, _ in self._tools.values():
            if not definition.is_enabled:
                continue

            # 1. Permission check
            if definition.required_permissions:
                actor_perms = set(context.permissions)
                if not set(definition.required_permissions).issubset(actor_perms):
                    continue

            # 2. Capability check
            if definition.required_capabilities:
                actor_caps = set(context.capabilities)
                if not set(definition.required_capabilities).issubset(actor_caps):
                    continue

            # 3. Policy-level allowlist
            if policy and policy.allowed_tools is not None:
                if definition.name not in policy.allowed_tools:
                    continue

            # 4. Side-effect constraints
            if policy:
                if definition.side_effect == ToolSideEffect.PROPOSAL and not policy.allow_proposals:
                    continue
                if definition.side_effect == ToolSideEffect.MUTATING and not policy.allow_mutating:
                    continue

            available.append(definition)

        return available


_default_registry: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = ToolRegistry()
    return _default_registry


def set_tool_registry(registry: ToolRegistry) -> None:
    global _default_registry
    _default_registry = registry

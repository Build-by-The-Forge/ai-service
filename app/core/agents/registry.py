from typing import Dict, List, Optional

from app.core.agents.contracts import (
    AgentContext,
    AgentDefinition,
    AgentExecutionPolicy,
)
from app.core.models.contracts import ModelPolicy
from app.core.tools.contracts import ToolExecutionPolicy

STUDY_ASSISTANT_DEFINITION = AgentDefinition(
    agent_id="study.assistant",
    name="Academic Study Assistant",
    version="1.0.0",
    description="Assists students and teachers by looking up curriculum structure, searching course materials, and drafting question proposals.",
    system_prompt=(
        "You are the BUILD Academic Study Assistant. Your role is to help users explore curriculum topics, "
        "search verified study materials, and draft quiz questions when requested. "
        "Always cite topics and course codes accurately. You have access to tools for curriculum lookup, "
        "content search, and drafting questions. Use tools when specific academic data is required."
    ),
    model_policy=ModelPolicy(quality="balanced", max_tokens=2048, timeout_seconds=30.0),
    tool_policy=ToolExecutionPolicy(
        allowed_tools=["curriculum.lookup", "content.search", "draft.question"],
        allow_proposals=True,
        allow_mutating=False,
    ),
    execution_policy=AgentExecutionPolicy(
        max_steps=8,
        max_model_calls=8,
        max_tool_calls=6,
        max_tokens=8192,
        max_runtime_seconds=60.0,
    ),
    allowed_tools=["curriculum.lookup", "content.search", "draft.question"],
    required_permissions=["curriculum:read", "content:read", "assessment:generate"],
    required_capabilities=["curriculum_intelligence", "content_search", "cbt_generation"],
    allow_proposals=True,
    allow_mutating=False,
    is_enabled=True,
)

CONTENT_ANALYZER_DEFINITION = AgentDefinition(
    agent_id="content.analyzer",
    name="Course Material Analyzer",
    version="1.0.0",
    description="Analyzes uploaded course materials and textbook packs for topic coverage and clarity.",
    system_prompt=(
        "You are the BUILD Content Analyzer. Your purpose is to analyze course materials and summarize "
        "key subject matter concepts from indexed text."
    ),
    model_policy=ModelPolicy(quality="balanced", max_tokens=2048, timeout_seconds=30.0),
    tool_policy=ToolExecutionPolicy(
        allowed_tools=["content.search"],
        allow_proposals=False,
        allow_mutating=False,
    ),
    execution_policy=AgentExecutionPolicy(
        max_steps=5,
        max_model_calls=5,
        max_tool_calls=4,
        max_tokens=4096,
        max_runtime_seconds=45.0,
    ),
    allowed_tools=["content.search"],
    required_permissions=["content:read"],
    required_capabilities=["content_search"],
    allow_proposals=False,
    allow_mutating=False,
    is_enabled=True,
)


class AgentRegistry:
    """
    Registry managing known AgentDefinitions and providing permission-filtered agent resolution.
    """

    def __init__(self, register_builtins: bool = True):
        self._agents: Dict[str, AgentDefinition] = {}
        if register_builtins:
            self.register(STUDY_ASSISTANT_DEFINITION)
            self.register(CONTENT_ANALYZER_DEFINITION)

    def register(self, definition: AgentDefinition) -> None:
        self._agents[definition.agent_id] = definition

    def unregister(self, agent_id: str) -> None:
        self._agents.pop(agent_id, None)

    def get(self, agent_id: str) -> Optional[AgentDefinition]:
        return self._agents.get(agent_id)

    def list_available(self, context: AgentContext) -> List[AgentDefinition]:
        available: List[AgentDefinition] = []
        actor_perms = set(context.permissions)
        actor_caps = set(context.capabilities)

        for definition in self._agents.values():
            if not definition.is_enabled:
                continue

            if definition.required_permissions and not set(
                definition.required_permissions
            ).issubset(actor_perms):
                continue

            if definition.required_capabilities and not set(
                definition.required_capabilities
            ).issubset(actor_caps):
                continue

            available.append(definition)

        return available


_default_registry: Optional[AgentRegistry] = None


def get_agent_registry() -> AgentRegistry:
    global _default_registry
    if _default_registry is None:
        _default_registry = AgentRegistry()
    return _default_registry


def set_agent_registry(registry: AgentRegistry) -> None:
    global _default_registry
    _default_registry = registry

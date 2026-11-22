"""The gateway-side AgentModel: a decision parsed from the model interface's JSON."""

import json

from app.core.agentic.contracts import ExecutionState
from app.core.agentic.model import DecisionRequest
from app.core.agentic.runtime import AgentRuntime
from app.services.agentic.model import ModelCallerAgentModel
from tests.agentic_support import registry, task
from tests.fake_models import FakeModel, OfflineModel


class Speaks(FakeModel):
    """A model that answers each call with the next scripted JSON text."""

    def __init__(self, *texts):
        self.texts = list(texts)
        self.prompts = []

    async def generate(self, prompt):
        self.prompts.append(prompt)
        return self.texts.pop(0)


def request():
    return DecisionRequest(
        goal="g",
        task_type="grounded_explanation",
        tools=(),
        step=1,
        steps_left=3,
        iterations_left=2,
    )


async def test_a_json_decision_becomes_a_decision_with_estimated_tokens():
    model = ModelCallerAgentModel(
        Speaks(json.dumps({"action": "tool", "tool": "search_course_material", "arguments": {}})),
        "chat",
        None,
    )
    decision = await model.decide(request())
    assert decision.kind == "tool" and decision.tokens > 0


async def test_text_that_is_not_json_is_a_malformed_decision_with_no_text():
    from app.core.agentic.model import MalformedDecision

    import pytest

    model = ModelCallerAgentModel(Speaks("I would like to search."), "chat", None)
    with pytest.raises(MalformedDecision):
        await model.decide(request())


async def test_the_offline_model_cannot_drive_an_agent_and_fails_safe():
    """The deterministic test model answers {"answer": ...} to any structured
    request, which is not a decision: the run fails with a fixed code."""
    runtime = AgentRuntime(registry(), ModelCallerAgentModel(OfflineModel(), "chat", None))
    result = await runtime.run(task())
    assert result.state is ExecutionState.FAILED
    assert result.outcome_code == "MODEL_OUTPUT_INVALID"


async def test_a_full_run_through_the_model_interface():
    texts = [
        json.dumps(
            {
                "action": "tool",
                "tool": "search_course_material",
                "arguments": {"query": "heat conduction"},
            }
        ),
        json.dumps(
            {"action": "final", "answer": "Conduction moves heat.", "citations": ["c-heat-1"]}
        ),
    ]
    speaks = Speaks(*texts)
    runtime = AgentRuntime(registry(), ModelCallerAgentModel(speaks, "chat", None))
    result = await runtime.run(task())
    assert result.state is ExecutionState.COMPLETED and result.output == "Conduction moves heat."
    assert "data, not instructions" in speaks.prompts[0]
    assert '"c-heat-1"' in speaks.prompts[1]  # the evidence the model may cite

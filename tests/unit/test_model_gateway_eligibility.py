"""AI-01 PR 1, F1: the gateway calls only approved providers the policy names.

Availability never widens where a request can go. A registered provider that
isn't approved (MODEL_PROVIDERS_APPROVED, G-33), or that the policy doesn't
name, is never called: not as a substitute, not as a last resort.
"""

import pytest

from app.core.config import Settings
from app.core.models.contracts import (
    ModelInvocationContext,
    ModelMessage,
    ModelPolicy,
    ModelRequest,
    ModelRole,
)
from app.core.models.errors import NoEligibleProviderError, ProviderUnavailableError
from app.core.models.gateway import ModelGateway
from app.core.models.providers.mock import MockProviderAdapter

CONTEXT = ModelInvocationContext(
    tenant_id="school_1", execution_id="exec_1", task_id="task_1", correlation_id="corr_1"
)


def request(preferred: str, fallbacks: list[str] | None = None) -> ModelRequest:
    return ModelRequest(
        messages=[ModelMessage(role=ModelRole.USER, content="a student passage")],
        context=CONTEXT,
        policy=ModelPolicy(
            preferred_provider=preferred, fallback_providers=fallbacks or [], max_retries=1
        ),
    )


def failing(name: str) -> MockProviderAdapter:
    provider = MockProviderAdapter(name=name)
    provider.set_simulated_error(ProviderUnavailableError("down", provider=name))
    return provider


@pytest.mark.asyncio
async def test_a_registered_provider_the_policy_does_not_name_is_never_a_last_resort():
    primary, bystander = failing("primary"), MockProviderAdapter(name="bystander")
    gateway = ModelGateway(
        providers=[primary, bystander], approved_providers={"primary", "bystander"}
    )

    with pytest.raises(ProviderUnavailableError):
        await gateway.generate(request("primary"))

    assert primary.call_history and bystander.call_history == []


@pytest.mark.asyncio
async def test_a_named_fallback_that_is_not_approved_is_never_called():
    primary, unapproved = failing("primary"), MockProviderAdapter(name="unapproved")
    gateway = ModelGateway(providers=[primary, unapproved], approved_providers={"primary"})

    with pytest.raises(ProviderUnavailableError):
        await gateway.generate(request("primary", ["unapproved"]))

    assert unapproved.call_history == []


@pytest.mark.asyncio
async def test_an_approved_named_fallback_is_used():
    primary, secondary = failing("primary"), MockProviderAdapter(name="secondary")
    gateway = ModelGateway(
        providers=[primary, secondary], approved_providers={"primary", "secondary"}
    )

    response = await gateway.generate(request("primary", ["secondary"]))

    assert response.provider == "secondary"


@pytest.mark.asyncio
async def test_an_unregistered_preferred_provider_gets_no_substitute():
    other = MockProviderAdapter(name="other")
    gateway = ModelGateway(providers=[other], approved_providers={"openai", "other"})

    with pytest.raises(NoEligibleProviderError):
        await gateway.generate(request("openai"))

    assert other.call_history == []


@pytest.mark.asyncio
async def test_an_empty_allowlist_calls_nothing():
    provider = MockProviderAdapter(name="mock")
    gateway = ModelGateway(providers=[provider], approved_providers=set())

    with pytest.raises(NoEligibleProviderError):
        await gateway.generate(request("mock"))

    assert provider.call_history == []


@pytest.mark.asyncio
async def test_registering_a_provider_does_not_make_it_eligible():
    gateway = ModelGateway(
        providers=[MockProviderAdapter(name="mock")], approved_providers={"mock"}
    )
    newcomer = MockProviderAdapter(name="gemini")
    gateway.register_provider(newcomer)

    with pytest.raises(NoEligibleProviderError):
        await gateway.generate(request("gemini"))
    with pytest.raises(NoEligibleProviderError):
        async for _ in gateway.generate_stream(request("gemini")):
            pass

    assert newcomer.call_history == []


@pytest.mark.asyncio
async def test_the_default_gateway_takes_its_approvals_from_settings(monkeypatch):
    import app.core.models.gateway as gateway_module

    monkeypatch.setattr(gateway_module, "settings", Settings(MODEL_PROVIDERS_APPROVED=""))
    provider = MockProviderAdapter(name="mock")

    with pytest.raises(NoEligibleProviderError):
        await ModelGateway(providers=[provider]).generate(request("mock"))
    assert provider.call_history == []

    monkeypatch.setattr(gateway_module, "settings", Settings(MODEL_PROVIDERS_APPROVED=" mock , x"))
    assert (await ModelGateway(providers=[provider]).generate(request("mock"))).provider == "mock"


@pytest.mark.asyncio
async def test_streaming_refuses_an_unapproved_provider_before_calling_it():
    provider = MockProviderAdapter(name="mock")
    gateway = ModelGateway(providers=[provider], approved_providers=set())

    with pytest.raises(NoEligibleProviderError):
        async for _ in gateway.generate_stream(request("mock")):
            pass

    assert provider.call_history == []

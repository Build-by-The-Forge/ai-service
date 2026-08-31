import time

from app.core.models.resilience import CircuitBreaker, CircuitState


def test_circuit_breaker_transitions_to_open_after_failure_threshold():
    cb = CircuitBreaker(provider_name="openai", failure_threshold=3, cooldown_seconds=1.0)
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # 1st failure
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True

    # 2nd failure
    cb.record_failure()
    assert cb.state == CircuitState.CLOSED

    # 3rd failure -> trips circuit to OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False


def test_circuit_breaker_recovers_through_half_open():
    cb = CircuitBreaker(
        provider_name="openai",
        failure_threshold=2,
        cooldown_seconds=0.05,
        half_open_success_threshold=2,
    )

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False

    # Wait for cooldown
    time.sleep(0.06)

    # In HALF_OPEN state, probe is allowed
    assert cb.state == CircuitState.HALF_OPEN
    assert cb.can_execute() is True

    # First success in HALF_OPEN
    cb.record_success()
    assert cb.state == CircuitState.HALF_OPEN

    # Second success in HALF_OPEN -> closes circuit
    cb.record_success()
    assert cb.state == CircuitState.CLOSED
    assert cb.can_execute() is True


def test_circuit_breaker_re_opens_on_half_open_failure():
    cb = CircuitBreaker(
        provider_name="openai",
        failure_threshold=2,
        cooldown_seconds=0.05,
    )

    cb.record_failure()
    cb.record_failure()
    assert cb.state == CircuitState.OPEN

    time.sleep(0.06)
    assert cb.state == CircuitState.HALF_OPEN

    # Failure during probe immediately trips back to OPEN
    cb.record_failure()
    assert cb.state == CircuitState.OPEN
    assert cb.can_execute() is False

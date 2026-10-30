"""The test-only fault-injection pause points (P3-6 §14) cannot become a
production control surface, and behave deterministically in tests."""

import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, settings
from app.services.tasks import fault_points
from app.services.tasks.fault_points import (
    BEFORE_INDEX_WRITE,
    FaultPointNotReleasedError,
    pause_point,
)

APP = Path(__file__).resolve().parents[2] / "app"


@pytest.mark.parametrize("environment", ["development", "staging", "production"])
def test_the_setting_is_refused_outside_testing(environment):
    with pytest.raises(ValidationError, match="INGESTION_FAULT_DIR"):
        Settings(APP_ENVIRONMENT=environment, INGESTION_FAULT_DIR="/faults")


def test_the_setting_is_accepted_only_in_testing():
    assert (
        Settings(APP_ENVIRONMENT="testing", INGESTION_FAULT_DIR="/faults").INGESTION_FAULT_DIR
        == "/faults"
    )


def test_only_the_ingestion_handler_reaches_the_pause_points():
    """No route, middleware or other module imports the hook: nothing a
    request carries can arm or release it."""
    users = sorted(
        str(path.relative_to(APP)).replace("\\", "/")
        for path in APP.rglob("*.py")
        if "fault_points" in path.read_text(encoding="utf-8") and path.name != "fault_points.py"
    )
    assert users == ["core/config.py", "services/tasks/handlers.py"]


async def test_inert_without_the_setting(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "INGESTION_FAULT_DIR", None)
    (tmp_path / f"{BEFORE_INDEX_WRITE}.pause").write_text("")
    await asyncio.wait_for(pause_point(BEFORE_INDEX_WRITE, "job-1"), timeout=1)
    assert list(tmp_path.glob("*.reached")) == []


async def test_refused_at_runtime_outside_testing(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "INGESTION_FAULT_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "APP_ENVIRONMENT", "development")
    with pytest.raises(RuntimeError, match="APP_ENVIRONMENT=testing"):
        await pause_point(BEFORE_INDEX_WRITE, "job-1")


async def test_an_unarmed_point_passes_straight_through(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "INGESTION_FAULT_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "APP_ENVIRONMENT", "testing")
    await asyncio.wait_for(pause_point(BEFORE_INDEX_WRITE, "job-1"), timeout=1)
    assert list(tmp_path.iterdir()) == []


async def test_an_armed_point_holds_until_released_and_cleans_up(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "INGESTION_FAULT_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "APP_ENVIRONMENT", "testing")
    gate = tmp_path / f"{BEFORE_INDEX_WRITE}.pause"
    gate.write_text("")
    held = asyncio.create_task(pause_point(BEFORE_INDEX_WRITE, "job-1"))
    marker = tmp_path / f"{BEFORE_INDEX_WRITE}.job-1.reached"
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.01)
    assert marker.exists() and not held.done()
    gate.unlink()
    await asyncio.wait_for(held, timeout=1)
    assert not marker.exists()


async def test_a_cancellation_while_held_does_not_recall_the_effect(monkeypatch, tmp_path):
    """Models an effect already past its ownership check: the hold ends on
    release, and the caller continues (its own fence check stops it)."""
    monkeypatch.setattr(settings, "INGESTION_FAULT_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "APP_ENVIRONMENT", "testing")
    gate = tmp_path / f"{BEFORE_INDEX_WRITE}.pause"
    gate.write_text("")
    effects = []

    async def handler():
        await pause_point(BEFORE_INDEX_WRITE, "job-1")
        effects.append("write")

    running = asyncio.create_task(handler())
    await asyncio.sleep(0.1)
    running.cancel()
    await asyncio.sleep(0.1)
    assert not running.done()
    gate.unlink()
    await asyncio.wait_for(running, timeout=1)
    assert effects == ["write"]


async def test_a_hold_that_is_never_released_fails(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "INGESTION_FAULT_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "APP_ENVIRONMENT", "testing")
    monkeypatch.setattr(fault_points, "MAX_HOLD_SECONDS", 0.2)
    (tmp_path / f"{BEFORE_INDEX_WRITE}.pause").write_text("")
    with pytest.raises(FaultPointNotReleasedError):
        await pause_point(BEFORE_INDEX_WRITE, "job-1")
    assert list(tmp_path.glob("*.reached")) == []


async def test_the_handler_pauses_before_its_index_write(monkeypatch, tmp_path):
    from tests.unit.test_ingestion_fetch_lifecycle import (
        KEY,
        PDF,
        FakeIndex,
        handler,
        payload,
        task,
    )
    from tests.unit.test_asset_fetcher import FakeS3
    from tests.unit.test_worker_callback_integration import run, worker_with

    faults = tmp_path / "faults"
    faults.mkdir()
    monkeypatch.setattr(settings, "INGESTION_FAULT_DIR", str(faults))
    monkeypatch.setattr(settings, "APP_ENVIRONMENT", "testing")
    gate = faults / f"{BEFORE_INDEX_WRITE}.pause"
    gate.write_text("")
    index = FakeIndex()
    worker, transport, _ = worker_with(handler(FakeS3({KEY: PDF}), tmp_path / "work", index=index))
    (tmp_path / "work").mkdir()

    running = asyncio.create_task(run(worker, transport, task(payload=payload())))
    for _ in range(200):
        if list(faults.glob("*.reached")):
            break
        await asyncio.sleep(0.02)
    assert list(faults.glob("*.reached")) and index.calls == []  # held before the write
    gate.unlink()
    await asyncio.wait_for(running, timeout=10)
    assert len(index.calls) == 1

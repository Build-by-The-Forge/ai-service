import pytest

import app.db.session as session_module


@pytest.mark.asyncio
async def test_initialize_database_preserves_engine_for_app_lifespan():
    await session_module.reset_engine()

    await session_module.initialize_database()

    assert session_module.engine is not None
    assert session_module.AsyncSessionLocal is not None

    await session_module.reset_engine()


@pytest.mark.asyncio
async def test_reset_engine_ignores_disposal_errors(monkeypatch):
    await session_module.reset_engine()

    class BrokenEngine:
        async def dispose(self):
            raise RuntimeError("dispose failed")

    monkeypatch.setattr(session_module, "engine", BrokenEngine())
    monkeypatch.setattr(session_module, "AsyncSessionLocal", object())

    await session_module.reset_engine()

    assert session_module.engine is None
    assert session_module.AsyncSessionLocal is None

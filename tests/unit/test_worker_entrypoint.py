import asyncio

from app import worker as worker_entry


async def test_the_worker_process_runs_until_stopped(monkeypatch):
    started: list[asyncio.Event] = []

    class FakeWorker:
        async def start(self, stop: asyncio.Event) -> None:
            started.append(stop)
            await stop.wait()

    monkeypatch.setattr(worker_entry, "TaskWorker", FakeWorker)
    task = asyncio.create_task(worker_entry.main())
    while not started:
        await asyncio.sleep(0)

    started[0].set()
    await asyncio.wait_for(task, timeout=2)

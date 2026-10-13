"""Task worker process: ``python -m app.worker``.

Consumes ingestion tasks from the Redis stream and reports to the backend
through signed callbacks to BACKEND_CALLBACK_URL. Runs beside the API
process (the compose ``ai-worker`` service); stops cleanly on SIGINT/SIGTERM.
"""

import asyncio
import logging
import signal

from app.core.logging import setup_logging
from app.services.tasks.worker import TaskWorker

logger = logging.getLogger(__name__)


async def main() -> None:
    setup_logging()
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows event loops
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    await TaskWorker().start(stop)


if __name__ == "__main__":
    asyncio.run(main())

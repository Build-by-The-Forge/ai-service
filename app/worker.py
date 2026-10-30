"""Task worker process: ``python -m app.worker``.

Consumes ingestion tasks from the Redis stream and reports to the backend
through signed callbacks to BACKEND_CALLBACK_URL. Runs beside the API
process (the compose ``ai-worker`` service); stops cleanly on SIGINT/SIGTERM.
"""

import asyncio
import logging
import signal

from prometheus_client import start_http_server

from app.core.logging import setup_logging
from app.services.tasks.worker import TaskWorker

logger = logging.getLogger(__name__)
WORKER_METRICS_PORT = 9102


async def main() -> None:
    setup_logging()
    metrics_server, _ = start_http_server(WORKER_METRICS_PORT)
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, stop.set)
        except NotImplementedError:  # Windows event loops
            signal.signal(sig, lambda *_: loop.call_soon_threadsafe(stop.set))
    try:
        await TaskWorker().start(stop)
    finally:
        metrics_server.shutdown()
        metrics_server.server_close()


if __name__ == "__main__":
    asyncio.run(main())

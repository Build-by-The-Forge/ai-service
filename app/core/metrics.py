"""Application and task-worker Prometheus metrics."""

import logging

from prometheus_client import Counter, Gauge

logger = logging.getLogger(__name__)

task_transitions_total = Counter(
    "ai_service_task_transitions_total",
    "AI-service task lifecycle transitions",
    ["outcome"],
)
callback_delivery_failures_total = Counter(
    "ai_service_callback_delivery_failures_total",
    "Callback delivery failures",
    ["event_type"],
)
stream_entries_reclaimed_total = Counter(
    "ai_service_stream_entries_reclaimed_total",
    "Stream entries resolved by the lease reclaimer",
    ["outcome"],
)
expired_lease_candidates = Gauge(
    "ai_service_expired_lease_candidates",
    "Expired task leases identified during the most recent reclaim pass",
)
learning_memory_enabled = Gauge(
    "ai_service_learning_memory_enabled",
    "1 when learning memory is enabled in this process, 0 when the gate keeps it off",
)
learning_memory_refusals_total = Counter(
    "ai_service_learning_memory_refusals_total",
    "Learning-memory reads and writes refused by the production gate",
    ["operation"],
)


def setup_metrics(app):
    """Expose HTTP instrumentation and application metrics at /metrics."""
    try:
        from prometheus_fastapi_instrumentator import Instrumentator
    except ImportError as exc:
        logger.warning(
            "ai_service_http_metrics_unavailable",
            extra={"error_type": type(exc).__name__},
        )
        return

    Instrumentator().instrument(app).expose(app)

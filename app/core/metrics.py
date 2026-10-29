"""Application and task-worker Prometheus metrics."""

from prometheus_client import Counter, Gauge

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


def setup_metrics(app):
    """Expose HTTP instrumentation and application metrics at /metrics."""
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)

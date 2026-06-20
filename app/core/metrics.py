"""Prometheus metrics setup."""

def setup_metrics(app):
    """
    Setup Prometheus metrics for FastAPI.
    Requires: pip install prometheus-fastapi-instrumentator
    """
    try:
        from prometheus_fastapi_instrumentator import Instrumentator

        Instrumentator().instrument(app).expose(app)
    except ImportError:
        # Metrics optional in development
        pass

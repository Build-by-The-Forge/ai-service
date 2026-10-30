import json
import logging
import sys

from app.core.config import settings
from app.core.correlation import get_correlation_id


class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "timestamp": self.formatTime(record),
        }
        for field in (
            "job_id",
            "task_id",
            "run_number",
            "attempt",
            "worker_id",
            "consumer",
            "entry_id",
            "event_id",
            "event_type",
            "outcome",
            "error_code",
            "error_type",
            "retryable",
            "delivery_attempt",
            "max_attempts",
            "status_code",
        ):
            value = getattr(record, field, None)
            if value is not None:
                log_obj[field] = value
        correlation_id = get_correlation_id()
        if correlation_id:
            log_obj["correlation_id"] = correlation_id
        return json.dumps(log_obj)


def setup_logging():
    logger = logging.getLogger()
    logger.setLevel(settings.LOG_LEVEL)

    handler = logging.StreamHandler(sys.stdout)
    formatter = JsonFormatter()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.INFO)


logger = logging.getLogger(__name__)

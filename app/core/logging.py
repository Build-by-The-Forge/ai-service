import logging
import sys
import json
from app.core.config import settings

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "timestamp": self.formatTime(record),
        }
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

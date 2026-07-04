import logging

from app.core.logging import setup_logging


def test_setup_logging_attaches_handler():
    root_logger = logging.getLogger()
    root_logger.handlers = []

    setup_logging()

    assert root_logger.handlers

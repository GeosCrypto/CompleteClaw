"""Structured logging helper."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str = "completeclaw", level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger for *name*.

    The logger writes to *stderr* in the format::

        2024-01-01 12:00:00 | INFO     | completeclaw.llm.openai | message

    If the logger already has handlers attached (e.g. the caller configured
    it themselves) this function does not add another handler.

    Example::

        from completeclaw.utils.logging import get_logger

        log = get_logger(__name__)
        log.info("Provider initialised")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger

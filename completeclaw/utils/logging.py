"""Structured logging helper."""

from __future__ import annotations

import logging
import sys
from typing import Optional


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Return a configured logger for *name*.

    If the logger has no handlers yet, a ``StreamHandler`` writing to
    ``stderr`` is added with a simple timestamped format.

    Parameters
    ----------
    name:
        Logger name – typically ``__name__`` of the calling module.
    level:
        Optional log level (e.g. ``logging.DEBUG``).  Defaults to
        ``logging.INFO``.

    Example::

        logger = get_logger(__name__)
        logger.info("Agent started")
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
    if level is not None:
        logger.setLevel(level)
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)
    return logger

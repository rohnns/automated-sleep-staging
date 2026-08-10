"""Shared logging helpers for the sleep-staging pipeline."""

from __future__ import annotations

import logging
from typing import Final

DEFAULT_LOG_FORMAT: Final[str] = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
)


def configure_logging(
    level: int | str = logging.INFO,
    *,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """Configure root logging once for library and CLI usage.

    Parameters
    ----------
    level:
        Logging level name or numeric level.
    log_format:
        ``logging`` format string.
    """
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    logging.basicConfig(level=level, format=log_format)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a pipeline module."""
    return logging.getLogger(name)

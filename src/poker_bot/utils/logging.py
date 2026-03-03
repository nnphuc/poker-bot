"""Logging setup using loguru."""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(log_file: str | Path | None = None, level: str = "INFO") -> None:
    """Configure loguru logger."""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        colorize=True,
    )
    if log_file:
        logger.add(
            str(log_file),
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
            rotation="50 MB",
            retention="7 days",
        )

"""Logging helpers for mini-apigw."""
from __future__ import annotations

import logging
from logging.config import dictConfig
from typing import Optional

from .config import LoggingConfig


def configure_logging(config: LoggingConfig) -> None:
    """Configure global logging based on configuration values."""

    level = getattr(logging, config.level.upper(), logging.INFO)
    log_format = "%(asctime)s %(levelname)s [%(name)s] %(message)s"

    dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": log_format,
                }
            },
            "handlers": {
                "default": {
                    "level": level,
                    "class": "logging.StreamHandler",
                    "formatter": "standard",
                }
            },
            "root": {
                "handlers": ["default"],
                "level": level,
            },
        }
    )

    logging.getLogger("uvicorn.access").disabled = not config.access_log


def redact_prompt(prompt: Optional[str], enabled: bool) -> Optional[str]:
    if not enabled:
        return prompt
    if prompt is None:
        return None
    return "<redacted>"


__all__ = ["configure_logging", "redact_prompt"]

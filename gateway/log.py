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

    handler_config = {
        "level": level,
        "formatter": "standard",
    }

    if config.file:
        handler_config.update(
            {
                "class": "logging.handlers.WatchedFileHandler",
                "filename": config.file,
                "encoding": "utf-8",
            }
        )
    else:
        handler_config["class"] = "logging.StreamHandler"

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
                "default": handler_config,
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

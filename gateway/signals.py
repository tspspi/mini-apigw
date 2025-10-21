"""Signal handling utilities."""
from __future__ import annotations

import asyncio
import signal
import logging

from .runtime import GatewayRuntime

log = logging.getLogger(__name__)

def install_signal_handlers(runtime: GatewayRuntime, enable_reload: bool) -> None:
    loop = asyncio.get_running_loop()

    def _sigterm_handler():
        log.info("SIGTERM received; shutting down gateway")
        asyncio.create_task(runtime.shutdown())

    try:
        loop.add_signal_handler(signal.SIGTERM, _sigterm_handler)
    except NotImplementedError:  # pragma: no cover - Windows
        signal.signal(signal.SIGTERM, lambda *_: asyncio.create_task(runtime.shutdown()))

    if enable_reload:
        def _sighup_handler():
            log.info("SIGHUP received; reloading configuration")
            asyncio.create_task(runtime.reload())

        try:
            loop.add_signal_handler(signal.SIGHUP, _sighup_handler)
        except (AttributeError, NotImplementedError):  # pragma: no cover
            signal.signal(signal.SIGHUP, lambda *_: asyncio.create_task(runtime.reload()))

__all__ = ["install_signal_handlers"]

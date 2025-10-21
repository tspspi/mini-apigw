from __future__ import annotations

import logging
from typing import Callable

# We use starlette since FastAPI is built on starlette and so it's always
# already installed

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..trace import TracePayload

log = logging.getLogger(__name__)

class TraceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        response = await call_next(request)
        payload = getattr(request.state, "trace_payload", None)
        if payload is None:
            return response
        if not isinstance(payload, TracePayload):
            log.warning("Ignoring trace payload with unexpected type: %r", type(payload))
            return response

        runtime = getattr(request.app.state, "runtime", None)
        if runtime is None:
            log.debug("Runtime not available; skipping trace logging")
            return response

        try:
            manager = runtime.trace_manager()
        except Exception:  # pragma: no cover - runtime misconfiguration
            log.exception("Trace manager unavailable")
            return response

        if manager is None:
            return response

        try:
            await manager.process(payload)
        except Exception:  # pragma: no cover - trace logging failures are non-fatal
            log.exception("Failed to record trace log entry")
        return response


__all__ = ["TraceMiddleware"]

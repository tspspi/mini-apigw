from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Dict, Optional

from ..config import BackendDefinition


@dataclass(slots=True)
class BackendUsage:
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    cost: Optional[float] = None


@dataclass(slots=True)
class BackendResult:
    body: Dict[str, Any]
    usage: BackendUsage = field(default_factory=BackendUsage)

    async def as_sse(self, request_id: str) -> AsyncIterator[bytes]:
        """Render a non-streaming response as a single SSE message."""

        payload = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "delta": self.body.get("choices", [{}])[0].get("message", {}),
                    "index": 0,
                    "finish_reason": "stop",
                }
            ],
        }
        data = json.dumps(payload).encode("utf-8")
        yield b"data: " + data + b"\n\n"
        yield b"data: [DONE]\n\n"


class BackendClient:
    """Abstract base class for model providers."""

    def __init__(self, definition: BackendDefinition):
        self.definition = definition

    async def chat(self, model: str, payload: Dict[str, Any], stream: bool = False) -> BackendResult | AsyncIterator[bytes]:
        raise NotImplementedError

    async def completions(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError

    async def embeddings(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError

    async def images(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError

    async def models(self) -> Dict[str, Any]:
        raise NotImplementedError

    async def _asyncify(self, func, *args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: func(*args, **kwargs))


__all__ = ["BackendClient", "BackendResult", "BackendUsage"]

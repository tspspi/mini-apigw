"""Helpers for Server-Sent Events responses."""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Iterable


async def sse_from_chunks(chunks: AsyncIterator[Dict[str, Any]]) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        payload = json.dumps(chunk).encode("utf-8")
        yield b"data: " + payload + b"\n\n"
    yield b"data: [DONE]\n\n"


async def sse_from_strings(chunks: AsyncIterator[str]) -> AsyncIterator[bytes]:
    async for chunk in chunks:
        data = chunk.encode("utf-8")
        yield b"data: " + data + b"\n\n"
    yield b"data: [DONE]\n\n"


async def async_iter(iterable: Iterable[Any]) -> AsyncIterator[Any]:
    for item in iterable:
        yield item


__all__ = ["sse_from_chunks", "sse_from_strings", "async_iter"]

from __future__ import annotations

import gzip
import io
import json
import zlib
from typing import Callable

import zstandard
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


class RequestDecompressionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        encoding = request.headers.get("content-encoding")
        if not encoding:
            return await call_next(request)

        raw_body = await request.body()
        if not raw_body:
            return await call_next(request)

        try:
            body = self._decode(raw_body, encoding)
        except ValueError as exc:
            payload = json.dumps({"detail": str(exc)}).encode("utf-8")
            return Response(payload, status_code=400, media_type="application/json")

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request._body = body
        request._receive = receive
        return await call_next(request)

    @staticmethod
    def _decode(body: bytes, encoding: str) -> bytes:
        normalized = encoding.strip().lower()
        if normalized in {"zstd", "x-zstd"}:
            try:
                with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body)) as reader:
                    return reader.read()
            except zstandard.ZstdError as exc:
                raise ValueError("Invalid zstd-compressed request body") from exc
        if normalized == "gzip":
            try:
                return gzip.decompress(body)
            except OSError as exc:
                raise ValueError("Invalid gzip-compressed request body") from exc
        if normalized == "deflate":
            try:
                return zlib.decompress(body)
            except zlib.error as exc:
                raise ValueError("Invalid deflate-compressed request body") from exc
        raise ValueError(f"Unsupported Content-Encoding '{encoding}'")


__all__ = ["RequestDecompressionMiddleware"]

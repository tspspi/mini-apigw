from __future__ import annotations

import gzip
import io
import json
import zlib
from typing import Callable

import zstandard
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.datastructures import MutableHeaders
from starlette.requests import Request
from starlette.responses import Response


class RequestDecompressionMiddleware(BaseHTTPMiddleware):
    _MAX_DECOMPRESSED_BODY_BYTES = 8 * 1024 * 1024

    async def dispatch(self, request: Request, call_next: Callable[[Request], Response]) -> Response:
        raw_encoding = request.headers.get("content-encoding")
        if not raw_encoding:
            return await call_next(request)

        raw_body = await request.body()
        if not raw_body:
            return await call_next(request)

        try:
            body = self._decode(raw_body, raw_encoding)
        except ValueError as exc:
            payload = json.dumps({"detail": str(exc)}).encode("utf-8")
            return Response(payload, status_code=400, media_type="application/json")

        async def receive():
            return {"type": "http.request", "body": body, "more_body": False}

        request._body = body
        request._receive = receive
        headers = MutableHeaders(scope=request.scope)
        if "content-encoding" in headers:
            del headers["content-encoding"]
        headers["content-length"] = str(len(body))
        return await call_next(request)

    @staticmethod
    def _decode(body: bytes, encoding: str) -> bytes:
        encodings = [token.strip().lower() for token in encoding.split(",") if token.strip()]
        if not encodings:
            raise ValueError("Unsupported Content-Encoding ''")
        data = body
        for normalized in reversed(encodings):
            data = RequestDecompressionMiddleware._decode_single(data, normalized, encoding)
        return data

    @staticmethod
    def _decode_single(body: bytes, normalized: str, original_encoding: str) -> bytes:
        if normalized in {"zstd", "x-zstd"}:
            try:
                with zstandard.ZstdDecompressor().stream_reader(io.BytesIO(body)) as reader:
                    return RequestDecompressionMiddleware._read_limited(
                        reader, RequestDecompressionMiddleware._MAX_DECOMPRESSED_BODY_BYTES
                    )
            except zstandard.ZstdError as exc:
                raise ValueError("Invalid zstd-compressed request body") from exc
        if normalized == "gzip":
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as reader:
                    return RequestDecompressionMiddleware._read_limited(
                        reader, RequestDecompressionMiddleware._MAX_DECOMPRESSED_BODY_BYTES
                    )
            except OSError as exc:
                raise ValueError("Invalid gzip-compressed request body") from exc
        if normalized == "deflate":
            try:
                obj = zlib.decompressobj()
                data = obj.decompress(body, RequestDecompressionMiddleware._MAX_DECOMPRESSED_BODY_BYTES + 1)
                if obj.unconsumed_tail or len(data) > RequestDecompressionMiddleware._MAX_DECOMPRESSED_BODY_BYTES:
                    raise ValueError("Decompressed request body too large")
                data += obj.flush()
                if len(data) > RequestDecompressionMiddleware._MAX_DECOMPRESSED_BODY_BYTES:
                    raise ValueError("Decompressed request body too large")
                return data
            except zlib.error as exc:
                raise ValueError("Invalid deflate-compressed request body") from exc
        raise ValueError(f"Unsupported Content-Encoding '{original_encoding}'")

    @staticmethod
    def _read_limited(reader, limit: int) -> bytes:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = reader.read(min(65536, limit + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ValueError("Decompressed request body too large")
            chunks.append(chunk)
        return b"".join(chunks)


__all__ = ["RequestDecompressionMiddleware"]

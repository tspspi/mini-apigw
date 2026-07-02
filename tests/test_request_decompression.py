import zstandard
import zlib
import gzip
import io

from fastapi import Body, FastAPI, Request
from fastapi.testclient import TestClient

from gateway.middleware.decompression import RequestDecompressionMiddleware


def test_zstd_request_body_is_decompressed_before_json_parsing():
    app = FastAPI()
    app.add_middleware(RequestDecompressionMiddleware)

    @app.post("/echo")
    async def echo(payload=Body(...)):
        return payload

    body = b'{"model":"test-model","input":"hello"}'
    compressed = zstandard.ZstdCompressor().compress(body)

    with TestClient(app) as client:
        response = client.post(
            "/echo",
            headers={"Content-Type": "application/json", "Content-Encoding": "zstd"},
            content=compressed,
        )

    assert response.status_code == 200
    assert response.json() == {"model": "test-model", "input": "hello"}


def test_zstd_request_without_content_size_is_decompressed():
    app = FastAPI()
    app.add_middleware(RequestDecompressionMiddleware)

    @app.post("/echo")
    async def echo(payload=Body(...)):
        return payload

    body = b'{"model":"test-model","input":"hello"}'
    compressed = zstandard.ZstdCompressor(write_content_size=False).compress(body)

    with TestClient(app) as client:
        response = client.post(
            "/echo",
            headers={"Content-Type": "application/json", "Content-Encoding": "zstd"},
            content=compressed,
        )

    assert response.status_code == 200
    assert response.json() == {"model": "test-model", "input": "hello"}


def test_zstd_request_that_expands_too_large_is_rejected():
    app = FastAPI()
    app.add_middleware(RequestDecompressionMiddleware)

    @app.post("/echo")
    async def echo(payload=Body(...)):
        return payload

    body = b"a" * ((8 * 1024 * 1024) + 1)
    compressed = zstandard.ZstdCompressor(write_content_size=False).compress(body)

    with TestClient(app) as client:
        response = client.post(
            "/echo",
            headers={"Content-Type": "application/json", "Content-Encoding": "zstd"},
            content=compressed,
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Decompressed request body too large"


def test_decompression_updates_request_headers_for_downstream_consumers():
    app = FastAPI()
    app.add_middleware(RequestDecompressionMiddleware)

    @app.post("/headers")
    async def headers_endpoint(request: Request):
        return {
            "content_encoding": request.headers.get("content-encoding"),
            "content_length": request.headers.get("content-length"),
            "body_len": len(await request.body()),
        }

    body = b'{"model":"test-model","input":"hello"}'
    compressed = zstandard.ZstdCompressor().compress(body)

    with TestClient(app) as client:
        response = client.post(
            "/headers",
            headers={"Content-Type": "application/json", "Content-Encoding": "zstd"},
            content=compressed,
        )

    assert response.status_code == 200
    assert response.json() == {
        "content_encoding": None,
        "content_length": str(len(body)),
        "body_len": len(body),
    }


def test_stacked_content_encodings_are_decoded_in_reverse_order():
    app = FastAPI()
    app.add_middleware(RequestDecompressionMiddleware)

    @app.post("/echo")
    async def echo(payload=Body(...)):
        return payload

    body = b'{"model":"test-model","input":"hello"}'
    gz_buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=gz_buffer, mode="wb") as writer:
        writer.write(body)
    compressed = zlib.compress(gz_buffer.getvalue())

    with TestClient(app) as client:
        response = client.post(
            "/echo",
            headers={"Content-Type": "application/json", "Content-Encoding": "gzip, deflate"},
            content=compressed,
        )

    assert response.status_code == 200
    assert response.json() == {"model": "test-model", "input": "hello"}

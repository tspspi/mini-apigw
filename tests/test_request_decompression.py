import zstandard

from fastapi import Body, FastAPI
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

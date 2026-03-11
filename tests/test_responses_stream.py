import asyncio
import json

import pytest

from gateway.config import BackendDefinition, BackendSupports, ResponsesShimBackendConfig
from gateway.responses import (
    ResponsesJobRegistry,
    ResponsesShim,
    ResponsesStateStore,
    ResponsesToolRegistry,
    ShimContext,
)


class _FakeClient:
    async def chat(self, model: str, payload, stream: bool = False):
        if not stream:
            raise AssertionError("stream flag expected in test")

        async def _generator():
            chunk = {
                "choices": [
                    {
                        "delta": {
                            "content": [
                                {"type": "output_text", "text": "Hello"},
                                {"type": "output_text", "text": " world"},
                            ]
                        }
                    }
                ]
            }
            yield b"data: " + json.dumps(chunk).encode("utf-8") + b"\n\n"
            yield b"data: [DONE]\n\n"

        return _generator()


@pytest.mark.asyncio
async def test_responses_shim_stream_produces_responses_events():
    backend = BackendDefinition(
        type="shim",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(chat=["shim:model"]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="chat"),
    )
    ctx = ShimContext(backend=backend, backend_model="shim:model", shim_config=backend.responses_shim)
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    fake_client = _FakeClient()
    stream = await shim.handle(
        ctx,
        fake_client,
        {"model": "shim-model", "input": [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]},
        stream=True,
    )
    events = []
    async for chunk in stream:
        data = chunk.decode("utf-8").strip()
        if not data.startswith("data: "):
            continue
        payload = data[6:]
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    assert events[0]["type"] == "response.created"
    assert events[1]["type"] == "response.output_text.delta"
    assert events[1]["delta"] == "Hello world"
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hello world"

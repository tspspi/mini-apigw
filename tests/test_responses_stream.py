import asyncio
import json

import pytest
from openai.types.responses.response import Response

from gateway.backends.base import BackendUsage
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
    assert events[0]["response"]["output"][0]["id"].startswith("msg-")
    assert events[0]["response"]["output"][0]["status"] == "completed"
    assert events[0]["response"]["output"][0]["content"][0]["annotations"] == []
    assert events[1]["type"] == "response.output_text.delta"
    assert events[1]["delta"] == "Hello world"
    assert events[-1]["type"] == "response.completed"
    assert events[-1]["response"]["output"][0]["content"][0]["text"] == "Hello world"


def test_responses_shim_non_stream_payload_matches_openai_response_schema():
    backend = BackendDefinition(
        type="shim",
        name="shim-backend",
        base_url="http://shim",
        supports=BackendSupports(chat=["shim:model"]),
        responses_shim=ResponsesShimBackendConfig(enabled=True, operation="chat"),
    )
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    )
    parsed = Response.model_validate(body)
    assert parsed.status == "completed"
    assert parsed.output[0].content[0].text == "Hello world"
    assert parsed.usage.input_tokens == 3


def test_responses_shim_coerces_iso_created_at_to_numeric_timestamp():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "created_at": "2026-07-02T17:34:14.089848Z",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5},
        },
    )
    assert isinstance(body["created_at"], int)


def test_single_response_stream_merges_backend_usage():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
        },
    )

    async def collect():
        stream = shim._single_response_stream(body, BackendUsage(input_tokens=2, output_tokens=3, total_tokens=5))
        events = []
        async for chunk in stream:
            data = chunk.decode("utf-8").strip()
            if not data.startswith("data: "):
                continue
            payload = data[6:]
            if payload == "[DONE]":
                continue
            events.append(json.loads(payload))
        return events

    events = asyncio.run(collect())
    assert events[-1]["response"]["usage"]["input_tokens"] == 2
    assert events[-1]["response"]["usage"]["output_tokens"] == 3
    assert events[-1]["response"]["usage"]["total_tokens"] == 5


def test_responses_shim_omits_usage_when_backend_does_not_report_it():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
        },
    )
    assert body["usage"] == {}


def test_responses_shim_preserves_partial_usage_fields():
    shim = ResponsesShim(ResponsesStateStore(), ResponsesToolRegistry(), ResponsesJobRegistry())
    body = shim._normalize_chat_response(
        "shim-model",
        {
            "id": "chatcmpl-test",
            "choices": [{"message": {"role": "assistant", "content": "Hello world"}}],
            "usage": {"input_tokens": 3},
        },
    )
    assert body["usage"] == {
        "input_tokens": 3,
        "input_tokens_details": {"cached_tokens": 0},
    }

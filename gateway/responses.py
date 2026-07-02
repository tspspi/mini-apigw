"""Responses shim scaffolding."""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, AsyncIterator

from .backends.base import BackendClient, BackendResult
from .config import BackendDefinition, ResponsesShimBackendConfig


@dataclass(slots=True)
class ShimContext:
    backend: BackendDefinition
    backend_model: str
    shim_config: ResponsesShimBackendConfig


class ResponsesStateStore:
    async def get(self, app_id: str, conversation_id: str) -> Optional[Dict[str, Any]]:
        return None

    async def append(self, app_id: str, conversation_id: str, block: Dict[str, Any]) -> None:
        return None

    async def prune(self, app_id: str, conversation_id: str) -> None:
        return None


class ResponsesJobRegistry:
    async def start(self, job_id: str, payload: Dict[str, Any]) -> None:
        return None

    async def update(self, job_id: str, status: str, payload: Optional[Dict[str, Any]] = None) -> None:
        return None

    async def fetch(self, job_id: str) -> Optional[Dict[str, Any]]:
        return None


class ResponsesToolRegistry:
    def register(self, adapter: Any) -> None:
        return None

    def allowed_tools(self, app, backend: BackendDefinition) -> list[Any]:
        return []

    async def execute(
        self,
        tool_binding: Any,
        invocation: Dict[str, Any],
        *,
        trace_ctx: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError


class ResponsesShim:
    def __init__(
        self,
        state_store: ResponsesStateStore,
        tool_registry: ResponsesToolRegistry,
        job_registry: ResponsesJobRegistry,
    ):
        self._state_store = state_store
        self._tool_registry = tool_registry
        self._job_registry = job_registry

    async def handle(
        self,
        ctx: ShimContext,
        client: BackendClient,
        payload: Dict[str, Any],
        *,
        stream: bool,
    ) -> BackendResult | Iterable[bytes]:
        operation = ctx.shim_config.operation or "chat"
        normalized_payload = dict(payload)
        normalized_payload.pop("model", None)
        model_name = payload.get("model")
        if operation == "chat":
            chat_payload = self._translate_chat_payload(normalized_payload)
            if stream:
                stream_result = await client.chat(ctx.backend_model, chat_payload, stream=True)
                if hasattr(stream_result, '__aiter__'):
                    return self._chat_stream_to_responses(model_name, stream_result)
                if isinstance(stream_result, BackendResult):
                    body = self._normalize_chat_response(model_name, stream_result.body)
                    return self._single_response_stream(body, stream_result.usage)
                raise RuntimeError("Responses shim expected streaming iterator or BackendResult")
            result = await client.chat(ctx.backend_model, chat_payload, stream=False)
            body = self._normalize_chat_response(model_name, result.body)
            return BackendResult(body=body, usage=result.usage)
        if operation == "completions":
            completion_payload = self._translate_completion_payload(normalized_payload)
            result = await client.completions(ctx.backend_model, completion_payload)
            body = self._normalize_completion_response(model_name, result.body)
            return BackendResult(body=body, usage=result.usage)
        raise RuntimeError(f"Responses shim unsupported operation '{operation}'")

    def _translate_chat_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request: Dict[str, Any] = {k: v for k, v in payload.items() if k not in {"input", "instructions"}}
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            messages = self._build_messages_from_input(payload)
        instructions = payload.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            system_message = {"role": "system", "content": instructions.strip()}
            messages = [system_message] + messages
        request["messages"] = messages
        return request

    def _translate_completion_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = dict(payload)
        if "prompt" not in request:
            input_value = payload.get("input")
            if isinstance(input_value, str):
                request["prompt"] = input_value
            elif isinstance(input_value, list):
                request["prompt"] = "\n".join(self._coerce_text_fragment(item) for item in input_value)
        return request

    def _build_messages_from_input(self, payload: Dict[str, Any]) -> list[Dict[str, Any]]:
        input_value = payload.get("input")
        if isinstance(input_value, list):
            messages: list[Dict[str, Any]] = []
            for entry in input_value:
                if not isinstance(entry, dict):
                    continue
                role = entry.get("role") or "user"
                parts = entry.get("content")
                if isinstance(parts, list):
                    text = "".join(self._coerce_text_fragment(part) for part in parts)
                else:
                    text = self._coerce_text_fragment(parts)
                text = text.strip()
                if not text:
                    continue
                messages.append({"role": role, "content": text})
            if messages:
                return messages
        text = ""
        if isinstance(input_value, str):
            text = input_value
        if not text:
            text = payload.get("instructions") or ""
        if not text:
            return []
        return [{"role": "user", "content": text}]

    def _coerce_text_fragment(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            item_type = value.get("type")
            if item_type in {"text", "input_text", "output_text"}:
                text_value = value.get("text") or value.get("value")
                if isinstance(text_value, str):
                    return text_value
            if item_type == "message":
                inner = value.get("content")
                if isinstance(inner, list):
                    return "".join(self._coerce_text_fragment(part) for part in inner)
        return ""

    def _normalize_chat_response(self, model: Optional[str], body: Dict[str, Any]) -> Dict[str, Any]:
        choices = body.get("choices") or []
        if choices:
            message = choices[0].get("message") or {}
            role = message.get("role", "assistant")
            content = message.get("content")
            if isinstance(content, list):
                text = "".join(self._coerce_text_fragment(item) for item in content)
            elif isinstance(content, str):
                text = content
            else:
                text = message.get("text") or ""
        else:
            role = "assistant"
            text = body.get("text") or ""
        output = [self._build_output_message(role, text)]
        response_id = body.get("id") or f"resp-{uuid.uuid4().hex}"
        usage = self._normalize_usage(body.get("usage"))
        return {
            "id": response_id,
            "created_at": self._normalize_created_at(body.get("created_at")),
            "object": "response",
            "model": model,
            "output": output,
            "parallel_tool_calls": bool(body.get("parallel_tool_calls", False)),
            "tool_choice": body.get("tool_choice") or "auto",
            "tools": body.get("tools") or [],
            "status": body.get("status") or "completed",
            "usage": usage,
        }

    def _normalize_completion_response(self, model: Optional[str], body: Dict[str, Any]) -> Dict[str, Any]:
        text = body.get("text") or body.get("completion") or ""
        choices = body.get("choices") or []
        if choices:
            text = choices[0].get("text") or text
        output = [self._build_output_message("assistant", text)]
        response_id = body.get("id") or f"resp-{uuid.uuid4().hex}"
        usage = self._normalize_usage(body.get("usage"))
        return {
            "id": response_id,
            "created_at": self._normalize_created_at(body.get("created_at")),
            "object": "response",
            "model": model,
            "output": output,
            "parallel_tool_calls": bool(body.get("parallel_tool_calls", False)),
            "tool_choice": body.get("tool_choice") or "auto",
            "tools": body.get("tools") or [],
            "status": body.get("status") or "completed",
            "usage": usage,
        }

    @staticmethod
    def _build_output_message(role: str, text: str) -> Dict[str, Any]:
        return {
            "id": f"msg-{uuid.uuid4().hex}",
            "type": "message",
            "role": role if role == "assistant" else "assistant",
            "status": "completed",
            "content": [
                {
                    "type": "output_text",
                    "text": text,
                    "annotations": [],
                }
            ],
        }

    @staticmethod
    def _normalize_usage(usage: Any) -> Dict[str, Any]:
        payload = dict(usage or {}) if isinstance(usage, dict) else {}
        input_tokens = int(payload.get("input_tokens") or 0)
        output_tokens = int(payload.get("output_tokens") or 0)
        total_tokens = payload.get("total_tokens")
        if total_tokens is None:
            total_tokens = input_tokens + output_tokens
        return {
            "input_tokens": int(input_tokens),
            "input_tokens_details": payload.get("input_tokens_details") or {"cached_tokens": 0},
            "output_tokens": int(output_tokens),
            "output_tokens_details": payload.get("output_tokens_details") or {"reasoning_tokens": 0},
            "total_tokens": int(total_tokens),
        }

    @staticmethod
    def _normalize_created_at(value: Any) -> int:
        if isinstance(value, (int, float)):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if text:
                try:
                    return int(float(text))
                except ValueError:
                    pass
                try:
                    if text.endswith("Z"):
                        text = text[:-1] + "+00:00"
                    return int(datetime.fromisoformat(text).timestamp())
                except ValueError:
                    pass
        return int(time.time())


    def _chat_stream_to_responses(self, model: Optional[str], source: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
        response_id = f"resp-{uuid.uuid4().hex}"
        usage: Dict[str, Any] = {}

        async def _generator() -> AsyncIterator[bytes]:
            snapshot = {
                "id": response_id,
                "model": model,
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]}
                ],
            }
            yield self._encode_sse({"type": "response.created", "response": snapshot})
            text_parts: list[str] = []
            async for chunk in source:
                for payload in self._extract_sse_payloads(chunk):
                    delta_text = self._extract_chat_delta(payload)
                    if delta_text:
                        text_parts.append(delta_text)
                        yield self._encode_sse(
                            {
                                "type": "response.output_text.delta",
                                "delta": delta_text,
                                "output_index": 0,
                                "content_index": 0,
                                "response": {"id": response_id, "model": model},
                            }
                        )
                    self._capture_stream_usage(usage, payload.get("usage"))
            final_body = self._build_stream_response_body(model, response_id, "".join(text_parts), usage)
            payload = {"type": "response.completed", "response": final_body}
            if final_body.get("usage"):
                payload["usage"] = final_body["usage"]
            yield self._encode_sse(payload)
            yield b"data: [DONE]\n\n"

        return _generator()

    def _build_stream_response_body(
        self, model: Optional[str], response_id: str, text: str, usage: Dict[str, Any]
    ) -> Dict[str, Any]:
        base = {"id": response_id, "choices": [{"message": {"role": "assistant", "content": text}}]}
        if usage:
            base["usage"] = usage
        return self._normalize_chat_response(model, base)

    @staticmethod
    def _encode_sse(payload: Dict[str, Any]) -> bytes:
        data = json.dumps(payload).encode("utf-8")
        return b"data: " + data + b"\n\n"

    @staticmethod
    def _extract_sse_payloads(chunk: bytes) -> Iterable[Optional[Dict[str, Any]]]:
        text = chunk.decode("utf-8", errors="ignore")
        for block in text.split("\n\n"):
            if not block.strip():
                continue
            data_lines: list[str] = []
            for line in block.splitlines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
            if not data_lines:
                continue
            data_str = "\n".join(data_lines).strip()
            if not data_str or data_str == "[DONE]":
                continue
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue


    def _single_response_stream(self, body: Dict[str, Any], usage: Any) -> AsyncIterator[bytes]:
        response_id = body.get("id") or f"resp-{uuid.uuid4().hex}"
        model = body.get("model")
        usage_payload: Dict[str, Any] = dict(body.get("usage") or {})
        if isinstance(usage, dict):
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = usage.get(key)
                if value is not None:
                    usage_payload.setdefault(key, value)
        payload_body = dict(body)
        payload_body["id"] = response_id
        if usage_payload:
            payload_body["usage"] = usage_payload

        async def _generator() -> AsyncIterator[bytes]:
            output_list = payload_body.get("output")
            if not isinstance(output_list, list) or not output_list:
                output_list = [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": ""}]}
                ]
                payload_body["output"] = output_list
            else:
                first_content = output_list[0].get("content") if isinstance(output_list[0], dict) else None
                if not isinstance(first_content, list) or not first_content:
                    output_list[0]["content"] = [{"type": "output_text", "text": ""}]
            snapshot = {"id": response_id, "model": model, "output": output_list}
            yield self._encode_sse({"type": "response.created", "response": snapshot})
            text = self._extract_body_output_text(payload_body)
            if text:
                yield self._encode_sse(
                    {
                        "type": "response.output_text.delta",
                        "delta": text,
                        "output_index": 0,
                        "content_index": 0,
                        "response": {"id": response_id, "model": model},
                    }
                )
            payload = {"type": "response.completed", "response": payload_body}
            if usage_payload:
                payload["usage"] = usage_payload
            yield self._encode_sse(payload)
            yield b"data: [DONE]\n\n"

        return _generator()

    def _extract_body_output_text(self, body: Dict[str, Any]) -> str:
        output = body.get("output")
        if isinstance(output, list):
            for item in output:
                if item.get("type") != "message":
                    continue
                content = item.get("content")
                if isinstance(content, list):
                    for part in content:
                        if part.get("type") == "output_text":
                            text = part.get("text")
                            if isinstance(text, str):
                                return text
        return ""

    def _extract_chat_delta(self, payload: Dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return ""
        fragments: list[str] = []
        for choice in choices:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, list):
                for item in content:
                    fragments.append(self._coerce_text_fragment(item))
            elif isinstance(content, dict):
                fragments.append(self._coerce_text_fragment(content))
            elif isinstance(content, str):
                fragments.append(content)
        return "".join(fragments)

    @staticmethod
    def _capture_stream_usage(storage: Dict[str, Any], usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                storage[key] = value


__all__ = [
    "ResponsesJobRegistry",
    "ResponsesShim",
    "ResponsesStateStore",
    "ResponsesToolRegistry",
    "ShimContext",
]

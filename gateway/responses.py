"""Responses shim scaffolding."""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

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
        if operation == "chat":
            chat_payload = self._translate_chat_payload(normalized_payload)
            result = await client.chat(ctx.backend_model, chat_payload, stream=False)
            body = self._normalize_chat_response(payload.get("model"), result.body)
            return BackendResult(body=body, usage=result.usage)
        if operation == "completions":
            completion_payload = self._translate_completion_payload(normalized_payload)
            result = await client.completions(ctx.backend_model, completion_payload)
            body = self._normalize_completion_response(payload.get("model"), result.body)
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
            if item_type in {"text", "input_text"}:
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
        output = [
            {
                "type": "message",
                "role": role,
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ]
        response_id = body.get("id") or f"resp-{uuid.uuid4().hex}"
        usage = body.get("usage") or {}
        return {
            "id": response_id,
            "object": "response",
            "model": model,
            "output": output,
            "usage": usage,
        }

    def _normalize_completion_response(self, model: Optional[str], body: Dict[str, Any]) -> Dict[str, Any]:
        text = body.get("text") or body.get("completion") or ""
        choices = body.get("choices") or []
        if choices:
            text = choices[0].get("text") or text
        output = [
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                    }
                ],
            }
        ]
        response_id = body.get("id") or f"resp-{uuid.uuid4().hex}"
        usage = body.get("usage") or {}
        return {
            "id": response_id,
            "object": "response",
            "model": model,
            "output": output,
            "usage": usage,
        }


__all__ = [
    "ResponsesJobRegistry",
    "ResponsesShim",
    "ResponsesStateStore",
    "ResponsesToolRegistry",
    "ShimContext",
]

"""Responses shim scaffolding."""
from __future__ import annotations

import copy
import json
import time
import uuid
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, AsyncIterator

from .backends.base import BackendClient, BackendResult, BackendUsage
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
    _RESPONSES_FALLBACK_ALLOWED_FIELDS = {
        "background",
        "conversation",
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "max_tool_calls",
        "metadata",
        "parallel_tool_calls",
        "previous_response_id",
        "prompt",
        "prompt_cache_key",
        "prompt_cache_retention",
        "reasoning",
        "safety_identifier",
        "service_tier",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "text",
        "tool_choice",
        "tools",
        "top_logprobs",
        "top_p",
        "truncation",
        "user",
    }
    _CHAT_FALLBACK_ALLOWED_FIELDS = {
        "audio",
        "format",
        "frequency_penalty",
        "function_call",
        "functions",
        "images",
        "keep_alive",
        "logit_bias",
        "logprobs",
        "max_completion_tokens",
        "max_tokens",
        "metadata",
        "messages",
        "min_p",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "modalities",
        "n",
        "num_predict",
        "options",
        "parallel_tool_calls",
        "penalize_newline",
        "presence_penalty",
        "repeat_penalty",
        "response_format",
        "seed",
        "service_tier",
        "stop",
        "store",
        "stream",
        "stream_options",
        "temperature",
        "tool_choice",
        "tools",
        "top_k",
        "top_logprobs",
        "top_p",
        "user",
        "web_search_options",
    }
    _CHAT_FALLBACK_DROP_FIELDS = {
        "include",
        "input",
        "instructions",
        "max_output_tokens",
        "previous_response_id",
        "prompt_cache_key",
        "reasoning",
        "safety_identifier",
        "text",
        "truncation",
    }

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
            chat_payload = self._translate_chat_payload(normalized_payload, backend_type=ctx.backend.type)
            if stream:
                stream_result = await client.chat(ctx.backend_model, chat_payload, stream=True)
                if hasattr(stream_result, '__aiter__'):
                    return self._chat_stream_to_responses(model_name, stream_result)
                if isinstance(stream_result, BackendResult):
                    body = self._normalize_chat_response(model_name, stream_result.body)
                    self._merge_usage_into_response(body, stream_result.usage)
                    return self._single_response_stream(body, stream_result.usage)
                raise RuntimeError("Responses shim expected streaming iterator or BackendResult")
            result = await client.chat(ctx.backend_model, chat_payload, stream=False)
            body = self._normalize_chat_response(model_name, result.body)
            self._merge_usage_into_response(body, result.usage)
            return BackendResult(body=body, usage=result.usage)
        if operation == "completions":
            completion_payload = self._translate_completion_payload(normalized_payload)
            result = await client.completions(ctx.backend_model, completion_payload)
            body = self._normalize_completion_response(model_name, result.body)
            self._merge_usage_into_response(body, result.usage)
            return BackendResult(body=body, usage=result.usage)
        if operation == "responses":
            responses_payload = self._translate_responses_payload(normalized_payload)
            result = await client.responses(ctx.backend_model, responses_payload, stream=False)
            if not isinstance(result, BackendResult):
                raise RuntimeError("Responses shim expected non-stream BackendResult for responses fallback")
            return result
        raise RuntimeError(f"Responses shim unsupported operation '{operation}'")

    def _translate_chat_payload(self, payload: Dict[str, Any], backend_type: Optional[str] = None) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            k: v
            for k, v in payload.items()
            if k in self._CHAT_FALLBACK_ALLOWED_FIELDS and k not in self._CHAT_FALLBACK_DROP_FIELDS
        }
        max_output_tokens = payload.get("max_output_tokens")
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            if backend_type == "ollama":
                request.setdefault("num_predict", max_output_tokens)
            elif "max_tokens" not in request and "max_completion_tokens" not in request:
                request["max_tokens"] = max_output_tokens
        response_format = self._translate_text_to_response_format(payload.get("text"))
        if response_format is not None and "response_format" not in request:
            request["response_format"] = response_format
        messages = request.get("messages")
        if not isinstance(messages, list) or not messages:
            messages = self._build_messages_from_input(payload)
        instructions = payload.get("instructions")
        if isinstance(instructions, str) and instructions.strip():
            system_message = {"role": "system", "content": instructions.strip()}
            messages = [system_message] + messages
        request["messages"] = [self._normalize_chat_message(message) for message in messages]
        tools = request.get("tools")
        if isinstance(tools, list):
            request["tools"] = self._translate_chat_tools(tools)
        return request

    @staticmethod
    def _translate_text_to_response_format(text_config: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(text_config, dict):
            return None
        format_config = text_config.get("format")
        if not isinstance(format_config, dict):
            return None
        format_type = format_config.get("type")
        if format_type == "json_object":
            return {"type": "json_object"}
        if format_type == "json_schema":
            response_format = {"type": "json_schema"}
            schema = format_config.get("schema")
            if schema is not None:
                response_format["json_schema"] = {
                    "name": format_config.get("name") or "structured_response",
                    "schema": schema,
                }
            elif isinstance(format_config.get("json_schema"), dict):
                response_format["json_schema"] = format_config["json_schema"]
            return response_format
        return None

    def _translate_completion_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = dict(payload)
        if "prompt" not in request:
            input_value = payload.get("input")
            if isinstance(input_value, str):
                request["prompt"] = input_value
            elif isinstance(input_value, list):
                request["prompt"] = "\n".join(self._coerce_text_fragment(item) for item in input_value)
        return request

    def _translate_responses_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request: Dict[str, Any] = {
            key: value
            for key, value in payload.items()
            if key in self._RESPONSES_FALLBACK_ALLOWED_FIELDS
        }
        client_metadata = payload.get("client_metadata")
        if "metadata" not in request and isinstance(client_metadata, dict):
            request["metadata"] = client_metadata
        input_value = request.get("input")
        if isinstance(input_value, list):
            request["input"] = [self._normalize_responses_input_item(item) for item in input_value]
        return request

    def _normalize_responses_input_item(self, item: Any) -> Any:
        if not isinstance(item, dict):
            return item
        normalized = dict(item)
        if normalized.get("type") != "message":
            return normalized
        content = normalized.get("content")
        if isinstance(content, str):
            return normalized
        if not isinstance(content, list):
            return normalized
        normalized["content"] = [self._normalize_responses_message_part(part) for part in content]
        return normalized

    def _normalize_responses_message_part(self, part: Any) -> Any:
        if isinstance(part, str):
            return {"type": "input_text", "text": part}
        if not isinstance(part, dict):
            return {"type": "input_text", "text": self._coerce_text_fragment(part)}
        normalized = dict(part)
        part_type = normalized.get("type")
        if part_type in {"output_text", "text"}:
            return {
                "type": "input_text",
                "text": normalized.get("text") or normalized.get("value") or "",
            }
        return normalized

    def _build_messages_from_input(self, payload: Dict[str, Any]) -> list[Dict[str, Any]]:
        input_value = payload.get("input")
        if isinstance(input_value, list):
            messages: list[Dict[str, Any]] = []
            for entry in input_value:
                if not isinstance(entry, dict):
                    continue
                entry_type = entry.get("type")
                if entry_type == "function_call":
                    tool_call = self._responses_function_call_to_chat(entry)
                    if tool_call is None:
                        continue
                    if messages and messages[-1].get("role") == "assistant":
                        existing = messages[-1].setdefault("tool_calls", [])
                        if isinstance(existing, list):
                            existing.append(tool_call)
                        messages[-1].setdefault("content", "")
                    else:
                        messages.append({"role": "assistant", "content": "", "tool_calls": [tool_call]})
                    continue
                if entry_type == "function_call_output":
                    tool_message = self._responses_function_call_output_to_chat(entry)
                    if tool_message is not None:
                        messages.append(tool_message)
                    continue
                if entry_type == "message":
                    role = self._normalize_chat_role(entry.get("role"))
                    content = entry.get("content")
                    text = "".join(self._coerce_text_fragment(part) for part in content) if isinstance(content, list) else self._coerce_text_fragment(content)
                    text = text.strip()
                    message: Dict[str, Any] = {"role": role, "content": text}
                    tool_call = self._responses_message_function_call_to_chat(entry)
                    if tool_call is not None:
                        message["tool_calls"] = [tool_call]
                    if text or tool_call is not None:
                        messages.append(message)
                    continue
                role = self._normalize_chat_role(entry.get("role"))
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

    @staticmethod
    def _normalize_chat_role(role: Any) -> str:
        if role == "developer":
            return "system"
        if isinstance(role, str) and role:
            return role
        return "user"

    def _normalize_chat_message(self, message: Any) -> Dict[str, Any]:
        if not isinstance(message, dict):
            return {"role": "user", "content": self._coerce_text_fragment(message)}
        normalized = dict(message)
        normalized["role"] = self._normalize_chat_role(message.get("role"))
        return normalized

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
            message = {}
            role = "assistant"
            text = body.get("text") or ""
        output = self._build_response_output(message, role, text)
        response_id = body.get("id") or f"resp-{uuid.uuid4().hex}"
        usage = self._normalize_usage(body.get("usage"))
        response = {
            "id": response_id,
            "created_at": self._normalize_created_at(body.get("created_at")),
            "object": "response",
            "model": model,
            "output": output,
            "parallel_tool_calls": bool(body.get("parallel_tool_calls", False)),
            "tool_choice": body.get("tool_choice") or "auto",
            "tools": body.get("tools") or [],
            "status": body.get("status") or "completed",
        }
        if usage:
            response["usage"] = usage
        return response

    def _normalize_completion_response(self, model: Optional[str], body: Dict[str, Any]) -> Dict[str, Any]:
        text = body.get("text") or body.get("completion") or ""
        choices = body.get("choices") or []
        if choices:
            text = choices[0].get("text") or text
        output = [self._build_output_message("assistant", text)]
        response_id = body.get("id") or f"resp-{uuid.uuid4().hex}"
        usage = self._normalize_usage(body.get("usage"))
        response = {
            "id": response_id,
            "created_at": self._normalize_created_at(body.get("created_at")),
            "object": "response",
            "model": model,
            "output": output,
            "parallel_tool_calls": bool(body.get("parallel_tool_calls", False)),
            "tool_choice": body.get("tool_choice") or "auto",
            "tools": body.get("tools") or [],
            "status": body.get("status") or "completed",
        }
        if usage:
            response["usage"] = usage
        return response

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

    def _build_response_output(self, message: Dict[str, Any], role: str, text: str) -> list[Dict[str, Any]]:
        output: list[Dict[str, Any]] = []
        tool_calls = message.get("tool_calls")
        has_tool_calls = isinstance(tool_calls, list) and bool(tool_calls)
        if text or not has_tool_calls:
            output.append(self._build_output_message(role, text))
        if has_tool_calls:
            for tool_call in tool_calls:
                item = self._build_function_call_output_item(tool_call)
                if item is not None:
                    output.append(item)
        return output

    @staticmethod
    def _build_function_call_output_item(tool_call: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(tool_call, dict):
            return None
        function_data = tool_call.get("function")
        if not isinstance(function_data, dict):
            return None
        name = function_data.get("name")
        if not isinstance(name, str) or not name:
            return None
        arguments = function_data.get("arguments")
        if isinstance(arguments, dict):
            arguments_text = json.dumps(arguments)
        elif isinstance(arguments, str):
            arguments_text = arguments
        elif arguments is None:
            arguments_text = "{}"
        else:
            arguments_text = json.dumps(arguments)
        call_id = tool_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"call-{uuid.uuid4().hex}"
        return {
            "id": f"fc-{uuid.uuid4().hex}",
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments_text,
            "status": "completed",
        }

    @staticmethod
    def _responses_function_call_to_chat(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            return None
        arguments = entry.get("arguments")
        if isinstance(arguments, dict):
            arguments_text = json.dumps(arguments)
        elif isinstance(arguments, str):
            arguments_text = arguments
        elif arguments is None:
            arguments_text = "{}"
        else:
            arguments_text = json.dumps(arguments)
        call_id = entry.get("call_id") or entry.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"toolcall-{uuid.uuid4().hex}"
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments_text,
            },
        }

    @staticmethod
    def _responses_function_call_output_to_chat(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        call_id = entry.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            return None
        output = entry.get("output")
        if isinstance(output, str):
            output_text = output
        elif output is None:
            output_text = ""
        else:
            output_text = json.dumps(output)
        return {
            "role": "tool",
            "tool_call_id": call_id,
            "content": output_text,
        }

    def _responses_message_function_call_to_chat(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        content = entry.get("content")
        if not isinstance(content, list):
            return None
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") != "function_call":
                continue
            return self._responses_function_call_to_chat(part)
        return None

    @staticmethod
    def _translate_chat_tools(tools: list[Any]) -> list[Dict[str, Any]]:
        translated: list[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type")
            if tool_type == "namespace":
                nested_tools = tool.get("tools")
                if isinstance(nested_tools, list):
                    translated.extend(ResponsesShim._translate_chat_tools(nested_tools))
                continue
            function_payload = tool.get("function")
            if isinstance(function_payload, dict):
                if tool_type in {None, "function"}:
                    normalized = dict(tool)
                    normalized["type"] = "function"
                    normalized["function"] = function_payload
                    translated.append(normalized)
                continue
            if tool_type not in {None, "function"}:
                continue
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                continue
            normalized_tool = {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description") or "",
                    "parameters": tool.get("parameters") or {},
                },
            }
            if "strict" in tool:
                normalized_tool["function"]["strict"] = bool(tool.get("strict"))
            translated.append(normalized_tool)
        return translated

    @staticmethod
    def _normalize_usage(usage: Any) -> Dict[str, Any]:
        payload = dict(usage or {}) if isinstance(usage, dict) else {}
        input_tokens = payload.get("input_tokens")
        output_tokens = payload.get("output_tokens")
        total_tokens = payload.get("total_tokens")
        if input_tokens is None and output_tokens is None and total_tokens is None:
            return {}
        normalized: Dict[str, Any] = {}
        if input_tokens is not None:
            normalized["input_tokens"] = int(input_tokens)
            normalized["input_tokens_details"] = payload.get("input_tokens_details") or {"cached_tokens": 0}
        if output_tokens is not None:
            normalized["output_tokens"] = int(output_tokens)
            normalized["output_tokens_details"] = payload.get("output_tokens_details") or {"reasoning_tokens": 0}
        if total_tokens is None and input_tokens is not None and output_tokens is not None:
            total_tokens = int(input_tokens) + int(output_tokens)
        if total_tokens is not None:
            normalized["total_tokens"] = int(total_tokens)
        return normalized

    @classmethod
    def _normalize_backend_usage(cls, usage: Any) -> Dict[str, Any]:
        if isinstance(usage, BackendUsage):
            payload: Dict[str, Any] = {}
            if usage.input_tokens is not None:
                payload["input_tokens"] = usage.input_tokens
            if usage.output_tokens is not None:
                payload["output_tokens"] = usage.output_tokens
            if usage.total_tokens is not None:
                payload["total_tokens"] = usage.total_tokens
            return cls._normalize_usage(payload)
        return cls._normalize_usage(usage)

    @classmethod
    def _merge_usage_into_response(cls, body: Dict[str, Any], usage: Any) -> None:
        merged = cls._normalize_backend_usage(body.get("usage"))
        normalized_usage = cls._normalize_backend_usage(usage)
        for key, value in normalized_usage.items():
            if key.endswith("_details"):
                merged.setdefault(key, value)
            elif value is not None and merged.get(key) in (None, 0):
                merged[key] = value
        if "total_tokens" not in merged:
            input_tokens = merged.get("input_tokens")
            output_tokens = merged.get("output_tokens")
            if input_tokens is not None and output_tokens is not None:
                merged["total_tokens"] = int(input_tokens) + int(output_tokens)
        if merged:
            body["usage"] = merged
        else:
            body.pop("usage", None)

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
            text_parts: list[str] = []
            async for chunk in source:
                for payload in self._extract_sse_payloads(chunk):
                    delta_text = self._extract_chat_delta(payload)
                    if delta_text:
                        text_parts.append(delta_text)
                    self._capture_stream_usage(usage, payload.get("usage"))
            final_body = self._build_stream_response_body(model, response_id, "".join(text_parts), usage)
            for payload in self._build_response_stream_events(final_body):
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
        payload_body = dict(body)
        payload_body["id"] = response_id
        self._merge_usage_into_response(payload_body, usage)

        async def _generator() -> AsyncIterator[bytes]:
            for payload in self._build_response_stream_events(payload_body):
                yield self._encode_sse(payload)
            yield b"data: [DONE]\n\n"

        return _generator()

    def _build_response_stream_events(self, body: Dict[str, Any]) -> list[Dict[str, Any]]:
        final_body = copy.deepcopy(body)
        output_items = final_body.get("output")
        if not isinstance(output_items, list):
            output_items = []
            final_body["output"] = output_items

        created_response = dict(final_body)
        created_response["output"] = []
        created_response["status"] = "in_progress"
        created_response.pop("usage", None)

        events: list[Dict[str, Any]] = []
        sequence_number = 0

        def push(event: Dict[str, Any]) -> None:
            nonlocal sequence_number
            event["sequence_number"] = sequence_number
            sequence_number += 1
            events.append(event)

        push({"type": "response.created", "response": created_response})

        for output_index, item in enumerate(output_items):
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                working_item = copy.deepcopy(item)
                content_items = working_item.get("content")
                if not isinstance(content_items, list):
                    content_items = []
                final_content_items = copy.deepcopy(content_items)
                working_item["content"] = []
                working_item["status"] = "in_progress"
                push(
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": working_item,
                    }
                )
                for content_index, part in enumerate(final_content_items):
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type != "output_text":
                        continue
                    item_id = item.get("id") or working_item.get("id") or f"msg-{uuid.uuid4().hex}"
                    empty_part = {"type": "output_text", "text": "", "annotations": part.get("annotations") or []}
                    push(
                        {
                            "type": "response.content_part.added",
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "part": empty_part,
                        }
                    )
                    text = part.get("text")
                    if isinstance(text, str) and text:
                        push(
                            {
                                "type": "response.output_text.delta",
                                "item_id": item_id,
                                "output_index": output_index,
                                "content_index": content_index,
                                "delta": text,
                                "logprobs": [],
                            }
                        )
                    push(
                        {
                            "type": "response.output_text.done",
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "text": part.get("text") if isinstance(part.get("text"), str) else "",
                            "logprobs": [],
                        }
                    )
                    push(
                        {
                            "type": "response.content_part.done",
                            "item_id": item_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "part": part,
                        }
                    )
                push(
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": item,
                    }
                )
            elif item_type == "function_call":
                working_item = copy.deepcopy(item)
                item_id = working_item.get("id") or f"fc-{uuid.uuid4().hex}"
                working_item["id"] = item_id
                arguments = working_item.get("arguments")
                if isinstance(arguments, str):
                    arguments_text = arguments
                elif arguments is None:
                    arguments_text = ""
                else:
                    arguments_text = json.dumps(arguments)
                working_item["arguments"] = ""
                working_item["status"] = "in_progress"
                push(
                    {
                        "type": "response.output_item.added",
                        "output_index": output_index,
                        "item": working_item,
                    }
                )
                if arguments_text:
                    push(
                        {
                            "type": "response.function_call_arguments.delta",
                            "item_id": item_id,
                            "output_index": output_index,
                            "delta": arguments_text,
                        }
                    )
                push(
                    {
                        "type": "response.function_call_arguments.done",
                        "item_id": item_id,
                        "output_index": output_index,
                        "arguments": arguments_text,
                        "name": item.get("name") or "",
                    }
                )
                push(
                    {
                        "type": "response.output_item.done",
                        "output_index": output_index,
                        "item": item,
                    }
                )

        completed_event = {"type": "response.completed", "response": final_body}
        if final_body.get("usage"):
            completed_event["usage"] = final_body["usage"]
        push(completed_event)
        return events

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

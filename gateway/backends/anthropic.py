"""Anthropic backend client."""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

import requests

from .base import BackendClient, BackendResult, BackendUsage


class AnthropicBackend(BackendClient):
    def __init__(self, definition):
        super().__init__(definition)
        if not definition.api_key:
            raise ValueError("Anthropic backend requires api_key")
        self._session = requests.Session()
        self._session.headers.update(
            {
                "x-api-key": definition.api_key,
                "content-type": "application/json",
                "anthropic-version": definition.metadata.get("anthropic_version", "2023-06-01"),
            }
        )

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.definition.base_url.rstrip('/')}/{path.lstrip('/')}"
        response = self._session.post(url, json=payload, timeout=self.definition.request_timeout_s or 600)
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            try:
                detail_payload = response.json()
            except ValueError:
                detail_payload = response.text
            raise RuntimeError({"status_code": response.status_code, "detail": detail_payload}) from exc
        body = response.json()
        return self._build_response(dict(body))

    def _post_stream(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.definition.base_url.rstrip('/')}/{path.lstrip('/')}"
        timeout = self.definition.request_timeout_s or 600
        with self._session.post(url, json=payload, timeout=timeout, stream=True) as response:
            response.raise_for_status()
            message: Dict[str, Any] | None = None
            usage: Dict[str, Any] | None = None
            stop_reason: Optional[str] = None
            stop_sequence: Optional[str] = None

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line or raw_line.startswith(":"):
                    continue
                if not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[5:].strip()
                if not data_str:
                    continue
                if data_str == "[DONE]":
                    break

                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                event_type = event.get("type")
                if event_type == "message_start":
                    message = event.get("message", {}) or {}
                    if not isinstance(message.get("content"), list):
                        message["content"] = []
                elif event_type == "content_block_start":
                    if message is None:
                        message = {"content": []}
                    index = event.get("index", 0)
                    content_block = event.get("content_block", {}) or {}
                    block = self._ensure_content_capacity(message, index)
                    block.clear()
                    block.update(content_block)
                elif event_type == "content_block_delta":
                    if message is None:
                        continue
                    index = event.get("index", 0)
                    delta = event.get("delta", {}) or {}
                    block = self._ensure_content_capacity(message, index)
                    delta_type = delta.get("type")
                    if delta_type == "text_delta":
                        block["type"] = "text"
                        block["text"] = block.get("text", "") + delta.get("text", "")
                    elif delta_type == "tool_use_delta":
                        block["type"] = "tool_use"
                        block.setdefault("input", {})
                        input_delta = delta.get("input") or {}
                        if isinstance(block["input"], dict) and isinstance(input_delta, dict):
                            block["input"].update(input_delta)
                    elif delta_type == "input_json_delta":
                        block["type"] = "tool_use"
                        chunk = delta.get("partial_json")
                        if isinstance(chunk, str):
                            buffer = block.setdefault("_input_json_buffer", "")
                            block["_input_json_buffer"] = buffer + chunk
                    else:
                        for key, value in delta.items():
                            if key == "type":
                                continue
                            block[key] = value
                elif event_type == "message_delta":
                    if message is None:
                        message = {}
                    delta = event.get("delta", {}) or {}
                    for key, value in delta.items():
                        if value is not None:
                            message[key] = value
                elif event_type == "message_stop":
                    usage = event.get("usage") or usage
                    stop_reason = event.get("stop_reason") or stop_reason
                    stop_sequence = event.get("stop_sequence") or stop_sequence
                elif event_type == "error":
                    error = event.get("error", {}) or {}
                    detail = error.get("message") or error.get("type") or "Unknown streaming error"
                    raise RuntimeError(f"Anthropic streaming error: {detail}")

            if message is None:
                raise RuntimeError("Anthropic streaming response missing message_start event")

            message.setdefault("type", "message")
            message.setdefault("role", "assistant")
            message.setdefault("content", [])
            message.setdefault("model", payload.get("model"))
            self._finalize_stream_tool_inputs(message)
            return self._build_response(
                message,
                usage=usage,
                finish_reason=stop_reason,
                stop_sequence=stop_sequence,
            )

    def _ensure_content_capacity(self, message: Dict[str, Any], index: int) -> Dict[str, Any]:
        content = message.setdefault("content", [])
        if not isinstance(content, list):
            content = []
            message["content"] = content
        while len(content) <= index:
            content.append({})
        return content[index]

    def _build_response(
        self,
        message: Dict[str, Any],
        *,
        usage: Optional[Dict[str, Any]] = None,
        finish_reason: Optional[str] = None,
        stop_sequence: Optional[str] = None,
    ) -> Dict[str, Any]:
        working_message = dict(message)
        payload: Dict[str, Any] = {}

        # Prefer explicit arguments, fall back to values embedded in the message body.
        embedded_usage = working_message.pop("usage", None)
        final_usage = usage or embedded_usage
        if final_usage is not None:
            payload["usage"] = final_usage

        embedded_reason = working_message.pop("stop_reason", None)
        final_reason = finish_reason or embedded_reason

        embedded_stop_sequence = working_message.pop("stop_sequence", None)
        final_stop_sequence = stop_sequence or embedded_stop_sequence
        if final_stop_sequence is not None:
            payload["stop_sequence"] = final_stop_sequence

        normalized_message = self._normalize_response_message(working_message)
        payload["message"] = normalized_message

        if final_reason is None and normalized_message.get("tool_calls"):
            final_reason = "tool_calls"
        elif final_reason == "tool_use" and normalized_message.get("tool_calls"):
            final_reason = "tool_calls"

        if final_reason is not None:
            payload["finish_reason"] = final_reason

        return payload

    def _normalize_response_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(message)
        content_blocks = normalized.get("content")
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []

        if isinstance(content_blocks, list):
            for block in content_blocks:
                if isinstance(block, dict):
                    block_type = block.get("type")
                    if block_type == "text":
                        text = block.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
                    elif block_type == "tool_use":
                        name = block.get("name")
                        if not isinstance(name, str) or not name:
                            continue
                        call_id = block.get("id")
                        if not isinstance(call_id, str) or not call_id:
                            call_id = f"toolu_{uuid.uuid4().hex}"
                        arguments = block.get("input")
                        if isinstance(arguments, dict):
                            arguments_payload = json.dumps(arguments)
                        else:
                            arguments_payload = json.dumps({})
                        tool_calls.append(
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {"name": name, "arguments": arguments_payload},
                            }
                        )
                    elif block_type == "thinking":
                        # Drop thinking content from user-visible response.
                        continue
                    else:
                        text_parts.append(json.dumps(block))
                elif isinstance(block, str):
                    text_parts.append(block)
        elif isinstance(content_blocks, str):
            text_parts.append(content_blocks)
        elif content_blocks is not None:
            text_parts.append(str(content_blocks))

        normalized["content"] = "".join(text_parts)
        if tool_calls:
            normalized["tool_calls"] = tool_calls
        else:
            normalized.pop("tool_calls", None)
        return normalized

    def _finalize_stream_tool_inputs(self, message: Dict[str, Any]) -> None:
        content_blocks = message.get("content")
        if not isinstance(content_blocks, list):
            return
        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") != "tool_use":
                continue
            buffer = block.pop("_input_json_buffer", None)
            if not buffer:
                continue
            try:
                parsed = json.loads(buffer)
            except json.JSONDecodeError:
                parsed = {}
            if isinstance(parsed, dict):
                block["input"] = parsed
            else:
                block["input"] = {}

    def _convert_messages(
        self,
        messages: Any,
        *,
        system_prompt: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        if not isinstance(messages, list):
            raise ValueError("Anthropic chat payload expects 'messages' as a list")

        converted: List[Dict[str, Any]] = []
        tool_id_map: Dict[str, str] = {}
        pending_tool_results: List[Dict[str, Any]] = []
        active_system = system_prompt

        for original in messages:
            if not isinstance(original, dict):
                continue
            role = original.get("role")
            if pending_tool_results and role != "tool":
                converted.append({"role": "user", "content": list(pending_tool_results)})
                pending_tool_results.clear()
            if role == "system":
                system_text = self._message_content_to_text(original.get("content"))
                if not system_text:
                    continue
                if active_system is None:
                    active_system = system_text
                    continue
                fallback = dict(original)
                fallback["role"] = "user"
                fallback.setdefault("content", system_text)
                converted.append(self._convert_default_message(fallback))
                continue
            if role == "assistant":
                converted.append(self._convert_assistant_message(original, tool_id_map))
            elif role == "tool":
                tool_block = self._convert_tool_message(original, tool_id_map)
                if tool_block:
                    pending_tool_results.append(tool_block)
            else:
                converted.append(self._convert_default_message(original))

        if pending_tool_results:
            converted.append({"role": "user", "content": list(pending_tool_results)})

        return converted, active_system

    def _convert_assistant_message(self, message: Dict[str, Any], tool_id_map: Dict[str, str]) -> Dict[str, Any]:
        new_message: Dict[str, Any] = {"role": "assistant"}
        blocks = self._as_blocks(message.get("content"), allow_empty=True)

        for block in blocks:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                block_id = block.get("id")
                if isinstance(block_id, str):
                    tool_id_map.setdefault(block_id, block_id)

        tool_calls = message.get("tool_calls") or []
        has_tool_use_block = any(
            isinstance(block, dict) and block.get("type") == "tool_use" for block in blocks
        )
        if isinstance(tool_calls, list) and tool_calls:
            if not has_tool_use_block:
                for call in tool_calls:
                    tool_block = self._convert_tool_use_block(call, tool_id_map)
                    if tool_block:
                        blocks.append(tool_block)
            else:
                for call in tool_calls:
                    call_id = call.get("id") if isinstance(call, dict) else None
                    if isinstance(call_id, str):
                        tool_id_map.setdefault(call_id, call_id)

        if not blocks and not tool_calls:
            blocks = [{"type": "text", "text": " "}]

        new_message["content"] = blocks
        return new_message

    def _convert_tool_message(self, message: Dict[str, Any], tool_id_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
        tool_call_id = message.get("tool_call_id") or message.get("id")
        if not isinstance(tool_call_id, str):
            return None

        mapped_id = tool_id_map.get(tool_call_id, tool_call_id)
        block: Dict[str, Any] = {
            "type": "tool_result",
            "tool_use_id": mapped_id,
        }

        content = message.get("content")
        converted_content = self._convert_tool_result_content(content)
        if converted_content:
            block["content"] = converted_content

        if isinstance(message.get("is_error"), bool):
            block["is_error"] = message["is_error"]

        return block

    def _convert_default_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        role = message.get("role", "user")
        content = message.get("content")
        converted: Dict[str, Any] = {"role": role}

        if isinstance(content, list):
            converted["content"] = self._as_blocks(content, allow_empty=True)
        else:
            converted["content"] = content

        return converted

    def _convert_tool_use_block(self, call: Dict[str, Any], tool_id_map: Dict[str, str]) -> Optional[Dict[str, Any]]:
        if not isinstance(call, dict):
            return None

        function_data = call.get("function") or {}
        name = function_data.get("name") or call.get("name")
        if not isinstance(name, str) or not name:
            return None

        call_id = call.get("id")
        if isinstance(call_id, str) and call_id:
            tool_id = call_id
        else:
            tool_id = f"toolu_{uuid.uuid4().hex}"

        if isinstance(call_id, str) and call_id:
            tool_id_map.setdefault(call_id, tool_id)
        tool_id_map.setdefault(tool_id, tool_id)

        arguments = function_data.get("arguments")
        parsed_arguments: Any = {}
        if isinstance(arguments, str):
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = arguments
            parsed_arguments = parsed if isinstance(parsed, dict) else {"value": parsed}
        elif isinstance(arguments, dict):
            parsed_arguments = arguments
        elif arguments is not None:
            parsed_arguments = {"value": arguments}

        return {
            "type": "tool_use",
            "id": tool_id,
            "name": name,
            "input": parsed_arguments if isinstance(parsed_arguments, dict) else {},
        }

    def _convert_tool_result_content(self, content: Any) -> List[Dict[str, Any]]:
        return self._as_blocks(content, allow_empty=True)

    def _message_content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            fragments: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type in {"text", "input_text"}:
                        text = item.get("text") or item.get("value")
                        if isinstance(text, str):
                            fragments.append(text)
                    elif item_type == "tool_result":
                        nested = self._message_content_to_text(item.get("content"))
                        if nested:
                            fragments.append(nested)
                elif isinstance(item, str):
                    fragments.append(item)
            return "\n".join(part for part in fragments if part)
        if content is None:
            return ""
        return str(content)

    def _as_blocks(self, content: Any, allow_empty: bool = False) -> List[Dict[str, Any]]:
        blocks: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type"):
                    blocks.append(item)
                elif isinstance(item, str):
                    stripped = item.strip()
                    if stripped:
                        blocks.append({"type": "text", "text": item})
                elif item is not None:
                    text_value = str(item)
                    if text_value:
                        blocks.append({"type": "text", "text": text_value})
        elif isinstance(content, dict) and content.get("type"):
            blocks.append(content)
        elif isinstance(content, str):
            stripped = content.strip()
            if stripped:
                blocks.append({"type": "text", "text": content})
        elif content is not None:
            text_value = str(content)
            if text_value:
                blocks.append({"type": "text", "text": text_value})

        if not blocks and not allow_empty:
            blocks.append({"type": "text", "text": " "})
        return blocks

    def _convert_tools(self, tools: Any) -> Optional[List[Dict[str, Any]]]:
        if not isinstance(tools, list):
            return None

        converted: List[Dict[str, Any]] = []
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            tool_type = tool.get("type")
            if tool_type == "function":
                fn = tool.get("function") or {}
                name = fn.get("name")
                if not isinstance(name, str) or not name:
                    continue
                entry: Dict[str, Any] = {"name": name}
                description = fn.get("description")
                if isinstance(description, str) and description:
                    entry["description"] = description
                parameters = fn.get("parameters")
                if isinstance(parameters, dict):
                    entry["input_schema"] = parameters
                converted.append(entry)
            else:
                converted.append(tool)

        return converted or None

    def _convert_tool_choice(self, tool_choice: Any) -> Optional[Dict[str, Any]]:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            if tool_choice in {"auto", "any", "none"}:
                return {"type": tool_choice}
            return {"type": "tool", "name": tool_choice}
        if isinstance(tool_choice, dict):
            choice_type = tool_choice.get("type")
            if choice_type == "function":
                fn = tool_choice.get("function") or {}
                name = fn.get("name")
                if isinstance(name, str) and name:
                    return {"type": "tool", "name": name}
                return {"type": "auto"}
            if choice_type in {"auto", "any", "none"}:
                return {"type": choice_type}
            if choice_type == "tool":
                name = tool_choice.get("name")
                if isinstance(name, str) and name:
                    return {"type": "tool", "name": name}
        return None

    # Structured output is achieved by injecting a synthetic tool that mirrors
    # the requested schema and forcing the assistant to call it. We rely on the
    # upstream client to provide a sensible JSON schema and only normalize it
    # enough for Anthropic's expectations.
    def _apply_response_format(
        self,
        response_format: Any,
        *,
        tools: Optional[List[Dict[str, Any]]],
        tool_choice: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[Dict[str, Any]]]:
        structured = self._prepare_structured_output_tool(response_format, tools)
        if structured is None:
            return tools, tool_choice

        new_tool, selected_name = structured
        updated_tools: List[Dict[str, Any]]
        if tools:
            existing_names = {t.get("name") for t in tools if isinstance(t, dict)}
            if selected_name not in existing_names:
                updated_tools = list(tools)
                updated_tools.append(new_tool)
            else:
                updated_tools = list(tools)
        else:
            updated_tools = [new_tool]

        if tool_choice is None or (
            isinstance(tool_choice, dict)
            and tool_choice.get("type") in {"auto", "any"}
        ):
            tool_choice = {"type": "tool", "name": selected_name}

        return updated_tools, tool_choice

    def _prepare_structured_output_tool(
        self,
        response_format: Any,
        tools: Optional[List[Dict[str, Any]]],
    ) -> Optional[Tuple[Dict[str, Any], str]]:
        normalized = self._normalize_response_format_spec(response_format)
        if normalized is None:
            return None

        base_name, schema = normalized
        name = self._dedupe_tool_name(base_name, tools)
        sanitized_schema = self._resolve_json_schema(schema) if schema else {"type": "object"}
        if not sanitized_schema:
            sanitized_schema = {"type": "object"}

        tool = {
            "name": name,
            "description": "Gateway-injected tool to enforce structured output.",
            "input_schema": sanitized_schema,
        }
        return tool, name

    def _normalize_response_format_spec(self, response_format: Any) -> Optional[Tuple[str, Dict[str, Any]]]:
        if response_format is None:
            return None

        if isinstance(response_format, str):
            if response_format in {"json", "json_object"}:
                return "structured_response", {"type": "object"}
            return None

        if not isinstance(response_format, dict):
            return None

        fmt_type = response_format.get("type")

        if fmt_type == "json_object" or (
            fmt_type is None and response_format.get("format") == "json_object"
        ):
            name = response_format.get("name") or "structured_response"
            return name, {"type": "object"}

        if fmt_type == "json_schema":
            schema_container = response_format.get("json_schema") or {}
            schema = schema_container.get("schema") or response_format.get("schema")
            if schema is None:
                schema = {"type": "object"}
            name = schema_container.get("name") or response_format.get("name") or "structured_response"
            return name, schema

        if fmt_type is None and any(key in response_format for key in {"type", "properties"}):
            name = response_format.get("name") or "structured_response"
            return name, response_format

        return None

    def _dedupe_tool_name(self, base_name: str, tools: Optional[List[Dict[str, Any]]]) -> str:
        if tools is None:
            return base_name
        existing = {
            tool.get("name")
            for tool in tools
            if isinstance(tool, dict) and isinstance(tool.get("name"), str)
        }
        if base_name not in existing:
            return base_name

        index = 2
        while True:
            candidate = f"{base_name}_{index}"
            if candidate not in existing:
                return candidate
            index += 1

    def _resolve_json_schema(self, schema: Any) -> Any:
        if not isinstance(schema, dict):
            return schema

        defs = schema.get("$defs") or schema.get("definitions") or {}
        inlined = self._inline_refs(schema, defs)
        inlined.pop("$defs", None)
        inlined.pop("definitions", None)
        return self._sanitize_schema(inlined)

    def _inline_refs(self, node: Any, defs: Dict[str, Any]) -> Any:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if ref.startswith("#/$defs/") or ref.startswith("#/definitions/"):
                    key = ref.split("/")[-1]
                    target = defs.get(key)
                    if target is None:
                        return {}
                    return self._inline_refs(target, defs)
                return {}
            return {k: self._inline_refs(v, defs) for k, v in node.items() if k != "$ref"}
        if isinstance(node, list):
            return [self._inline_refs(item, defs) for item in node]
        return node

    def _sanitize_schema(self, node: Any) -> Any:
        if isinstance(node, dict):
            sanitized: Dict[str, Any] = {}
            for key, value in node.items():
                if key in {"$schema", "$id", "$defs", "definitions"}:
                    continue
                if key == "additionalProperties":
                    if isinstance(value, bool):
                        sanitized[key] = value
                    else:
                        sanitized[key] = self._sanitize_schema(value)
                    continue
                sanitized[key] = self._sanitize_schema(value)
            return sanitized
        if isinstance(node, list):
            return [self._sanitize_schema(item) for item in node]
        return node

    async def chat(self, model: str, payload: Dict[str, Any], stream: bool = False):
        messages = payload.get("messages")
        if messages is None:
            raise ValueError("Anthropic chat payload requires 'messages'")

        stream_requested = stream or bool(payload.get("stream", False))
        existing_system = payload.get("system")
        converted_messages, system_prompt = self._convert_messages(
            messages,
            system_prompt=existing_system if isinstance(existing_system, str) and existing_system else None,
        )
        request_body: Dict[str, Any] = {
            "model": model,
            "messages": converted_messages,
            "max_tokens": payload.get("max_tokens", 1024),
            "stream": stream_requested,
        }
        if "temperature" in payload and payload["temperature"] is not None:
            request_body["temperature"] = payload["temperature"]

        if system_prompt:
            request_body["system"] = system_prompt

        tools = self._convert_tools(payload.get("tools"))
        tool_choice = self._convert_tool_choice(payload.get("tool_choice"))

        response_format = payload.get("response_format")
        if response_format is not None:
            tools, tool_choice = self._apply_response_format(
                response_format,
                tools=tools,
                tool_choice=tool_choice,
            )

        if tools:
            request_body["tools"] = tools

        if tool_choice:
            request_body["tool_choice"] = tool_choice

        if stream_requested:
            body = await self._asyncify(self._post_stream, "/v1/messages", request_body)
        else:
            body = await self._asyncify(self._post, "/v1/messages", request_body)

        usage = BackendUsage(
            input_tokens=body.get("usage", {}).get("input_tokens"),
            output_tokens=body.get("usage", {}).get("output_tokens"),
        )
        return BackendResult(body=body, usage=usage)

    async def completions(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError("Anthropic does not expose OpenAI completions API")

    async def embeddings(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError("Anthropic embeddings not supported")

    async def images(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError("Anthropic images not supported")

    async def models(self) -> Dict[str, Any]:
        url = f"{self.definition.base_url.rstrip('/')}/v1/models"
        response = await self._asyncify(self._session.get, url, timeout=self.definition.request_timeout_s or 600)
        response.raise_for_status()
        return response.json()


__all__ = ["AnthropicBackend"]

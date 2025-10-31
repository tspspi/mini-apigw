from __future__ import annotations

import json
import uuid
from copy import deepcopy
from typing import Any, Dict, List, Tuple

from ollama import Client, RequestError, ResponseError

from .base import BackendClient, BackendResult, BackendUsage


class OllamaBackend(BackendClient):
    _OPTION_KEYS = {
        "temperature",
        "top_p",
        "top_k",
        "repeat_penalty",
        "presence_penalty",
        "frequency_penalty",
        "seed",
        "num_predict",
        "stop",
        "min_p",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "penalize_newline",
    }

    def __init__(self, definition):
        super().__init__(definition)
        client_kwargs = {"host": definition.base_url}
        if definition.request_timeout_s is not None:
            client_kwargs["timeout"] = definition.request_timeout_s
        self._client = Client(**client_kwargs)

    async def chat(self, model: str, payload: Dict[str, Any], stream: bool = False):
        request_payload = self._prepare_chat_payload(model, payload, stream)

        def _call():
            try:
                response = self._client.chat(
                    model=model,
                    messages=request_payload.get("messages"),
                    tools=request_payload.get("tools"),
                    stream=False,
                    format=request_payload.get("format"),
                    options=request_payload.get("options"),
                    keep_alive=request_payload.get("keep_alive"),
                )
            except (RequestError, ResponseError) as exc:
                raise RuntimeError(f"Ollama chat error: {exc}") from exc
            body = response.model_dump()
            body.setdefault("message", body.get("message"))
            body.setdefault("model", response.model)
            return body

        body = await self._asyncify(_call)
        body = self._coerce_chat_response(body)
        usage = BackendUsage(
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
        )
        return BackendResult(body=body, usage=usage)

    async def completions(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        request_payload = self._prepare_generate_payload(model, payload)

        def _call():
            try:
                response = self._client.generate(**request_payload)
            except (RequestError, ResponseError) as exc:
                raise RuntimeError(f"Ollama generate error: {exc}") from exc
            return response.model_dump()

        body = await self._asyncify(_call)
        usage = BackendUsage(
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
        )
        return BackendResult(body=body, usage=usage)

    async def embeddings(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        prompts, request_template = self._prepare_embeddings_payload(model, payload)

        def _call():
            data: List[Dict[str, Any]] = []
            created = None
            for index, prompt in enumerate(prompts):
                request = dict(request_template)
                request["prompt"] = prompt
                try:
                    response = self._client.embeddings(**request)
                except (RequestError, ResponseError) as exc:
                    raise RuntimeError(f"Ollama embeddings error: {exc}") from exc
                body = response.model_dump()
                embedding = body.get("embedding")
                if embedding is None:
                    continue
                if created is None and body.get("created") is not None:
                    created = body.get("created")
                data.append(
                    {
                        "object": body.get("object", "embedding"),
                        "index": index,
                        "embedding": embedding,
                    }
                )

            aggregated: Dict[str, Any] = {
                "object": "list",
                "model": request_template.get("model", model),
                "data": data,
            }
            if created is not None:
                aggregated["created"] = created
            return aggregated

        body = await self._asyncify(_call)
        return BackendResult(body=body)

    async def images(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError("Ollama does not support image generation")

    async def models(self) -> Dict[str, Any]:
        def _call():
            try:
                response = self._client.list()
            except (RequestError, ResponseError) as exc:
                raise RuntimeError(f"Ollama list models error: {exc}") from exc
            return response.model_dump()

        data = await self._asyncify(_call)
        models = data.get("models") or data.get("data") or []
        return {"data": models}

    def _prepare_chat_payload(self, model: str, payload: Dict[str, Any], stream: bool) -> Dict[str, Any]:
        request_payload = deepcopy(payload)
        request_payload["model"] = model
        request_payload["stream"] = False

        tools = request_payload.get("tools")
        functions: List[Dict[str, Any]] = []
        if tools:
            for tool in tools:
                if not isinstance(tool, dict):
                    continue
                if tool.get("type") != "function":
                    continue
                fn = tool.get("function", {})
                name = fn.get("name")
                if not name:
                    continue
                functions.append(
                    {
                        "name": name,
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {}),
                    }
                )
        if functions:
            request_payload["functions"] = functions

        messages = []
        for original in request_payload.get("messages", []) or []:
            message = deepcopy(original)
            tool_calls = message.pop("tool_calls", None)
            if tool_calls:
                call = tool_calls[0]
                fn = call.get("function", {}) if isinstance(call, dict) else {}
                name = fn.get("name") or ""
                arguments = fn.get("arguments") or "{}"
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                elif not isinstance(arguments, str):
                    arguments = json.dumps(arguments)
                message["function_call"] = {"name": name, "arguments": arguments}
                message["content"] = ""
            if message.get("role") == "tool":
                name = message.get("name") or message.get("tool_call_id") or ""
                message["name"] = name
                message["role"] = "tool"
            message.pop("tool_call_id", None)
            text, images = self._extract_multimodal_content(message.get("content"))
            if images:
                message["images"] = images
            message["content"] = text
            messages.append(message)
        request_payload["messages"] = messages
        options = self._extract_options(request_payload)
        if options:
            request_payload["options"] = options
        else:
            request_payload.pop("options", None)
        format_value = self._extract_format(request_payload)
        if format_value is not None:
            request_payload["format"] = format_value
        else:
            request_payload.pop("format", None)
        return request_payload

    def _prepare_generate_payload(self, model: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        request: Dict[str, Any] = {"model": model, "stream": False}
        request.update(
            {
                key: payload[key]
                for key in [
                    "prompt",
                    "suffix",
                    "system",
                    "template",
                    "context",
                    "raw",
                    "format",
                    "images",
                    "keep_alive",
                ]
                if key in payload
            }
        )

        # OpenAI-style payloads may use "input" for prompt.
        if "prompt" not in request:
            prompt = payload.get("input")
            if isinstance(prompt, list):
                prompt = "\n".join(str(item) for item in prompt)
            if prompt is not None:
                request["prompt"] = prompt

        options = dict(payload.get("options") or {})
        for key in self._OPTION_KEYS:
            if key in payload:
                options[key] = payload[key]
        if options:
            request["options"] = options

        format_value = self._extract_format(payload)
        if format_value is not None:
            request["format"] = format_value

        return request

    def _prepare_embeddings_payload(self, model: str, payload: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        raw_input = payload.get("input")
        if raw_input is None:
            raw_input = payload.get("prompt")

        prompts: List[str] = []
        if isinstance(raw_input, list):
            for item in raw_input:
                if item is None:
                    continue
                prompts.append(str(item))
        elif raw_input is None:
            prompts.append("")
        else:
            prompts.append(str(raw_input))

        request: Dict[str, Any] = {"model": model}

        if payload.get("keep_alive") is not None:
            request["keep_alive"] = payload["keep_alive"]

        options = dict(payload.get("options") or {})
        for key in self._OPTION_KEYS:
            if key in payload:
                options[key] = payload[key]
        if options:
            request["options"] = options

        return prompts, request

    def _extract_multimodal_content(self, content: Any) -> Tuple[str, List[str]]:
        if not isinstance(content, list):
            if isinstance(content, str):
                return content, []
            return "" if content is None else str(content), []

        text_parts: List[str] = []
        images: List[str] = []
        for item in content:
            if isinstance(item, str):
                text_parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type") or ""
            if item_type in {"text", "input_text"}:
                text = item.get("text") or item.get("value")
                if isinstance(text, str):
                    text_parts.append(text)
            elif item_type in {"image", "image_url", "input_image"}:
                image_spec = item.get("image_url") or item.get("image") or {}
                if isinstance(image_spec, str):
                    url = image_spec
                elif isinstance(image_spec, dict):
                    url = image_spec.get("url") or image_spec.get("data")
                else:
                    url = None
                if isinstance(url, str):
                    prefix = "base64,"
                    if "data:" in url and prefix in url:
                        url = url.split(prefix, 1)[1]
                    images.append(url)
        text_content = "\n".join(part.strip() for part in text_parts if part.strip())
        return text_content, images

    def _extract_options(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        options = dict(payload.get("options") or {})
        for key in list(payload.keys()):
            if key in self._OPTION_KEYS:
                options[key] = payload.pop(key)
        return options

    def _extract_format(self, payload: Dict[str, Any]) -> Any:
        explicit = payload.pop("format", None)
        if explicit is not None:
            return explicit

        response_format = payload.pop("response_format", None)
        if not isinstance(response_format, dict):
            return None

        fmt_type = response_format.get("type")
        if fmt_type == "json_object":
            return "json"
        if fmt_type == "json_schema":
            schema = response_format.get("json_schema")
            if isinstance(schema, dict):
                return schema.get("schema") or schema
            return response_format.get("schema")
        # fallback: pass through raw structure
        return response_format

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
            if "$ref" in node:
                ref = node["$ref"]
                if ref.startswith("#/$defs/") or ref.startswith("#/definitions/"):
                    key = ref.split("/")[-1]
                    target = defs.get(key)
                    if target is None:
                        return {}
                    return self._inline_refs(target, defs)
                return {}
            return {k: self._inline_refs(v, defs) for k, v in node.items()}
        if isinstance(node, list):
            return [self._inline_refs(item, defs) for item in node]
        return node

    def _sanitize_schema(self, node: Any) -> Any:
        if isinstance(node, dict):
            allowed_keys = {"type", "properties", "required", "items", "enum", "additionalProperties"}
            sanitized = {}
            for key, value in node.items():
                if key in {"title", "examples", "description", "$schema", "$id", "$comment", "default", "deprecated", "readOnly", "writeOnly", "format", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minItems", "maxItems", "minLength", "maxLength", "pattern", "const"}:
                    continue
                if key not in allowed_keys:
                    continue
                if key == "properties" and isinstance(value, dict):
                    sanitized[key] = {prop: self._sanitize_schema(schema) for prop, schema in value.items()}
                elif key == "additionalProperties":
                    if isinstance(value, bool):
                        sanitized[key] = value
                    else:
                        sanitized[key] = self._sanitize_schema(value)
                else:
                    sanitized[key] = self._sanitize_schema(value)
            return sanitized
        if isinstance(node, list):
            return [self._sanitize_schema(item) for item in node]
        return node

    @staticmethod
    def _coerce_chat_response(body: Dict[str, Any]) -> Dict[str, Any]:
        if "choices" in body:
            return body

        message = body.get("message") or {}
        if isinstance(message, dict):
            OllamaBackend._maybe_promote_function_call(message)
            OllamaBackend._normalize_tool_calls(message)
            if "role" not in message:
                message["role"] = "assistant"

        finish_reason = "stop"
        if message.get("tool_calls"):
            finish_reason = "tool_calls"
        elif body.get("done_reason"):
            finish_reason = body.get("done_reason")

        coerced = dict(body)
        coerced["choices"] = [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ]
        return coerced

    @staticmethod
    def _normalize_tool_calls(message: Dict[str, Any]) -> None:
        # Ollama may return tool information in non-OpenAI formats. Normalize when possible.
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            normalized: List[Dict[str, Any]] = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                fn = call.get("function") or {}
                name = fn.get("name")
                if not name:
                    continue
                arguments = fn.get("arguments")
                if isinstance(arguments, dict):
                    arguments = json.dumps(arguments)
                elif arguments is None:
                    arguments = "{}"
                normalized.append(
                    {
                        "id": call.get("id") or f"toolcall-{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {
                            "name": name,
                            "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments),
                        },
                    }
                )
            if normalized:
                message["tool_calls"] = normalized
                return

        # Some Ollama versions embed tool data inside content entries.
        content_calls = OllamaBackend._extract_tool_calls_from_content(message)
        if content_calls:
            message["tool_calls"] = content_calls

    @staticmethod
    def _maybe_promote_function_call(message: Dict[str, Any]) -> None:
        function_call = message.pop("function_call", None)
        if not isinstance(function_call, dict):
            return
        name = function_call.get("name")
        if not name:
            return
        arguments = function_call.get("arguments") or "{}"
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        elif not isinstance(arguments, str):
            arguments = json.dumps(arguments)
        message["tool_calls"] = [
            {
                "id": f"toolcall-{uuid.uuid4().hex}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        ]

    @staticmethod
    def _extract_tool_calls_from_content(message: Dict[str, Any]) -> List[Dict[str, Any]]:
        content = message.get("content")
        tool_calls: List[Dict[str, Any]] = []

        if isinstance(content, list):
            remaining_items = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_call":
                    payload = item.get("tool_call") or {}
                    name = payload.get("name")
                    if not name:
                        continue
                    arguments = payload.get("arguments") or "{}"
                    if isinstance(arguments, dict):
                        arguments_json = json.dumps(arguments)
                    elif isinstance(arguments, str):
                        arguments_json = arguments
                    else:
                        arguments_json = json.dumps(arguments)
                    tool_calls.append(
                        {
                            "id": payload.get("id") or f"toolcall-{uuid.uuid4().hex}",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments_json},
                        }
                    )
                else:
                    remaining_items.append(item)
            if remaining_items != content:
                content = remaining_items
                message["content"] = remaining_items

        text_segments: List[str] = []
        if isinstance(content, str):
            text_segments = [content]
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    text_segments.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("value")
                    if isinstance(text, str):
                        text_segments.append(text)
        elif isinstance(content, dict):
            text = content.get("text") or content.get("value")
            if isinstance(text, str):
                text_segments.append(text)

        remaining_lines: List[str] = []
        for segment in text_segments:
            for line in segment.splitlines():
                stripped = line.strip().strip("`")
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    remaining_lines.append(line)
                    continue
                if not isinstance(payload, dict):
                    remaining_lines.append(line)
                    continue
                name = payload.get("name")
                if not isinstance(name, str):
                    remaining_lines.append(line)
                    continue
                arguments = payload.get("arguments")
                if arguments is None:
                    arguments = payload.get("parameters", {})
                if isinstance(arguments, dict):
                    arguments_json = json.dumps(arguments)
                elif isinstance(arguments, str):
                    arguments_json = arguments
                else:
                    arguments_json = json.dumps(arguments)
                tool_calls.append(
                    {
                        "id": f"toolcall-{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments_json},
                    }
                )

        if tool_calls:
            remaining_text = "\n".join(line for line in remaining_lines if line.strip())
            message["content"] = remaining_text
        return tool_calls


__all__ = ["OllamaBackend"]

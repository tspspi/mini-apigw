from __future__ import annotations

import json
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from .base import BackendClient, BackendResult, BackendUsage


class GoogleBackend(BackendClient):
    """Backend implementation for the Google Gemini API."""

    def __init__(self, definition):
        super().__init__(definition)
        if not definition.api_key:
            raise ValueError("Google backend requires api_key")

        base_url = (definition.base_url or "https://generativelanguage.googleapis.com").rstrip("/")
        timeout = definition.request_timeout_s or 600

        headers = {"x-goog-api-key": definition.api_key, "content-type": "application/json"}
        if definition.extra_headers:
            headers.update(dict(definition.extra_headers))

        self._client = httpx.Client(base_url=base_url, headers=headers, timeout=timeout)

    async def chat(self, model: str, payload: Dict[str, Any], stream: bool = False):
        request_body = self._build_chat_request(payload)
        path = f"/v1beta/models/{model}:generateContent"
        response_payload = await self._asyncify(self._post, path, request_body)
        normalized = self._convert_chat_response(response_payload, model)
        usage_meta = normalized.get("usage", {})
        usage = BackendUsage(
            input_tokens=usage_meta.get("prompt_tokens"),
            output_tokens=usage_meta.get("completion_tokens"),
            total_tokens=usage_meta.get("total_tokens"),
        )
        return BackendResult(body=normalized, usage=usage)

    async def completions(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        prompt = payload.get("prompt")
        if prompt is None:
            raise ValueError("Google completions payload requires 'prompt'")

        prompts = self._ensure_sequence(prompt)
        if not prompts:
            raise ValueError("Google completions payload requires at least one prompt")

        contents = [{"role": "user", "parts": [{"text": prompts[0]}]}]
        request_body = self._build_generation_request(payload, contents, system_instruction=None)
        path = f"/v1beta/models/{model}:generateContent"
        response_payload = await self._asyncify(self._post, path, request_body)
        normalized = self._convert_completions_response(response_payload, model)
        usage_meta = normalized.get("usage", {})
        usage = BackendUsage(
            input_tokens=usage_meta.get("prompt_tokens"),
            output_tokens=usage_meta.get("completion_tokens"),
            total_tokens=usage_meta.get("total_tokens"),
        )
        return BackendResult(body=normalized, usage=usage)

    async def embeddings(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raw_input = payload.get("input")
        if raw_input is None:
            raise ValueError("Google embeddings payload requires 'input'")

        inputs = self._ensure_sequence(raw_input)
        if not inputs:
            raise ValueError("Google embeddings payload requires at least one input item")

        if len(inputs) == 1:
            body = {
                "model": model,
                "content": self._build_embed_content(inputs[0]),
            }
            path = f"/v1beta/models/{model}:embedContent"
            response = await self._asyncify(self._post, path, body)
            embeddings_data = [response.get("embedding", {}).get("values") or []]
        else:
            requests = [{"content": self._build_embed_content(item)} for item in inputs]
            body = {"model": model, "requests": requests}
            path = f"/v1beta/models/{model}:batchEmbedContents"
            response = await self._asyncify(self._post, path, body)
            embeddings_data = [
                entry.get("values") or [] for entry in response.get("embeddings", [])
            ]

        data = [
            {"object": "embedding", "index": index, "embedding": vector}
            for index, vector in enumerate(embeddings_data)
        ]
        normalized = {"object": "list", "model": model, "data": data}
        return BackendResult(body=normalized)

    async def images(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError("Google Gemini API does not provide image generation")

    async def models(self) -> Dict[str, Any]:
        response = await self._asyncify(self._get, "/v1beta/models")
        models = response.get("models") or []
        entries: List[Dict[str, Any]] = []
        for model_info in models:
            if not isinstance(model_info, dict):
                continue
            full_name = model_info.get("name") or ""
            model_id = full_name.split("/")[-1] if "/" in full_name else full_name
            entry = {
                "id": model_id,
                "object": "model",
                "owned_by": "google",
            }
            if "displayName" in model_info:
                entry["display_name"] = model_info["displayName"]
            if "description" in model_info:
                entry["description"] = model_info["description"]
            entries.append(entry)
        return {"object": "list", "data": entries}

    def _build_chat_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        messages = payload.get("messages")
        if messages is None:
            raise ValueError("Google chat payload requires 'messages'")

        contents, system_instruction = self._convert_messages(messages)
        if not contents:
            raise ValueError("Google chat payload produced no messages after conversion")

        return self._build_generation_request(payload, contents, system_instruction)

    def _build_generation_request(
        self,
        payload: Dict[str, Any],
        contents: List[Dict[str, Any]],
        system_instruction: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"contents": contents}
        if system_instruction:
            body["systemInstruction"] = system_instruction

        generation_config = self._convert_generation_config(payload)
        if generation_config:
            body["generationConfig"] = generation_config

        tools = self._convert_tools(payload.get("tools"))
        if tools:
            body["tools"] = tools

        tool_config = self._convert_tool_config(payload.get("tool_choice"))
        if tool_config:
            body["toolConfig"] = tool_config

        response_format = payload.get("response_format")
        response_config = self._convert_response_format(response_format)
        if response_config:
            generation_config = body.setdefault("generationConfig", {})
            generation_config.update(response_config)

        safety_settings = payload.get("safety_settings")
        if isinstance(safety_settings, list) and safety_settings:
            body["safetySettings"] = safety_settings

        return body

    def _convert_messages(
        self, messages: Sequence[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        contents: List[Dict[str, Any]] = []
        system_parts: List[Dict[str, Any]] = []

        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            if role == "system":
                system_parts.extend(self._convert_content_parts(message.get("content")))
                continue
            if role == "assistant":
                converted = self._convert_assistant_message(message)
            elif role == "tool":
                converted = self._convert_tool_message(message)
            else:
                converted = self._convert_user_message(message)
            if converted:
                contents.append(converted)

        system_instruction = {"parts": system_parts} if system_parts else None
        return contents, system_instruction

    def _convert_user_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        parts = self._convert_content_parts(message.get("content"))
        if not parts:
            return None
        converted: Dict[str, Any] = {"role": "user", "parts": parts}
        name = message.get("name")
        if isinstance(name, str) and name:
            converted["name"] = name
        return converted

    def _convert_assistant_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        parts = self._convert_content_parts(message.get("content"))
        tool_calls = message.get("tool_calls") or []
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function") or {}
            name = function.get("name")
            if not name:
                continue
            args = self._parse_arguments(function.get("arguments"))
            parts.append({"functionCall": {"name": name, "args": args}})
        if not parts:
            return None
        return {"role": "model", "parts": parts}

    def _convert_tool_message(self, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        tool_name = message.get("name") or message.get("tool_call_id")
        content = message.get("content")
        parsed = self._try_parse_json(content)
        value = parsed if parsed is not None else content
        if value is None:
            return None
        response_payload = self._format_function_response_value(value)
        if response_payload is None:
            return None
        parts = [
            {
                "functionResponse": {
                    "name": tool_name or "tool",
                    "response": response_payload,
                }
            }
        ]
        text_value = self._stringify_tool_response(value)
        if text_value:
            parts.append({"text": text_value})
        return {"role": "user", "parts": parts}

    def _convert_content_parts(self, content: Any) -> List[Dict[str, Any]]:
        if content is None:
            return []
        if isinstance(content, str):
            return [{"text": content}]
        parts: List[Dict[str, Any]] = []
        if isinstance(content, list):
            for item in content:
                if isinstance(item, str):
                    parts.append({"text": item})
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append({"text": text})
                elif item_type == "image_url":
                    mapped = self._convert_image_url(item.get("image_url") or {})
                    if mapped:
                        parts.append(mapped)
                elif item_type == "input_text":
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append({"text": text})
                else:
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append({"text": text})
        return parts

    def _stringify_tool_response(self, value: Any) -> Optional[str]:
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)

    def _format_function_response_value(self, value: Any) -> Optional[Dict[str, Any]]:
        if isinstance(value, dict):
            return value
        if isinstance(value, list):
            return {"items": value}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return {"result": value}
        try:
            return {"result": json.loads(value)}
        except Exception:
            return {"result": str(value)}

    def _normalize_response_schema(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(schema, dict):
            return {}
        defs = schema.get("$defs") or schema.get("definitions") or {}
        normalized = self._inline_refs(schema, defs)
        return self._sanitize_schema(normalized)

    def _inline_refs(self, node: Any, defs: Dict[str, Any]) -> Any:
        if isinstance(node, dict):
            local_defs = node.get("$defs") or node.get("definitions") or {}
            if isinstance(local_defs, dict) and local_defs:
                merged_defs = dict(defs)
                merged_defs.update(local_defs)
            else:
                merged_defs = defs
            ref = node.get("$ref")
            if isinstance(ref, str) and ref.startswith("#/"):
                parts = ref.split("/")[1:]
                target = None
                remaining: List[str] = []
                if parts and parts[0] in {"$defs", "definitions"}:
                    if len(parts) >= 2:
                        key = parts[1]
                        target = merged_defs.get(key)
                        remaining = parts[2:]
                if target is None:
                    return {}
                for segment in remaining:
                    if isinstance(target, dict):
                        target = target.get(segment)
                    elif isinstance(target, list):
                        try:
                            index = int(segment)
                        except (TypeError, ValueError):
                            target = None
                        else:
                            target = target[index] if 0 <= index < len(target) else None
                    else:
                        target = None
                    if target is None:
                        break
                if target is None:
                    return {}
                return self._inline_refs(target, merged_defs)
            inlined: Dict[str, Any] = {}
            for key, value in node.items():
                if key in {"$defs", "definitions"}:
                    continue
                inlined[key] = self._inline_refs(value, merged_defs)
            return inlined
        if isinstance(node, list):
            return [self._inline_refs(item, defs) for item in node]
        return node

    def _sanitize_schema(self, node: Any) -> Any:
        if isinstance(node, dict):
            sanitized: Dict[str, Any] = {}
            for key, value in node.items():
                if key in {"$schema", "$id", "$defs", "definitions"}:
                    continue
                if key == "additionalProperties" and value not in (True, False):
                    sanitized[key] = self._sanitize_schema(value)
                    continue
                sanitized[key] = self._sanitize_schema(value)
            return sanitized
        if isinstance(node, list):
            return [self._sanitize_schema(item) for item in node]
        return node

    def _convert_image_url(self, descriptor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(descriptor, dict):
            return None
        uri = descriptor.get("url") or descriptor.get("file")
        if not isinstance(uri, str) or not uri:
            return None
        if uri.startswith("data:"):
            if ";base64," in uri:
                header, data = uri.split(";base64,", 1)
                mime = header.split("data:", 1)[1] if "data:" in header else "application/octet-stream"
                return {"inlineData": {"mimeType": mime, "data": data}}
            return {"inlineData": {"mimeType": "application/octet-stream", "data": uri.split(",", 1)[-1]}}
        if descriptor.get("b64_json"):
            raw = descriptor["b64_json"]
            if isinstance(raw, str):
                mime = descriptor.get("mime_type") or "image/png"
                return {"inlineData": {"mimeType": mime, "data": raw}}
        return {"fileData": {"fileUri": uri}}

    def _convert_generation_config(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        config: Dict[str, Any] = {}
        temperature = payload.get("temperature")
        if temperature is not None:
            config["temperature"] = temperature
        top_p = payload.get("top_p")
        if top_p is not None:
            config["topP"] = top_p
        top_k = payload.get("top_k")
        if top_k is not None:
            config["topK"] = top_k
        max_output_tokens = payload.get("max_output_tokens") or payload.get("max_tokens")
        if max_output_tokens is not None:
            config["maxOutputTokens"] = max_output_tokens
        stop = payload.get("stop")
        if stop:
            config["stopSequences"] = self._ensure_sequence(stop)
        frequency_penalty = payload.get("frequency_penalty")
        if frequency_penalty is not None:
            config["frequencyPenalty"] = frequency_penalty
        presence_penalty = payload.get("presence_penalty")
        if presence_penalty is not None:
            config["presencePenalty"] = presence_penalty
        candidate_count = payload.get("n")
        if candidate_count is not None:
            config["candidateCount"] = candidate_count
        if candidate_count and candidate_count > 1:
            diversity = payload.get("diversity_penalty")
            if diversity is not None:
                config["diversityPenalty"] = diversity
        return {k: v for k, v in config.items() if v is not None}

    def _convert_tools(self, tools: Any) -> List[Dict[str, Any]]:
        if not isinstance(tools, list):
            return []
        function_declarations: List[Dict[str, Any]] = []
        for entry in tools:
            if not isinstance(entry, dict):
                continue
            if entry.get("type") != "function":
                continue
            function = entry.get("function") or {}
            name = function.get("name")
            if not isinstance(name, str) or not name:
                continue
            declaration: Dict[str, Any] = {"name": name}
            description = function.get("description")
            if isinstance(description, str) and description:
                declaration["description"] = description
            parameters = function.get("parameters")
            if isinstance(parameters, dict) and parameters:
                declaration["parameters"] = parameters
            function_declarations.append(declaration)
        if not function_declarations:
            return []
        return [{"functionDeclarations": function_declarations}]

    def _convert_tool_config(self, tool_choice: Any) -> Optional[Dict[str, Any]]:
        if tool_choice is None or tool_choice == "auto":
            return None
        if tool_choice == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if isinstance(tool_choice, dict):
            choice_type = tool_choice.get("type")
            if choice_type == "function":
                function = tool_choice.get("function") or {}
                name = function.get("name")
                if isinstance(name, str) and name:
                    return {"functionCallingConfig": {"mode": "ANY", "allowedFunctionNames": [name]}}
            if choice_type == "none":
                return {"functionCallingConfig": {"mode": "NONE"}}
        return None

    def _convert_response_format(self, response_format: Any) -> Optional[Dict[str, Any]]:
        if not response_format:
            return None
        if isinstance(response_format, str):
            if response_format in {"json", "json_object"}:
                return {"responseMimeType": "application/json"}
            return None
        if not isinstance(response_format, dict):
            return None

        fmt_type = response_format.get("type") or response_format.get("format")
        if fmt_type in {"json", "json_object"}:
            schema = response_format.get("json_schema", {}).get("schema") or response_format.get("schema")
            config = {"responseMimeType": "application/json"}
            normalized_schema = None
            if isinstance(schema, dict) and schema:
                normalized_schema = self._normalize_response_schema(schema)
            if normalized_schema:
                config["responseSchema"] = normalized_schema
            return config

        if fmt_type == "json_schema":
            schema_container = response_format.get("json_schema", {})
            schema = schema_container.get("schema")
            config = {"responseMimeType": "application/json"}
            normalized_schema = None
            if isinstance(schema, dict) and schema:
                normalized_schema = self._normalize_response_schema(schema)
            if normalized_schema:
                config["responseSchema"] = normalized_schema
            return config

        if any(key in response_format for key in {"properties", "required", "type"}):
            normalized = self._normalize_response_schema(dict(response_format))
            return {
                "responseMimeType": "application/json",
                "responseSchema": normalized or {},
            }
        return None

    def _convert_chat_response(self, response: Dict[str, Any], model: str) -> Dict[str, Any]:
        usage_meta = response.get("usageMetadata") or {}
        choices: List[Dict[str, Any]] = []
        candidates = response.get("candidates") or []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            message = self._convert_candidate_to_message(candidate)
            finish_reason = self._map_finish_reason(candidate.get("finishReason"))
            choices.append(
                {
                    "index": index,
                    "message": message,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            )
        if not choices:
            choices.append(
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            )
        payload: Dict[str, Any] = {
            "id": response.get("responseId") or f"gemini-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": usage_meta.get("promptTokenCount"),
                "completion_tokens": usage_meta.get("candidatesTokenCount"),
                "total_tokens": usage_meta.get("totalTokenCount"),
            },
        }
        safety = response.get("safetyRatings")
        if safety:
            payload["safety_ratings"] = safety
        return payload

    def _convert_completions_response(self, response: Dict[str, Any], model: str) -> Dict[str, Any]:
        usage_meta = response.get("usageMetadata") or {}
        candidates = response.get("candidates") or []
        choices: List[Dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            if not isinstance(candidate, dict):
                continue
            text = self._candidate_text(candidate)
            finish_reason = self._map_finish_reason(candidate.get("finishReason"))
            choices.append(
                {
                    "index": index,
                    "text": text,
                    "logprobs": None,
                    "finish_reason": finish_reason,
                }
            )
        if not choices:
            choices.append({"index": 0, "text": "", "logprobs": None, "finish_reason": "stop"})
        return {
            "id": response.get("responseId") or f"gemini-{uuid.uuid4().hex}",
            "object": "text_completion",
            "model": model,
            "choices": choices,
            "usage": {
                "prompt_tokens": usage_meta.get("promptTokenCount"),
                "completion_tokens": usage_meta.get("candidatesTokenCount"),
                "total_tokens": usage_meta.get("totalTokenCount"),
            },
        }

    def _convert_candidate_to_message(self, candidate: Dict[str, Any]) -> Dict[str, Any]:
        content = candidate.get("content") or {}
        parts = content.get("parts") or []
        text_fragments: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "text" in part:
                text = part.get("text")
                if isinstance(text, str):
                    text_fragments.append(text)
            elif "functionCall" in part:
                call = part.get("functionCall") or {}
                name = call.get("name")
                if not name:
                    continue
                args = call.get("args") or {}
                if isinstance(args, str):
                    arguments = args
                else:
                    try:
                        arguments = json.dumps(args)
                    except (TypeError, ValueError):
                        arguments = json.dumps({})
                tool_calls.append(
                    {
                        "id": f"call_{uuid.uuid4().hex}",
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
        content_text = "".join(text_fragments)
        message: Dict[str, Any] = {"role": "assistant", "content": content_text or ""}
        if tool_calls:
            message["tool_calls"] = tool_calls
            if not content_text:
                message["content"] = None
        return message

    def _candidate_text(self, candidate: Dict[str, Any]) -> str:
        message = self._convert_candidate_to_message(candidate)
        content = message.get("content")
        if isinstance(content, str):
            return content
        return "" if content is None else str(content)

    def _map_finish_reason(self, reason: Any) -> str:
        if not isinstance(reason, str):
            return "stop"
        mapping = {
            "STOP": "stop",
            "MAX_TOKENS": "length",
            "SAFETY": "content_filter",
            "RECITATION": "content_filter",
            "OTHER": "stop",
        }
        return mapping.get(reason.upper(), "stop")

    def _parse_arguments(self, arguments: Any) -> Dict[str, Any]:
        if isinstance(arguments, dict):
            return arguments
        if isinstance(arguments, str):
            try:
                return json.loads(arguments)
            except json.JSONDecodeError:
                pass
        if arguments is None:
            return {}
        return {"value": arguments}

    def _build_embed_content(self, value: Any) -> Dict[str, Any]:
        if isinstance(value, str):
            return {"parts": [{"text": value}]}
        if isinstance(value, (dict, list)):
            return {"parts": [{"text": json.dumps(value)}]}
        return {"parts": [{"text": str(value)}]}

    def _ensure_sequence(self, value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _try_parse_json(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                pass
        return value

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._client.post(path, json=payload)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(response)
            raise RuntimeError(detail) from exc
        return response.json()

    def _get(self, path: str) -> Dict[str, Any]:
        response = self._client.get(path)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = self._extract_error_detail(response)
            raise RuntimeError(detail) from exc
        return response.json()

    def _extract_error_detail(self, response: httpx.Response) -> Dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            return {"status_code": response.status_code, "detail": response.text}
        return {"status_code": response.status_code, "detail": payload}

    def __del__(self):  # pragma: no cover - destructor semantics vary
        try:
            self._client.close()
        except Exception:
            pass


__all__ = ["GoogleBackend"]

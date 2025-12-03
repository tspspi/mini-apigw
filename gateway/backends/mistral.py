from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, AsyncIterator, Dict

import httpx
from mistralai import Mistral

from .base import BackendClient, BackendResult, BackendUsage

try:  # pragma: no cover - defensive import for older mistralai versions
    from mistralai.models.mistralerror import MistralError  # type: ignore
except Exception:  # pragma: no cover - fall back when models module layout changes
    MistralError = Exception  # type: ignore


class MistralBackend(BackendClient):
    """Backend implementation for the native Mistral AI API."""

    def __init__(self, definition):
        super().__init__(definition)
        if not definition.api_key:
            raise ValueError("Mistral backend requires api_key")

        timeout = definition.request_timeout_s or 600
        headers = dict(definition.extra_headers or {})
        self._http_client = httpx.Client(timeout=timeout, headers=headers or None)

        metadata = dict(definition.metadata or {})
        server_name = metadata.get("server")
        server_url = metadata.get("server_url")
        base_url = (definition.base_url or "https://api.mistral.ai").rstrip("/")
        if not server_url:
            server_url = base_url.removesuffix("/v1")

        client_kwargs: Dict[str, Any] = {
            "api_key": definition.api_key,
            "client": self._http_client,
        }
        if server_name:
            client_kwargs["server"] = server_name
        if server_url:
            client_kwargs["server_url"] = server_url

        self._client = Mistral(**client_kwargs)

    async def chat(self, model: str, payload: Dict[str, Any], stream: bool = False):
        request_payload = self._prepare_payload(payload, stream_support=True)

        if stream and self._requires_stream_fallback(request_payload):
            # Mistral's streaming API omits tool call metadata, so fall back to the
            # non-stream flow when tools are involved to preserve correct semantics.
            stream = False

        if stream:
            return await self._stream_chat(model, request_payload)

        def _call() -> Dict[str, Any]:
            try:
                response = self._client.chat.complete(model=model, **request_payload)
                body = self._serialize(response)
                return self._normalize_tool_calls(body)
            except MistralError as exc:  # pragma: no cover - requires live API
                raise RuntimeError(f"Mistral chat error: {exc}") from exc

        body = await self._asyncify(_call)
        usage = self._extract_usage(body)
        return BackendResult(body=body, usage=usage)

    async def completions(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError("Mistral does not expose the legacy completions API")

    async def embeddings(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        request_payload = self._prepare_payload(payload, stream_support=False)

        def _call() -> Dict[str, Any]:
            try:
                response = self._client.embeddings.create(model=model, **request_payload)
                return self._serialize(response)
            except MistralError as exc:  # pragma: no cover - requires live API
                raise RuntimeError(f"Mistral embeddings error: {exc}") from exc

        body = await self._asyncify(_call)
        return BackendResult(body=body)

    async def images(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        raise NotImplementedError("Mistral does not provide image generation")

    async def models(self) -> Dict[str, Any]:
        def _call() -> Dict[str, Any]:
            try:
                response = self._client.models.list()
            except MistralError as exc:  # pragma: no cover - requires live API
                raise RuntimeError(f"Mistral list models error: {exc}") from exc
            return self._serialize(response)

        return await self._asyncify(_call)

    @staticmethod
    def _prepare_payload(payload: Dict[str, Any], *, stream_support: bool) -> Dict[str, Any]:
        request_payload = dict(payload or {})
        request_payload.pop("model", None)
        if stream_support:
            request_payload.pop("stream", None)
        return request_payload

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()  # type: ignore[attr-defined]
        if isinstance(value, dict):
            return {k: cls._serialize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [cls._serialize(item) for item in value]
        return value

    @staticmethod
    def _extract_usage(body: Any) -> BackendUsage:
        if not isinstance(body, dict):
            return BackendUsage()
        usage = body.get("usage")
        if not isinstance(usage, dict):
            return BackendUsage()
        return BackendUsage(
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
        )

    async def _stream_chat(self, model: str, payload: Dict[str, Any]) -> AsyncIterator[bytes]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[Any] = asyncio.Queue()
        sentinel = object()

        def _enqueue(value: Any) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, value)

        def _producer() -> None:
            try:
                with self._client.chat.stream(model=model, **payload) as stream:
                    for event in stream:
                        chunk = getattr(event, "data", event)
                        payload = self._coerce_stream_payload(chunk)
                        self._normalize_tool_calls(payload)
                        _enqueue(payload)
            except MistralError as exc:  # pragma: no cover - requires live API
                _enqueue(RuntimeError(f"Mistral chat stream error: {exc}"))
            except Exception as exc:  # pragma: no cover - defensive
                _enqueue(exc)
            finally:
                _enqueue(sentinel)

        loop.run_in_executor(None, _producer)

        async def _generator() -> AsyncIterator[bytes]:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                data = json.dumps(item).encode("utf-8")
                yield b"data: " + data + b"\n\n"
            yield b"data: [DONE]\n\n"

        return _generator()

    def _coerce_stream_payload(self, chunk: Any) -> Dict[str, Any]:
        serialized = self._serialize(chunk)
        if isinstance(serialized, dict):
            inner = serialized.get("data")
            if isinstance(inner, dict) and len(serialized) == 1:
                return inner
            return serialized
        return {"data": serialized}

    def _normalize_tool_calls(self, body: Any) -> Any:
        if not isinstance(body, dict):
            return body
        choices = body.get("choices")
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue
                changed = False
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    changed = self._promote_function_call(delta) or changed
                message = choice.get("message")
                if isinstance(message, dict):
                    changed = self._promote_function_call(message) or changed
                if changed and choice.get("finish_reason") in (None, "stop"):
                    choice["finish_reason"] = "tool_calls"
        message_root = body.get("message")
        if isinstance(message_root, dict):
            if self._promote_function_call(message_root) and body.get("finish_reason") in (None, "stop"):
                body["finish_reason"] = "tool_calls"
        return body

    def _promote_function_call(self, target: Dict[str, Any]) -> bool:
        fn_call = target.pop("function_call", None)
        if not isinstance(fn_call, dict):
            return False

        arguments = fn_call.get("arguments")
        if isinstance(arguments, str):
            args_str = arguments
        elif arguments is None:
            args_str = ""
        else:
            try:
                args_str = json.dumps(arguments)
            except TypeError:
                args_str = str(arguments)

        entry_index = self._next_tool_call_index(target)
        entry: Dict[str, Any] = {
            "index": entry_index,
            "id": fn_call.get("id") or f"toolcall-{uuid.uuid4().hex}",
            "type": fn_call.get("type") or "function",
            "function": {
                "name": fn_call.get("name") or "",
                "arguments": args_str,
            },
        }

        tool_calls = target.get("tool_calls")
        if not isinstance(tool_calls, list):
            tool_calls = []
            target["tool_calls"] = tool_calls
        tool_calls.append(entry)
        if target.get("content") is None:
            target["content"] = ""
        return True

    @staticmethod
    def _next_tool_call_index(target: Dict[str, Any]) -> int:
        tool_calls = target.get("tool_calls")
        if isinstance(tool_calls, list):
            indices = [entry.get("index") for entry in tool_calls if isinstance(entry, dict) and isinstance(entry.get("index"), int)]
            if indices:
                return max(indices) + 1
        return 0

    @staticmethod
    def _requires_stream_fallback(payload: Dict[str, Any]) -> bool:
        tools = payload.get("tools")
        if isinstance(tools, list) and tools:
            return True
        tool_choice = payload.get("tool_choice")
        if tool_choice:
            return True
        return False

    def __del__(self):  # pragma: no cover - destructor semantics vary
        try:
            self._http_client.close()
        except Exception:
            pass


__all__ = ["MistralBackend"]

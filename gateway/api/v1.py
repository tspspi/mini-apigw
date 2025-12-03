"""
    Public OpenAI-compatible API layer. This does _not_ implement all 
    features of OpenAI but for the endpoints it implements it supports
    the same API.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Iterable, List, Optional, Sequence

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..accounting import AccountingRecord
from ..auth import AuthError
from ..backends.base import BackendResult, BackendUsage
from ..runtime import GatewayRuntime
from ..trace import TracePayload

router = APIRouter(prefix="/v1")


async def get_runtime(request: Request) -> GatewayRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Gateway not ready")
    return runtime


def _require_api_key(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    if not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Invalid Authorization header")
    return authorization.split(" ", 1)[1].strip()


@router.post("/chat/completions")
async def chat_completions(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
    runtime: GatewayRuntime = Depends(get_runtime),
):
    api_key = _require_api_key(authorization)
    try:
        auth = await runtime.authenticate(api_key)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model field is required")

    decision = await runtime.check_policy(auth.app, model)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    if await runtime.app_over_limit(auth.app):
        raise HTTPException(status_code=403, detail="Cost limit exceeded")

    stream = bool(payload.get("stream", False))
    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    result, record, backend = await runtime.execute_operation(
        operation="chat", model=model, payload=payload, stream=stream
    )

    if stream:
        if isinstance(result, BackendResult) and hasattr(result, "as_sse"):
            normalized_body = _normalize_chat_response(result.body, request_id, model, backend)
            source_result = BackendResult(body=normalized_body, usage=result.usage)
            collector: _StreamCollector = _StaticStreamCollector(normalized_body, result.usage)
            source_iter = source_result.as_sse(request_id)
        else:
            collector = _ChatStreamCollector(request_id, model, backend)
            source_iter = result

        async def iterator():
            try:
                async for chunk in source_iter:
                    collector.feed(chunk)
                    yield chunk
            finally:
                body, usage = collector.finalize()
                _apply_stream_usage(record, usage, runtime, backend, model)
                trace_payload = _attach_trace_payload(
                    request,
                    runtime,
                    app_id=auth.app.app_id,
                    api_key=api_key,
                    operation="chat",
                    model=model,
                    backend=backend,
                    request_payload=payload,
                    response_payload=body,
                    record=record,
                    stream=True,
                )
                if trace_payload is not None:
                    manager = runtime.trace_manager()
                    if manager is not None:
                        await manager.process(trace_payload)
                await runtime.record_usage(record, auth.app.app_id)

        return StreamingResponse(iterator(), media_type="text/event-stream")

    if not isinstance(result, BackendResult):
        raise HTTPException(status_code=500, detail="Unexpected backend response type")
    body = _normalize_chat_response(result.body, request_id, model, backend)
    await runtime.record_usage(record, auth.app.app_id)
    _attach_trace_payload(
        request,
        runtime,
        app_id=auth.app.app_id,
        api_key=api_key,
        operation="chat",
        model=model,
        backend=backend,
        request_payload=payload,
        response_payload=body,
        record=record,
    )
    return JSONResponse(body)


@router.post("/completions")
async def completions(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
    runtime: GatewayRuntime = Depends(get_runtime),
):
    api_key = _require_api_key(authorization)
    try:
        auth = await runtime.authenticate(api_key)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model field is required")

    decision = await runtime.check_policy(auth.app, model)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    if await runtime.app_over_limit(auth.app):
        raise HTTPException(status_code=403, detail="Cost limit exceeded")

    result, record, backend = await runtime.execute_operation(
        operation="completions", model=model, payload=payload
    )
    await runtime.record_usage(record, auth.app.app_id)

    body = _normalize_completion_response(result.body, model, backend)
    _attach_trace_payload(
        request,
        runtime,
        app_id=auth.app.app_id,
        api_key=api_key,
        operation="completions",
        model=model,
        backend=backend,
        request_payload=payload,
        response_payload=body,
        record=record,
    )
    return JSONResponse(body)


@router.post("/embeddings")
async def embeddings(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
    runtime: GatewayRuntime = Depends(get_runtime),
):
    api_key = _require_api_key(authorization)
    try:
        auth = await runtime.authenticate(api_key)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model field is required")

    decision = await runtime.check_policy(auth.app, model)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    if await runtime.app_over_limit(auth.app):
        raise HTTPException(status_code=403, detail="Cost limit exceeded")

    result, record, backend = await runtime.execute_operation(
        operation="embeddings", model=model, payload=payload
    )
    await runtime.record_usage(record, auth.app.app_id)
    body = _normalize_embeddings_response(result.body, model)
    _attach_trace_payload(
        request,
        runtime,
        app_id=auth.app.app_id,
        api_key=api_key,
        operation="embeddings",
        model=model,
        backend=backend,
        request_payload=payload,
        response_payload=body,
        record=record,
    )
    return JSONResponse(body)


@router.post("/images/generations")
async def images(
    request: Request,
    payload: Dict[str, Any] = Body(...),
    authorization: str | None = Header(default=None),
    runtime: GatewayRuntime = Depends(get_runtime),
):
    api_key = _require_api_key(authorization)
    try:
        auth = await runtime.authenticate(api_key)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    model = payload.get("model")
    if not model:
        raise HTTPException(status_code=400, detail="model field is required")

    decision = await runtime.check_policy(auth.app, model)
    if not decision.allowed:
        raise HTTPException(status_code=403, detail=decision.reason)

    if await runtime.app_over_limit(auth.app):
        raise HTTPException(status_code=403, detail="Cost limit exceeded")

    result, record, backend = await runtime.execute_operation(
        operation="images", model=model, payload=payload
    )
    await runtime.record_usage(record, auth.app.app_id)
    body = result.body
    image_entries: Optional[Sequence[Dict[str, Any]]] = None
    if isinstance(body, dict):
        data = body.get("data")
        if isinstance(data, Sequence):
            image_entries = [item for item in data if isinstance(item, dict)]
    _attach_trace_payload(
        request,
        runtime,
        app_id=auth.app.app_id,
        api_key=api_key,
        operation="images",
        model=model,
        backend=backend,
        request_payload=payload,
        response_payload=body,
        record=record,
        image_payloads=image_entries,
    )
    return JSONResponse(body)


def _attach_trace_payload(
    request: Request,
    runtime: GatewayRuntime,
    *,
    app_id: str,
    api_key: Optional[str],
    operation: str,
    model: str,
    backend: str,
    request_payload: Dict[str, Any],
    response_payload: Any,
    record: AccountingRecord,
    stream: bool = False,
    image_payloads: Optional[Sequence[Dict[str, Any]]] = None,
) -> Optional[TracePayload]:
    manager = runtime.trace_manager()
    if manager is None:
        return None
    if manager.config_for(app_id) is None:
        return None
    payload = TracePayload(
        app_id=app_id,
        operation=operation,
        model=model,
        backend=backend,
        request_payload=request_payload,
        response_payload=response_payload,
        record=record,
        api_key=api_key,
        stream=stream,
        image_payloads=image_payloads,
    )
    request.state.trace_payload = payload
    return payload


@router.get("/models")
async def models(
    authorization: str | None = Header(default=None),
    runtime: GatewayRuntime = Depends(get_runtime),
):
    """Return models available to the authenticated application."""
    api_key = _require_api_key(authorization)
    try:
        auth = await runtime.authenticate(api_key)
    except AuthError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    router = runtime.router()
    payload = router.build_models_payload()
    data = payload.get("data") if isinstance(payload, dict) else None
    entries = data if isinstance(data, list) else []

    entries_by_id: Dict[str, Dict[str, Any]] = {}
    models = []
    seen_ids: set[str] = set()

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        if not model_id:
            continue
        entries_by_id[model_id] = entry
        decision = await runtime.check_policy(auth.app, model_id)
        if decision.allowed:
            models.append(entry)
            seen_ids.add(model_id)

    for alias, _ in router.aliases().items():
        if alias in seen_ids:
            continue
        decision = await runtime.check_policy(auth.app, alias)
        if not decision.allowed:
            continue
        canonical = router.expand_alias(alias)
        template = entries_by_id.get(canonical)
        if template:
            alias_entry = dict(template)
        else:
            alias_entry = {"object": "model"}
        alias_entry["id"] = alias
        if template and "owned_by" in template:
            alias_entry.setdefault("owned_by", template["owned_by"])
        models.append(alias_entry)
        seen_ids.add(alias)

    result = dict(payload) if isinstance(payload, dict) else {"data": []}
    result["data"] = models
    return JSONResponse(result)


def _normalize_chat_response(body: Dict[str, Any], request_id: str, model: str, backend: str) -> Dict[str, Any]:
    """
        This normalization is needed to work with both ollama and OpenAI backends.
    """
    if "choices" in body:
        return body
    message = body.get("message") or body.get("messages", [{}])[-1]
    content = message.get("content") if isinstance(message, dict) else message
    content_str = _coerce_stream_text(content)
    assistant_message: Dict[str, Any]
    if isinstance(message, dict):
        assistant_message = {
            "role": message.get("role", "assistant"),
            "content": content_str,
        }
        tool_calls = message.get("tool_calls")
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
    else:
        assistant_message = {"role": "assistant", "content": content_str}

    return {
        "id": request_id,
        "object": "chat.completion",
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": assistant_message,
                "finish_reason": body.get("finish_reason", "stop"),
            }
        ],
        "provider": backend,
    }


class _StreamCollector:
    def feed(self, chunk: Any) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def finalize(self) -> tuple[Dict[str, Any], BackendUsage]:  # pragma: no cover - interface
        raise NotImplementedError


class _StaticStreamCollector(_StreamCollector):
    def __init__(self, body: Dict[str, Any], usage: BackendUsage):
        self._body = body
        self._usage = usage

    def feed(self, chunk: Any) -> None:
        return

    def finalize(self) -> tuple[Dict[str, Any], BackendUsage]:
        return self._body, self._usage


class _ChatStreamCollector(_StreamCollector):
    def __init__(self, request_id: str, model: str, backend: str):
        self._request_id = request_id
        self._model = model
        self._backend = backend
        self._content_parts: List[str] = []
        self._tool_calls: Dict[int, Dict[str, Any]] = {}
        self._finish_reason: Optional[str] = None
        self._role: str = "assistant"
        self._usage = BackendUsage()
        self._object = "chat.completion"
        self._model_override: Optional[str] = None

    def feed(self, chunk: Any) -> None:
        if chunk is None:
            return
        if isinstance(chunk, (bytes, bytearray)):
            data = bytes(chunk)
        elif isinstance(chunk, str):
            data = chunk.encode("utf-8", errors="ignore")
        elif isinstance(chunk, dict):
            data = json.dumps(chunk).encode("utf-8", errors="ignore")
        else:
            data = str(chunk).encode("utf-8", errors="ignore")
        for payload in _extract_sse_payloads(data):
            if payload is None:
                continue
            self._process_payload(payload)

    def _process_payload(self, payload: Dict[str, Any]) -> None:
        model_value = payload.get("model")
        if isinstance(model_value, str):
            self._model_override = model_value
        object_value = payload.get("object")
        if isinstance(object_value, str):
            self._object = object_value
        usage = payload.get("usage")
        if isinstance(usage, dict):
            self._usage.input_tokens = usage.get("prompt_tokens")
            self._usage.output_tokens = usage.get("completion_tokens")
            self._usage.total_tokens = usage.get("total_tokens")
        choices = payload.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta") or {}
            role = delta.get("role")
            if isinstance(role, str):
                self._role = role
            content = delta.get("content")
            if content is not None:
                text = _coerce_stream_text(content)
                if text:
                    self._content_parts.append(text)
            for entry in delta.get("tool_calls") or []:
                if isinstance(entry, dict):
                    _merge_tool_call_fragment(self._tool_calls, entry)
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                self._finish_reason = finish_reason
            message_obj = choice.get("message")
            if isinstance(message_obj, dict):
                msg_content = message_obj.get("content")
                if msg_content is not None:
                    text = _coerce_stream_text(msg_content)
                    if text:
                        self._content_parts.append(text)
                for entry in message_obj.get("tool_calls") or []:
                    if isinstance(entry, dict):
                        _merge_tool_call_fragment(self._tool_calls, entry)

    def finalize(self) -> tuple[Dict[str, Any], BackendUsage]:
        message: Dict[str, Any] = {
            "role": self._role,
            "content": "".join(self._content_parts),
        }
        if not message["content"]:
            message["content"] = ""
        if self._tool_calls:
            ordered = [self._tool_calls[index] for index in sorted(self._tool_calls)]
            message["tool_calls"] = ordered
        finish_reason = self._finish_reason
        if finish_reason is None:
            finish_reason = "tool_calls" if self._tool_calls else "stop"
        body = {
            "id": self._request_id,
            "object": self._object,
            "model": self._model_override or self._model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }
        normalized = _normalize_chat_response(body, self._request_id, self._model, self._backend)
        return normalized, self._usage


def _extract_sse_payloads(chunk: bytes) -> Iterable[Optional[Dict[str, Any]]]:
    text = chunk.decode("utf-8", errors="ignore")
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        data_lines: List[str] = []
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
        except json.JSONDecodeError:  # pragma: no cover - defensive
            continue


def _coerce_stream_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        fragments = [_coerce_stream_text(item) for item in content]
        return "".join(fragments)
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        value = content.get("value")
        if isinstance(value, str):
            return value
    return ""


def _merge_tool_call_fragment(storage: Dict[int, Dict[str, Any]], entry: Dict[str, Any]) -> None:
    index = entry.get("index")
    if not isinstance(index, int):
        index = max(storage.keys(), default=-1) + 1
    builder = storage.setdefault(
        index,
        {
            "id": entry.get("id"),
            "type": entry.get("type", "function"),
            "function": {"name": "", "arguments": ""},
        },
    )
    if entry.get("id"):
        builder["id"] = entry["id"]
    builder["type"] = entry.get("type", builder.get("type", "function"))
    fn = entry.get("function") or {}
    function_builder = builder.setdefault("function", {"name": "", "arguments": ""})
    if fn.get("name"):
        function_builder["name"] = fn["name"]
    if fn.get("arguments"):
        args_fragment = fn["arguments"]
        if not isinstance(args_fragment, str):
            args_fragment = json.dumps(args_fragment)
        existing = function_builder.get("arguments", "")
        function_builder["arguments"] = existing + args_fragment


def _apply_stream_usage(
    record: AccountingRecord,
    usage: BackendUsage,
    runtime: GatewayRuntime,
    backend_name: str,
    model: str,
) -> None:
    updated = False
    if usage.input_tokens is not None:
        record.prompt_tokens = usage.input_tokens
        updated = True
    if usage.output_tokens is not None:
        record.completion_tokens = usage.output_tokens
        updated = True
    if usage.total_tokens is not None:
        record.total_tokens = usage.total_tokens
        updated = True
    if not updated:
        return
    try:
        backend_client = runtime.backend_client(backend_name)
    except Exception:  # pragma: no cover - backend lookup should succeed
        return
    record.cost = runtime._estimate_cost(usage, backend_client.definition, model)


def _normalize_completion_response(body: Dict[str, Any], model: str, backend: str) -> Dict[str, Any]:
    """
        This normalization is needed to work with both ollama and OpenAI backends.
    """
    if "choices" in body:
        return body
    text = body.get("response") or body.get("text")
    return {
        "id": f"cmpl-{uuid.uuid4().hex}",
        "object": "text_completion",
        "model": model,
        "choices": [
            {"text": text, "index": 0, "finish_reason": body.get("finish_reason", "stop")}
        ],
        "provider": backend,
    }


def _normalize_embeddings_response(body: Dict[str, Any], model: str) -> Dict[str, Any]:
    """
        This normalization is needed to work with both ollama and OpenAI backends.
    """
    if "data" in body:
        return body
    embedding = body.get("embedding") or []
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": embedding, "index": 0}],
        "model": model,
    }


__all__ = ["router"]

"""
    Public OpenAI-compatible API layer. This does _not_ implement all 
    features of OpenAI but for the endpoints it implements it supports
    the same API.
"""
from __future__ import annotations

import uuid
from typing import Any, Dict, Optional, Sequence

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..accounting import AccountingRecord
from ..auth import AuthError
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
    request_id = f"chatcmpl-{uuid.uuid4().hex}" if stream else f"chatcmpl-{uuid.uuid4().hex}"
    result, record, backend = await runtime.execute_operation(
        operation="chat", model=model, payload=payload, stream=stream
    )
    await runtime.record_usage(record, auth.app.app_id)

    if stream:
        if hasattr(result, "as_sse"):

            async def iterator():
                async for chunk in result.as_sse(request_id):
                    yield chunk

            stream_response = StreamingResponse(iterator(), media_type="text/event-stream")
        else:
            stream_response = StreamingResponse(result, media_type="text/event-stream")

        _attach_trace_payload(
            request,
            runtime,
            app_id=auth.app.app_id,
            api_key=api_key,
            operation="chat",
            model=model,
            backend=backend,
            request_payload=payload,
            response_payload=None,
            record=record,
            stream=True,
        )
        return stream_response

    body = _normalize_chat_response(result.body, request_id, model, backend)
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
) -> None:
    manager = runtime.trace_manager()
    if manager is None:
        return
    if manager.config_for(app_id) is None:
        return
    request.state.trace_payload = TracePayload(
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


@router.get("/models")
async def models(runtime: GatewayRuntime = Depends(get_runtime)):
    """
        This currentlsy is somewhat hacked, TODO as soon as possible
    """
    payload = runtime.router().build_models_payload()
    return JSONResponse(payload)


def _normalize_chat_response(body: Dict[str, Any], request_id: str, model: str, backend: str) -> Dict[str, Any]:
    """
        This normalization is needed to work with both ollama and OpenAI backends.
    """
    if "choices" in body:
        return body
    message = body.get("message") or body.get("messages", [{}])[-1]
    content = message.get("content") if isinstance(message, dict) else message
    assistant_message: Dict[str, Any]
    if isinstance(message, dict):
        assistant_message = {
            "role": message.get("role", "assistant"),
            "content": content,
        }
        tool_calls = message.get("tool_calls")
        if tool_calls:
            assistant_message["tool_calls"] = tool_calls
    else:
        assistant_message = {"role": "assistant", "content": content}

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

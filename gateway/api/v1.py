"""Public OpenAI-compatible API layer."""
from __future__ import annotations

import uuid
from typing import Any, Dict

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..auth import AuthError
from ..runtime import GatewayRuntime

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
            return StreamingResponse(iterator(), media_type="text/event-stream")
        return StreamingResponse(result, media_type="text/event-stream")

    body = _normalize_chat_response(result.body, request_id, model, backend)
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
    return JSONResponse(body)


@router.get("/models")
async def models(runtime: GatewayRuntime = Depends(get_runtime)):
    payload = runtime.router().build_models_payload()
    return JSONResponse(payload)


def _normalize_chat_response(body: Dict[str, Any], request_id: str, model: str, backend: str) -> Dict[str, Any]:
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
    if "data" in body:
        return body
    embedding = body.get("embedding") or []
    return {
        "object": "list",
        "data": [{"object": "embedding", "embedding": embedding, "index": 0}],
        "model": model,
    }


__all__ = ["router"]

from __future__ import annotations

from typing import Any, Dict
from openai import OpenAI
from .base import BackendClient, BackendResult, BackendUsage

class OpenAIBackend(BackendClient):
    def __init__(self, definition):
        super().__init__(definition)
        if not definition.api_key:
            raise ValueError("OpenAI backend requires api_key")
        self._client = OpenAI(api_key=definition.api_key, base_url=definition.base_url)

    async def chat(self, model: str, payload: Dict[str, Any], stream: bool = False):
        request_payload = dict(payload)
        request_payload.pop("model", None)
        request_payload.pop("stream", None)

        def _call():
            return self._client.chat.completions.create(model=model, stream=stream, **request_payload)

        response = await self._asyncify(_call)
        if stream:
            # Since the OpenAI Python SDK returns a generator for stream, convert to single payload.
            chunks = [chunk.model_dump() for chunk in response]
            body = {"choices": chunks}
            return BackendResult(body=body)
        body = response.model_dump()
        usage = BackendUsage(
            input_tokens=body.get("usage", {}).get("prompt_tokens"),
            output_tokens=body.get("usage", {}).get("completion_tokens"),
            total_tokens=body.get("usage", {}).get("total_tokens"),
        )
        return BackendResult(body=body, usage=usage)

    async def completions(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        request_payload = dict(payload)
        request_payload.pop("model", None)
        response = await self._asyncify(lambda: self._client.completions.create(model=model, **request_payload))
        body = response.model_dump()
        usage = BackendUsage(
            input_tokens=body.get("usage", {}).get("prompt_tokens"),
            output_tokens=body.get("usage", {}).get("completion_tokens"),
            total_tokens=body.get("usage", {}).get("total_tokens"),
        )
        return BackendResult(body=body, usage=usage)

    async def embeddings(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        request_payload = dict(payload)
        request_payload.pop("model", None)
        response = await self._asyncify(lambda: self._client.embeddings.create(model=model, **request_payload))
        return BackendResult(body=response.model_dump())

    async def images(self, model: str, payload: Dict[str, Any]) -> BackendResult:
        request_payload = dict(payload)
        request_payload.pop("model", None)
        response = await self._asyncify(lambda: self._client.images.generate(model=model, **request_payload))
        return BackendResult(body=response.model_dump())

    async def models(self) -> Dict[str, Any]:
        response = await self._asyncify(lambda: self._client.models.list())
        return response.model_dump()


__all__ = ["OpenAIBackend"]

"""Anthropic backend client."""
from __future__ import annotations

import requests
from typing import Any, Dict

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
        response.raise_for_status()
        return response.json()

    async def chat(self, model: str, payload: Dict[str, Any], stream: bool = False):
        messages = payload.get("messages")
        if messages is None:
            raise ValueError("Anthropic chat payload requires 'messages'")
        body = await self._asyncify(
            self._post,
            "/v1/messages",
            {
                "model": model,
                "messages": messages,
                "max_tokens": payload.get("max_tokens", 1024),
                "stream": False,
            },
        )
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

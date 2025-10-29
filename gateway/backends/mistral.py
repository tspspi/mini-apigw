from __future__ import annotations

from typing import Any, Dict

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

        def _call() -> Dict[str, Any]:
            try:
                if stream:
                    chunks = self._client.chat.stream(model=model, **request_payload)
                    serialized = [self._serialize(chunk) for chunk in chunks]
                    return {"choices": serialized}
                response = self._client.chat.complete(model=model, **request_payload)
                return self._serialize(response)
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

    def __del__(self):  # pragma: no cover - destructor semantics vary
        try:
            self._http_client.close()
        except Exception:
            pass


__all__ = ["MistralBackend"]

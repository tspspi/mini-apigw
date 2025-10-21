from __future__ import annotations

from typing import Dict, Type

from ..config import BackendDefinition
from .anthropic import AnthropicBackend
from .base import BackendClient
from .ollama import OllamaBackend
from .openai import OpenAIBackend


_BACKEND_TYPES: Dict[str, Type[BackendClient]] = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
}


def create_backend(definition: BackendDefinition) -> BackendClient:
    try:
        cls = _BACKEND_TYPES[definition.type]
    except KeyError as exc:
        raise ValueError(f"Unknown backend type: {definition.type}") from exc
    return cls(definition)


__all__ = ["create_backend", "BackendClient"]

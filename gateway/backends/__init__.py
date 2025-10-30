from __future__ import annotations

from dataclasses import replace
from typing import Dict, Type

from ..config import BackendDefinition
from .anthropic import AnthropicBackend
from .base import BackendClient
from .ollama import OllamaBackend
from .mistral import MistralBackend
from .openai import OpenAIBackend


_BACKEND_TYPES: Dict[str, Type[BackendClient]] = {
    "ollama": OllamaBackend,
    "openai": OpenAIBackend,
    "anthropic": AnthropicBackend,
    "mistral": MistralBackend,
}

_ALIASED_BACKENDS: Dict[str, tuple[Type[BackendClient], str]] = {
    # X.AI (Grok) exposes an OpenAI-compatible API with a fixed base URL.
    "xai": (OpenAIBackend, "https://api.x.ai/v1"),
    "grok": (OpenAIBackend, "https://api.x.ai/v1"),
    "google": (OpenAIBackend, "https://generativelanguage.googleapis.com/v1beta/openai/"),
}


def create_backend(definition: BackendDefinition) -> BackendClient:
    backend_type = definition.type.lower()

    if backend_type in _ALIASED_BACKENDS:
        cls, base_url = _ALIASED_BACKENDS[backend_type]
        # Override the base URL while keeping the original configuration intact.
        effective_definition = replace(definition, base_url=base_url)
        return cls(effective_definition)

    try:
        cls = _BACKEND_TYPES[backend_type]
    except KeyError as exc:
        raise ValueError(f"Unknown backend type: {definition.type}") from exc
    return cls(definition)


__all__ = ["create_backend", "BackendClient"]

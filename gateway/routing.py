"""Model routing and backend resolution."""
from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional

from .config import BackendConfig, BackendDefinition


class RoutingError(RuntimeError):
    pass


@dataclass(slots=True)
class CandidateBackend:
    backend: BackendDefinition
    resolved_model: str
    backend_model: str


@dataclass(slots=True)
class AutoModelEntry:
    chat: List[str] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    embeddings: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)


_OPERATION_ATTR = {
    "chat": "chat",
    "completions": "completions",
    "embeddings": "embeddings",
    "images": "images",
}


class ModelRouter:
    """Resolves models to eligible backends."""

    def __init__(self, config: BackendConfig):
        self._aliases: Dict[str, str] = {}
        self._backends: Dict[str, BackendDefinition] = {}
        self._auto_models: Dict[str, AutoModelEntry] = {}
        self.refresh(config)

    def refresh(self, config: BackendConfig) -> None:
        self._aliases = dict(config.aliases)
        self._backends = {backend.name: backend for backend in config.backends}
        for backend in config.backends:
            if backend.auto_models and backend.name not in self._auto_models:
                self._auto_models[backend.name] = AutoModelEntry()

    def aliases(self) -> Dict[str, str]:
        """Return a copy of known model aliases."""
        return dict(self._aliases)

    def update_auto_models(
        self,
        backend_name: str,
        *,
        chat: Optional[Iterable[str]] = None,
        completions: Optional[Iterable[str]] = None,
        embeddings: Optional[Iterable[str]] = None,
        images: Optional[Iterable[str]] = None,
    ) -> None:
        entry = self._auto_models.setdefault(backend_name, AutoModelEntry())
        if chat is not None:
            entry.chat = list(chat)
        if completions is not None:
            entry.completions = list(completions)
        if embeddings is not None:
            entry.embeddings = list(embeddings)
        if images is not None:
            entry.images = list(images)

    def expand_alias(self, model: str) -> str:
        seen: set[str] = set()
        current = model
        while current in self._aliases:
            if current in seen:
                raise RoutingError(f"Alias cycle detected at '{current}'")
            seen.add(current)
            current = self._aliases[current]
        return current

    def candidates(self, model: str, operation: str) -> List[CandidateBackend]:
        if operation not in _OPERATION_ATTR:
            raise RoutingError(f"Unknown operation '{operation}'")
        model_name = self.expand_alias(model)
        attr = _OPERATION_ATTR[operation]
        matches: List[CandidateBackend] = []
        for backend in self._backends.values():
            if self._supports(backend, model_name, operation, attr):
                backend_model = self._normalize_model_for_backend(backend, model_name)
                matches.append(
                    CandidateBackend(
                        backend=backend,
                        resolved_model=model_name,
                        backend_model=backend_model,
                    )
                )
        if not matches:
            raise RoutingError(f"No backend available for model '{model}' ({operation})")
        return matches

    def build_models_payload(self) -> Dict[str, List[Dict[str, str]]]:
        payload: List[Dict[str, str]] = []
        for backend in self._backends.values():
            for model in backend.supports.chat:
                payload.append({"id": model, "object": "model", "owned_by": backend.name})
            entry = self._auto_models.get(backend.name)
            if entry:
                for model in entry.chat:
                    payload.append({"id": model, "object": "model", "owned_by": backend.name})
        return {"data": payload}

    def _supports(self, backend: BackendDefinition, model: str, operation: str, attr: str) -> bool:
        patterns = getattr(backend.supports, attr)
        if self._match_patterns(model, patterns):
            return True
        entry = self._auto_models.get(backend.name)
        if not entry:
            return False
        auto_patterns = getattr(entry, attr)
        return self._match_patterns(model, auto_patterns)

    @staticmethod
    def _match_patterns(model: str, patterns: Iterable[str]) -> bool:
        for pattern in patterns:
            if pattern == model:
                return True
            if pattern in {"*", "any"}:
                return True
            if fnmatch.fnmatch(model, pattern):
                return True
        return False

    @staticmethod
    def _normalize_model_for_backend(backend: BackendDefinition, model: str) -> str:
        if ":" not in model:
            return model
        prefix, remainder = model.split(":", 1)
        normalized_prefixes = {backend.type, backend.name}
        if prefix in normalized_prefixes:
            return remainder
        return model


__all__ = ["ModelRouter", "RoutingError", "CandidateBackend"]

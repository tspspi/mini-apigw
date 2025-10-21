"""Authentication and app lookup."""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Dict, Optional

from .config import AppDefinition, AppsConfig


@dataclass(slots=True)
class AuthResult:
    app: AppDefinition


class AuthError(RuntimeError):
    pass


class AuthManager:
    """Resolves API keys to app definitions."""

    def __init__(self, apps_config: AppsConfig):
        self._index = self._build_index(apps_config)

    @staticmethod
    def _build_index(config: AppsConfig) -> Dict[str, AppDefinition]:
        return config.api_key_index()

    def refresh(self, apps_config: AppsConfig) -> None:
        self._index = self._build_index(apps_config)

    def authenticate(self, api_key: str) -> AuthResult:
        try:
            app = self._index[api_key]
        except KeyError as exc:
            raise AuthError("Unknown API key") from exc
        return AuthResult(app=app)


def mask_api_key(api_key: str) -> str:
    """Mask an API key for safe logging."""
    if len(api_key) <= 8:
        return "***"
    prefix = api_key[:4]
    suffix = api_key[-4:]
    return f"{prefix}…{suffix}"


def constant_time_compare(a: str, b: str) -> bool:
    return hmac.compare_digest(a, b)


__all__ = ["AuthError", "AuthManager", "AuthResult", "mask_api_key", "constant_time_compare"]

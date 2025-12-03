"""Trace logging utilities."""
from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .accounting import AccountingRecord
from .auth import mask_api_key
from .config import AppTraceConfig, AppsConfig

log = logging.getLogger(__name__)


@dataclass(slots=True)
class TracePayload:
    """Holds request/response metadata for trace logging."""

    app_id: str
    operation: str
    model: str
    backend: str
    request_payload: Dict[str, Any]
    response_payload: Any
    record: AccountingRecord
    api_key: Optional[str] = None
    stream: bool = False
    image_payloads: Optional[Sequence[Dict[str, Any]]] = None


class TraceManager:
    """Dispatches trace events to JSONL files and optional image stores."""

    def __init__(self, apps_config: AppsConfig):
        self._configs = self._build_configs(apps_config)
        self._locks: Dict[Path, asyncio.Lock] = {}

    @staticmethod
    def _build_configs(apps_config: AppsConfig) -> Dict[str, AppTraceConfig]:
        configs: Dict[str, AppTraceConfig] = {}
        for app in apps_config.apps:
            cfg = app.trace
            if cfg is None:
                continue
            if not cfg.file and not cfg.image_dir:
                continue
            configs[app.app_id] = cfg
        return configs

    def config_for(self, app_id: str) -> Optional[AppTraceConfig]:
        return self._configs.get(app_id)

    async def process(self, payload: TracePayload) -> None:
        config = self.config_for(payload.app_id)
        if config is None:
            return

        timestamp = payload.record.created_at.timestamp()
        event: Dict[str, Any] = {
            "timestamp": timestamp,
            "appid": payload.app_id,
            "type": payload.operation,
            "model": payload.model,
            "backend": payload.backend,
            "total_cost": payload.record.cost,
            "tokens": {
                "prompt": payload.record.prompt_tokens,
                "completion": payload.record.completion_tokens,
                "total": payload.record.total_tokens,
            },
            "latency_ms": payload.record.latency_ms,
        }

        prompt_value = self._extract_prompt(payload.operation, payload.request_payload)
        if config.include_prompts and prompt_value is not None:
            event["prompt"] = prompt_value

        if config.include_response and payload.response_payload is not None:
            event["response"] = payload.response_payload

        if config.include_keys and payload.api_key:
            event["api_key"] = mask_api_key(payload.api_key)

        if payload.stream:
            event["stream"] = True

        image_entries = payload.image_payloads or []
        if config.image_dir and payload.operation == "images" and image_entries:
            image_paths = await self._store_images(Path(config.image_dir), image_entries, timestamp)
            if image_paths:
                event["images"] = image_paths

        if config.file:
            await self._write_event(Path(config.file), event)

    async def _write_event(self, path: Path, event: Dict[str, Any]) -> None:
        lock = self._locks.setdefault(path, asyncio.Lock())
        loop = asyncio.get_running_loop()
        data = json.dumps(event, ensure_ascii=False, default=self._json_default)

        async with lock:
            await loop.run_in_executor(None, self._append_line, path, data)

    @staticmethod
    def _append_line(path: Path, data: str) -> None:
        path = path.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(data)
            handle.write("\n")

    async def _store_images(
        self, directory: Path, entries: Sequence[Dict[str, Any]], timestamp: float
    ) -> List[Dict[str, Any]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._write_images, directory, entries, timestamp)

    @staticmethod
    def _write_images(
        directory: Path, entries: Sequence[Dict[str, Any]], timestamp: float
    ) -> List[Dict[str, Any]]:
        directory = directory.expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        saved: List[Dict[str, Any]] = []
        for index, entry in enumerate(entries):
            data_b64 = entry.get("b64_json") or entry.get("b64")
            if isinstance(data_b64, str) and data_b64:
                try:
                    raw = base64.b64decode(data_b64, validate=True)
                except Exception:  # pragma: no cover - depends on backend payloads
                    log.warning("Failed to decode base64 image for trace logging", exc_info=True)
                    continue
                suffix = TraceManager._detect_extension(raw)
                filename = TraceManager._image_filename(timestamp, index, suffix)
                path = directory / filename
                path.write_bytes(raw)
                saved.append({"path": str(path), "source": "b64"})
                continue
            url = entry.get("url")
            if isinstance(url, str) and url:
                filename = TraceManager._image_filename(timestamp, index, "url")
                path = directory / f"{filename}.txt"
                path.write_text(url, encoding="utf-8")
                saved.append({"path": str(path), "source": "url"})
        return saved

    @staticmethod
    def _image_filename(timestamp: float, index: int, suffix: str) -> str:
        prefix = int(timestamp)
        token = uuid.uuid4().hex[:8]
        cleaned_suffix = suffix.lstrip(".")
        return f"{prefix}_{token}_{index}.{cleaned_suffix}" if cleaned_suffix else f"{prefix}_{token}_{index}"

    @staticmethod
    def _detect_extension(raw: bytes) -> str:
        if raw.startswith(b"\x89PNG\r\n\x1a\n"):
            return "png"
        if raw[:3] == b"\xff\xd8\xff":
            return "jpg"
        if raw[:2] == b"BM":
            return "bmp"
        return "bin"

    @staticmethod
    def _extract_prompt(operation: str, payload: Dict[str, Any]) -> Any:
        if operation == "chat":
            return payload.get("messages")
        if operation == "completions":
            return payload.get("prompt")
        if operation == "embeddings":
            return payload.get("input")
        if operation == "images":
            return payload.get("prompt") or payload.get("input")
        return None

    @staticmethod
    def _json_default(value: Any) -> Any:
        return repr(value)


__all__ = ["TraceManager", "TracePayload"]

"""Tests for trace logging helpers."""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from gateway.accounting import AccountingRecord
from gateway.config import AppDefinition, AppPolicy, AppTraceConfig, AppsConfig
from gateway.trace import TraceManager, TracePayload


@pytest.mark.asyncio
async def test_trace_manager_writes_jsonl_and_images(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    image_dir = tmp_path / "images"
    trace_cfg = AppTraceConfig(
        file=str(trace_file),
        image_dir=str(image_dir),
        include_prompts=True,
        include_response=True,
        include_keys=True,
    )
    app = AppDefinition(app_id="demo", api_keys=["sk-demo"], policy=AppPolicy(), trace=trace_cfg)
    manager = TraceManager(AppsConfig(apps=[app]))

    record = AccountingRecord(
        app_id="demo",
        backend="backend-a",
        model="model-a",
        operation="chat",
        cost=1.5,
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        latency_ms=42,
    )

    payload = TracePayload(
        app_id="demo",
        operation="chat",
        model="model-a",
        backend="backend-a",
        request_payload={"model": "model-a", "messages": [{"role": "user", "content": "hi"}]},
        response_payload={"id": "resp", "choices": []},
        record=record,
        api_key="sk-demo",
    )
    await manager.process(payload)

    assert trace_file.exists()
    contents = trace_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(contents) == 1
    entry = json.loads(contents[0])
    assert entry["appid"] == "demo"
    assert entry["backend"] == "backend-a"
    assert entry["tokens"]["total"] == 15
    assert entry["prompt"][0]["content"] == "hi"
    assert entry["api_key"] != "sk-demo"

    image_record = AccountingRecord(
        app_id="demo",
        backend="backend-a",
        model="model-a",
        operation="images",
        cost=0.0,
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
        latency_ms=10,
    )
    image_payload = TracePayload(
        app_id="demo",
        operation="images",
        model="model-a",
        backend="backend-a",
        request_payload={"model": "model-a", "prompt": "draw"},
        response_payload={"data": [{"b64_json": base64.b64encode(b"image-bytes").decode()}]},
        record=image_record,
        api_key="sk-demo",
        image_payloads=[{"b64_json": base64.b64encode(b"image-bytes").decode()}],
    )
    await manager.process(image_payload)

    saved_images = list(image_dir.glob("*"))
    assert saved_images, "expected image files to be written"

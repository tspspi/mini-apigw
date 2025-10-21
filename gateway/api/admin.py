"""Admin and stats endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ..ipacl import is_allowed
from ..runtime import GatewayRuntime

router = APIRouter()


async def get_runtime(request: Request) -> GatewayRuntime:
    runtime = getattr(request.app.state, "runtime", None)
    if runtime is None:
        raise HTTPException(status_code=503, detail="Gateway not ready")
    return runtime


def _require_local_access(request: Request, runtime: GatewayRuntime) -> None:
    client = request.client
    if client is None:
        raise HTTPException(status_code=403, detail="Forbidden")
    host = client.host
    if host in {"127.0.0.1", "::1"}:
        return
    allowlist = runtime.config_bundle.daemon.admin.stats_networks
    if not allowlist:
        raise HTTPException(status_code=403, detail="Forbidden")
    if not is_allowed(host, allowlist):
        raise HTTPException(status_code=403, detail="Forbidden")


@router.get("/stats/live")
async def stats_live(request: Request, runtime: GatewayRuntime = Depends(get_runtime)):
    _require_local_access(request, runtime)
    scheduler_stats = runtime.scheduler_stats()
    accounting = await runtime.accounting_snapshot()
    payload = {
        "per_backend": scheduler_stats.get("per_backend", {}),
        "per_group": scheduler_stats.get("per_group", {}),
        "per_app": accounting,
    }
    return JSONResponse(payload)


@router.get("/stats/usage")
async def stats_usage(
    request: Request,
    runtime: GatewayRuntime = Depends(get_runtime),
    app_id: str | None = Query(default=None),
    since: str | None = Query(default=None),
):
    _require_local_access(request, runtime)
    accounting = await runtime.accounting_snapshot()
    if app_id is not None:
        accounting = {app_id: accounting.get(app_id, {"total_cost": 0.0, "request_count": 0})}
    payload = {
        "since": since or datetime.utcnow().isoformat() + "Z",
        "apps": accounting,
    }
    return JSONResponse(payload)


__all__ = ["router"]

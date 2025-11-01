"""
    Admin and stats endpoints.

    Those endpoints are usually exposed only to localhost (admin). The statistics
    endpoints can be exposed to any host by editing in the configuration file. The
    admin endpoints allow termination of the daemon as well as the reloading of the
    configuration (as SIGHUP and SIGTERM). The statistic endpoints expose current day
    and live statistics of the router service
"""
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
    """
        Require local access enforces that our endpoint is one of the loopback
        adresses or the unix domain socket.
    """
    client = request.client
    listen_cfg = runtime.config_bundle.daemon.listen

    if client is None:
        if listen_cfg.unix_socket:
            return
        raise HTTPException(status_code=403, detail="Forbidden")

    host = client.host
    if host in {"127.0.0.1", "::1"}:
        return

    allowlist = runtime.config_bundle.daemon.admin.stats_networks
    if not allowlist:
        if listen_cfg.unix_socket:
            # When running behind a reverse proxy on a unix domain socket we
            # trust the proxy to enforce access control.
            return
        raise HTTPException(status_code=403, detail="Forbidden")
    if not is_allowed(host, allowlist):
        raise HTTPException(status_code=403, detail="Forbidden")



def _aggregate_usage_per_model(accounting: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    aggregated: Dict[str, Dict[str, int]] = {}
    for app_data in accounting.values():
        for model_entry in app_data.get("models", []):
            model_name = model_entry.get("model")
            if not model_name:
                continue
            metrics = aggregated.setdefault(
                model_name,
                {"request_count": 0, "total_tokens": 0, "latency_ms": 0},
            )
            metrics["request_count"] += int(model_entry.get("request_count") or 0)
            total_tokens = model_entry.get("total_tokens")
            if total_tokens is None:
                total_tokens = (model_entry.get("prompt_tokens") or 0) + (model_entry.get("completion_tokens") or 0)
            metrics["total_tokens"] += int(total_tokens or 0)
            metrics["latency_ms"] += int(model_entry.get("latency_ms") or 0)
    return aggregated


@router.get("/stats/live")
async def stats_live(request: Request, runtime: GatewayRuntime = Depends(get_runtime)):
    """
        Live statistics (accessible from local host or allowlist) provide really _current_
        in flight or in-queue requests.
    """
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
    """
        The usage endpoint provides JSON data about the current daily usage (from current day).
        It resets every day and gets populated on server restart.
    """
    _require_local_access(request, runtime)
    accounting = await runtime.accounting_snapshot()
    if app_id is not None:
        accounting = {app_id: accounting.get(app_id, {"total_cost": 0.0, "request_count": 0})}
    payload = {
        "since": since or datetime.utcnow().isoformat() + "Z",
        "apps": accounting,
    }
    return JSONResponse(payload)


@router.get("/stat/usagepermodel/requests")
async def stats_usage_per_model_requests(
    request: Request,
    runtime: GatewayRuntime = Depends(get_runtime),
):
    """
        Aggregate request counts per model for the current day.
    """
    _require_local_access(request, runtime)
    accounting = await runtime.accounting_snapshot()
    aggregated = _aggregate_usage_per_model(accounting)
    payload = [
        {"metric": model_name, "value": metrics["request_count"]}
        for model_name, metrics in sorted(aggregated.items())
    ]
    return JSONResponse(payload)


@router.get("/stat/usagepermodel/tokens")
async def stats_usage_per_model_tokens(
    request: Request,
    runtime: GatewayRuntime = Depends(get_runtime),
):
    """
        Aggregate token usage per model for the current day.
    """
    _require_local_access(request, runtime)
    accounting = await runtime.accounting_snapshot()
    aggregated = _aggregate_usage_per_model(accounting)
    payload = [
        {"metric": model_name, "value": metrics["total_tokens"]}
        for model_name, metrics in sorted(aggregated.items())
    ]
    return JSONResponse(payload)


@router.get("/stat/usagepermodel/time")
async def stats_usage_per_model_time(
    request: Request,
    runtime: GatewayRuntime = Depends(get_runtime),
):
    """
        Aggregate total processing time per model for the current day.
    """
    _require_local_access(request, runtime)
    accounting = await runtime.accounting_snapshot()
    aggregated = _aggregate_usage_per_model(accounting)
    payload = [
        {"metric": model_name, "value": metrics["latency_ms"] / 1000.0}
        for model_name, metrics in sorted(aggregated.items())
    ]
    return JSONResponse(payload)


@router.post("/admin/reload")
async def admin_reload(request: Request, runtime: GatewayRuntime = Depends(get_runtime)):
    """
        Reloads the server - this re-reads configuration files (!)
        This is equal to SIGHUP.
    """
    _require_local_access(request, runtime)
    await runtime.reload()
    return JSONResponse({"status": "reloaded"})


@router.post("/admin/shutdown")
async def admin_shutdown(request: Request, runtime: GatewayRuntime = Depends(get_runtime)):
    """
        Terminate our server like SIGTERM
    """
    _require_local_access(request, runtime)
    server = getattr(request.app.state, "server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="Shutdown controller unavailable")
    if hasattr(server, "should_exit"):
        server.should_exit = True
    return JSONResponse({"status": "shutting_down"})


__all__ = ["router"]

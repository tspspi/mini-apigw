"""ASGI application factory."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI

from .api import admin as admin_router
from .api import v1 as v1_router
from .config import ConfigError
from .log import configure_logging
from .middleware.trace import TraceMiddleware
from .runtime import GatewayRuntime
from .signals import install_signal_handlers


def _resolve_config_dir(config_dir: str | os.PathLike[str] | None) -> Path:
    if config_dir is not None:
        return Path(config_dir)
    env_dir = os.environ.get("MINIAPIGW_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.cwd() / "config"


def create_app(config_dir: str | os.PathLike[str] | None = None) -> FastAPI:
    cfg_dir = _resolve_config_dir(config_dir)
    runtime = GatewayRuntime(cfg_dir)

    app = FastAPI(
        title="mini-apigw",
        version="0.1.0",
        docs_url=None,
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    # Register all routers
    app.add_middleware(TraceMiddleware)
    app.include_router(v1_router.router)
    app.include_router(admin_router.router)

    app.state.runtime = runtime

    @app.on_event("startup")
    async def on_startup() -> None:
        await runtime.initialize()
        configure_logging(runtime.config_bundle.daemon.logging)
        install_signal_handlers(runtime, runtime.config_bundle.daemon.reload.enable_sighup)

    @app.on_event("shutdown")
    async def on_shutdown() -> None:
        await runtime.shutdown()

    return app


__all__ = ["create_app"]

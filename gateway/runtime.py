"""Runtime wiring for the gateway."""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from time import perf_counter
from typing import Any, Dict, Optional

from .accounting import AccountingRecord, AccountingRecorder
from .auth import AuthError, AuthManager
from .backends import BackendClient, create_backend
from .backends.base import BackendResult, BackendUsage
from .config import ConfigBundle, ConfigError, ConfigManager
from .policies import PolicyDecision, PolicyEngine
from .routing import ModelRouter, RoutingError
from .scheduling import Scheduler
from .trace import TraceManager

log = logging.getLogger(__name__)


class GatewayRuntime:
    def __init__(self, config_dir: Path):
        self._config_dir = config_dir
        self._config_manager = ConfigManager(
            config_dir / "daemon.json",
            config_dir / "backends.json",
            config_dir / "apps.json",
        )
        self._bundle: Optional[ConfigBundle] = None
        self._backends: Dict[str, BackendClient] = {}
        self._router: Optional[ModelRouter] = None
        self._scheduler: Optional[Scheduler] = None
        self._auth_manager: Optional[AuthManager] = None
        self._policy_engine = PolicyEngine()
        self._accounting: Optional[AccountingRecorder] = None
        self._trace_manager: Optional[TraceManager] = None
        self._lock = asyncio.Lock()

    @property
    def config_bundle(self) -> ConfigBundle:
        if self._bundle is None:
            raise ConfigError("Configuration not loaded")
        return self._bundle

    async def initialize(self) -> None:
        async with self._lock:
            bundle = self._config_manager.load()
            await self._apply_bundle(bundle)

    async def reload(self) -> None:
        async with self._lock:
            bundle = self._config_manager.load()
            log.info("Configuration reload requested")
            await self._shutdown_components()
            await self._apply_bundle(bundle)

    async def shutdown(self) -> None:
        async with self._lock:
            await self._shutdown_components()

    async def _apply_bundle(self, bundle: ConfigBundle) -> None:
        backends: Dict[str, BackendClient] = {}
        for definition in bundle.backends.backends:
            backends[definition.name] = create_backend(definition)
        self._backends = backends
        self._router = ModelRouter(bundle.backends)
        self._scheduler = Scheduler(bundle.backends, backends)
        self._auth_manager = AuthManager(bundle.apps)
        self._trace_manager = TraceManager(bundle.apps)
        self._bundle = bundle
        self._accounting = AccountingRecorder(bundle.daemon.database)
        await self._accounting.start()
        log.info("Gateway runtime initialized with %d backends", len(backends))

    async def _shutdown_components(self) -> None:
        if self._accounting is not None:
            await self._accounting.stop()
            self._accounting = None
        self._trace_manager = None
        self._backends = {}
        self._router = None
        self._scheduler = None
        self._auth_manager = None
        self._bundle = None

    async def authenticate(self, api_key: str):
        if self._auth_manager is None:
            raise AuthError("Auth manager not ready")
        return self._auth_manager.authenticate(api_key)

    async def check_policy(self, app, model: str) -> PolicyDecision:
        return self._policy_engine.is_allowed(app, model)

    def router(self) -> ModelRouter:
        if self._router is None:
            raise RoutingError("Router not initialized")
        return self._router

    def scheduler(self) -> Scheduler:
        if self._scheduler is None:
            raise RuntimeError("Scheduler not initialized")
        return self._scheduler

    def trace_manager(self) -> Optional[TraceManager]:
        return self._trace_manager

    def backend_client(self, backend_name: str) -> BackendClient:
        try:
            return self._backends[backend_name]
        except KeyError as exc:
            raise RuntimeError(f"Unknown backend '{backend_name}'") from exc

    @staticmethod
    def _estimate_cost(usage: BackendUsage, backend_definition, model: str) -> float:
        prompt_tokens = usage.input_tokens or 0
        completion_tokens = usage.output_tokens or 0
        total = usage.total_tokens or (prompt_tokens + completion_tokens)
        rate = backend_definition.cost.rate_for(model)
        prompt_rate = float(rate.prompt or 0.0)
        completion_rate = float(rate.completion or 0.0)
        cost = 0.0
        if prompt_tokens:
            cost += (prompt_tokens / 1000.0) * prompt_rate
        if completion_tokens:
            cost += (completion_tokens / 1000.0) * completion_rate
        if total and cost == 0.0:
            cost += (total / 1000.0) * prompt_rate
        return cost

    async def execute_operation(
        self,
        *,
        operation: str,
        model: str,
        payload: Dict[str, Any],
        stream: bool = False,
    ) -> tuple[BackendResult | Any, AccountingRecord, str]:
        router = self.router()
        scheduler = self.scheduler()
        candidates = router.candidates(model, operation)
        # Naive selection: choose the first candidate backend.
        selected = candidates[0]
        backend_def = selected.backend
        client = self.backend_client(backend_def.name)
        resolved_model = selected.resolved_model
        backend_model = selected.backend_model

        start = perf_counter()
        async with await scheduler.acquire(backend_def.name):
            if operation == "chat":
                result = await client.chat(backend_model, payload, stream=stream)
            elif operation == "completions":
                result = await client.completions(backend_model, payload)
            elif operation == "embeddings":
                result = await client.embeddings(backend_model, payload)
            elif operation == "images":
                result = await client.images(backend_model, payload)
            else:
                raise RuntimeError(f"Unsupported operation '{operation}'")
        latency_ms = int((perf_counter() - start) * 1000)

        if isinstance(result, BackendResult):
            usage = result.usage
        else:
            usage = BackendUsage()

        cost = self._estimate_cost(usage, backend_def, resolved_model)
        record = AccountingRecord(
            app_id="<unassigned>",
            backend=backend_def.name,
            model=resolved_model,
            operation=operation,
            cost=cost,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=latency_ms,
        )
        return result, record, backend_def.name

    async def record_usage(self, record: AccountingRecord, app_id: str) -> None:
        if self._accounting is None:
            return
        record.app_id = app_id
        await self._accounting.record(record)

    async def app_over_limit(self, app) -> bool:
        if self._accounting is None:
            return False
        return await self._accounting.over_cost_limit(app)

    async def accounting_snapshot(self) -> Dict[str, Dict[str, Any]]:
        if self._accounting is None:
            return {}
        snapshot = await self._accounting.snapshot()
        result = {}
        for app_id, state in snapshot.items():
            models = [
                {
                    "backend": model_state.backend,
                    "model": model_state.model,
                    "cost": model_state.total_cost,
                    "request_count": model_state.request_count,
                    "prompt_tokens": model_state.total_prompt_tokens,
                    "completion_tokens": model_state.total_completion_tokens,
                    "total_tokens": model_state.total_tokens,
                    "latency_ms": model_state.total_latency_ms,
                }
                for model_state in state.models.values()
            ]
            models.sort(key=lambda entry: (entry["backend"], entry["model"]))
            result[app_id] = {
                "total_cost": state.total_cost,
                "request_count": state.request_count,
                "prompt_tokens": state.total_prompt_tokens,
                "completion_tokens": state.total_completion_tokens,
                "total_tokens": state.total_tokens,
                "latency_ms": state.total_latency_ms,
                "models": models,
            }
        return result

    def scheduler_stats(self) -> Dict[str, Any]:
        if self._scheduler is None:
            return {}
        return self._scheduler.stats()


__all__ = ["GatewayRuntime"]

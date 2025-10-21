"""Usage accounting and PostgreSQL backend"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Sequence

from .config import AppDefinition, CostLimitConfig, DatabaseConfig

log = logging.getLogger(__name__)

try:
    import psycopg  # type: ignore[attr-defined]
    _DB_DRIVER_NAME = "psycopg"
except ImportError:  # pragma: no cover - psycopg optional
    try:
        import psycopg2 as psycopg  # type: ignore[assignment]
        _DB_DRIVER_NAME = "psycopg2"
    except ImportError:  # pragma: no cover - psycopg optional
        psycopg = None  # type: ignore[assignment]
        _DB_DRIVER_NAME = None

@dataclass(slots=True)
class AccountingRecord:
    app_id: str
    backend: str
    model: str
    operation: str
    cost: float
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    total_tokens: Optional[int]
    latency_ms: Optional[int]
    created_at: datetime = datetime.now(timezone.utc)

@dataclass(slots=True)
class ModelCostState:
    backend: str
    model: str
    total_cost: float = 0.0
    request_count: int = 0

@dataclass(slots=True)
class CostState:
    total_cost: float = 0.0
    request_count: int = 0
    models: Dict[tuple[str, str], ModelCostState] = field(default_factory=dict)


class AccountingRecorder:
    """Persist usage records and expose aggregates."""

    def __init__(self, db_config: Optional[DatabaseConfig]):
        self._db_config = db_config
        self._queue: "asyncio.Queue[AccountingRecord | None]" = asyncio.Queue()
        self._task: Optional[asyncio.Task[None]] = None
        self._costs: Dict[str, CostState] = {}
        self._cost_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self._db_conn = None
        self._db_driver_name = _DB_DRIVER_NAME
        if db_config is None:
            log.warning("Accounting recorder running without database; usage persisted in-memory only")
        elif psycopg is None:
            log.warning("Accounting recorder has database configuration but no Postgres driver (psycopg/psycopg2) is available")
        else:
            log.info("Accounting recorder using %s driver for Postgres persistence", self._db_driver_name)

    async def start(self) -> None:
        if self._task is None:
            await self._bootstrap_costs()
            self._task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        if self._task is None:
            return
        await self._queue.put(None)
        try:
            await self._task
        except asyncio.CancelledError:  # pragma: no cover - shutdown cancellation
            pass
        self._task = None
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception:  # pragma: no cover
                pass
            self._db_conn = None

    async def record(self, record: AccountingRecord) -> None:
        async with self._cost_lock:
            state = self._costs.setdefault(record.app_id, CostState())
            state.total_cost += record.cost
            state.request_count += 1
            key = (record.backend, record.model)
            model_state = state.models.get(key)
            if model_state is None:
                model_state = ModelCostState(backend=record.backend, model=record.model)
                state.models[key] = model_state
            model_state.total_cost += record.cost
            model_state.request_count += 1
        await self._queue.put(record)

    async def _worker(self) -> None:
        try:
            while True:
                record = await self._queue.get()
                if record is None:
                    break
                if self._db_config is None:
                    continue
                await self._write_record(record)
        except asyncio.CancelledError:  # pragma: no cover - shutdown cancellation
            return

    async def _write_record(self, record: AccountingRecord) -> None:
        if psycopg is None:
            return
        async with self._db_lock:
            if not self._ensure_connection():
                return
            try:
                with self._db_conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO requests (
                            app_id, backend, model, operation,
                            cost, prompt_tokens, completion_tokens, total_tokens, latency_ms, created_at
                        ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            record.app_id,
                            record.backend,
                            record.model,
                            record.operation,
                            record.cost,
                            record.prompt_tokens,
                            record.completion_tokens,
                            record.total_tokens,
                            record.latency_ms,
                            record.created_at,
                        ),
                    )
            except Exception as exc:  # pragma: no cover
                log.error("Failed to persist accounting record: %s", exc)

    async def cost_total(self, app_id: str) -> CostState:
        async with self._cost_lock:
            state = self._costs.get(app_id)
            if state is None:
                return CostState()
            return CostState(total_cost=state.total_cost, request_count=state.request_count)

    async def over_cost_limit(self, app: AppDefinition) -> bool:
        if app.cost_limit is None:
            return False
        state = await self.cost_total(app.app_id)
        return state.total_cost >= app.cost_limit.limit

    async def snapshot(self) -> Dict[str, CostState]:
        async with self._cost_lock:
            return {
                app_id: CostState(
                    total_cost=state.total_cost,
                    request_count=state.request_count,
                    models={
                        key: ModelCostState(
                            backend=model_state.backend,
                            model=model_state.model,
                            total_cost=model_state.total_cost,
                            request_count=model_state.request_count,
                        )
                        for key, model_state in state.models.items()
                    },
                )
                for app_id, state in self._costs.items()
            }

    async def _bootstrap_costs(self) -> None:
        if psycopg is None or self._db_config is None:
            return
        start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        rows: Sequence[tuple[str, str, str, Any, Any]] = []
        async with self._db_lock:
            if not self._ensure_connection():
                return
            try:
                with self._db_conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT app_id, backend, model, SUM(cost) AS total_cost, COUNT(*) AS request_count
                        FROM requests
                        WHERE created_at >= %s
                        GROUP BY app_id, backend, model
                        """,
                        (start_of_day,),
                    )
                    rows = cur.fetchall()
            except Exception as exc:  # pragma: no cover
                log.error("Failed to load prior accounting state: %s", exc)
                return

        if not rows:
            return

        async with self._cost_lock:
            for app_id, backend, model, total_cost, request_count in rows:
                state = self._costs.setdefault(app_id, CostState())
                state.total_cost += float(total_cost)
                state.request_count += int(request_count)
                key = (backend, model)
                model_state = state.models.get(key)
                if model_state is None:
                    model_state = ModelCostState(backend=backend, model=model)
                    state.models[key] = model_state
                model_state.total_cost += float(total_cost)
                model_state.request_count += int(request_count)

        log.info("Restored accounting state for %d app/model combinations", len(rows))

    def _ensure_connection(self) -> bool:
        if self._db_config is None:
            return False
        if self._db_conn is not None:
            return True
        try:
            self._db_conn = psycopg.connect(
                host=self._db_config.host,
                port=self._db_config.port,
                dbname=self._db_config.database,
                user=self._db_config.username,
                password=self._db_config.password,
                connect_timeout=int(self._db_config.connect_timeout),
            )
            self._db_conn.autocommit = True
            return True
        except Exception as exc:  # pragma: no cover - network/db errors
            log.error("Failed to connect to Postgres: %s", exc)
            self._db_conn = None
            return False


def compute_cost(limit: Optional[CostLimitConfig], usage_tokens: Optional[int], cost_per_unit: float) -> float:
    if limit is None or usage_tokens is None:
        return 0.0
    return (usage_tokens / 1000.0) * cost_per_unit


__all__ = ["AccountingRecorder", "AccountingRecord", "compute_cost"]

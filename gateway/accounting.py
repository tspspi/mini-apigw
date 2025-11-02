"""Usage accounting and PostgreSQL backend"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from time import monotonic
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

if psycopg is not None:
    _DB_RECONNECT_ERRORS = tuple(
        getattr(psycopg, name)
        for name in ("OperationalError", "InterfaceError", "DatabaseError")
        if hasattr(psycopg, name)
    )
    if not _DB_RECONNECT_ERRORS:
        _DB_RECONNECT_ERRORS = (getattr(psycopg, "Error"),)
else:
    _DB_RECONNECT_ERRORS = (Exception,)


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
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass(slots=True)
class ModelCostState:
    backend: str
    model: str
    total_cost: float = 0.0
    request_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0

@dataclass(slots=True)
class CostState:
    total_cost: float = 0.0
    request_count: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_latency_ms: int = 0
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
        self._queries_since_connect = 0
        self._last_activity = monotonic()
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
            prompt_tokens = int(record.prompt_tokens or 0)
            completion_tokens = int(record.completion_tokens or 0)
            total_tokens = int(record.total_tokens or (prompt_tokens + completion_tokens))
            latency_ms = int(record.latency_ms or 0)
            state.total_prompt_tokens += prompt_tokens
            state.total_completion_tokens += completion_tokens
            state.total_tokens += total_tokens
            state.total_latency_ms += latency_ms
            key = (record.backend, record.model)
            model_state = state.models.get(key)
            if model_state is None:
                model_state = ModelCostState(backend=record.backend, model=record.model)
                state.models[key] = model_state
            model_state.total_cost += record.cost
            model_state.request_count += 1
            model_state.total_prompt_tokens += prompt_tokens
            model_state.total_completion_tokens += completion_tokens
            model_state.total_tokens += total_tokens
            model_state.total_latency_ms += latency_ms
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
        last_error: Optional[BaseException] = None
        for attempt in range(2):
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
                    self._queries_since_connect += 1
                    self._last_activity = monotonic()
                    if (
                        self._db_config is not None
                        and self._db_config.max_queries
                        and self._queries_since_connect >= self._db_config.max_queries
                    ):
                        self._reset_connection()
                    return
                except _DB_RECONNECT_ERRORS as exc:  # type: ignore[misc]
                    last_error = exc
                    log.warning("Database write failed on attempt %d/%d (%s); resetting connection", attempt + 1, 2, exc)
                    self._reset_connection()
                except Exception as exc:  # pragma: no cover - unexpected DB errors
                    log.error("Failed to persist accounting record: %s", exc)
                    return
        if last_error is not None:  # pragma: no cover - repeated connection failures
            log.error("Failed to persist accounting record after reconnect attempts: %s", last_error)


    async def cost_total(self, app_id: str) -> CostState:
        async with self._cost_lock:
            state = self._costs.get(app_id)
            if state is None:
                return CostState()
            return CostState(
                total_cost=state.total_cost,
                request_count=state.request_count,
                total_prompt_tokens=state.total_prompt_tokens,
                total_completion_tokens=state.total_completion_tokens,
                total_tokens=state.total_tokens,
                total_latency_ms=state.total_latency_ms,
            )

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
                    total_prompt_tokens=state.total_prompt_tokens,
                    total_completion_tokens=state.total_completion_tokens,
                    total_tokens=state.total_tokens,
                    total_latency_ms=state.total_latency_ms,
                    models={
                        key: ModelCostState(
                            backend=model_state.backend,
                            model=model_state.model,
                            total_cost=model_state.total_cost,
                            request_count=model_state.request_count,
                            total_prompt_tokens=model_state.total_prompt_tokens,
                            total_completion_tokens=model_state.total_completion_tokens,
                            total_tokens=model_state.total_tokens,
                            total_latency_ms=model_state.total_latency_ms,
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
        rows: Sequence[tuple[str, str, str, Any, Any, Any, Any, Any, Any]] = []
        async with self._db_lock:
            if not self._ensure_connection():
                return
            try:
                with self._db_conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT
                            app_id,
                            backend,
                            model,
                            SUM(cost) AS total_cost,
                            COUNT(*) AS request_count,
                            COALESCE(SUM(prompt_tokens), 0) AS prompt_tokens,
                            COALESCE(SUM(completion_tokens), 0) AS completion_tokens,
                            COALESCE(SUM(total_tokens), 0) AS total_tokens,
                            COALESCE(SUM(latency_ms), 0) AS latency_ms
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
            for (
                app_id,
                backend,
                model,
                total_cost,
                request_count,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                latency_ms,
            ) in rows:
                state = self._costs.setdefault(app_id, CostState())
                state.total_cost += float(total_cost)
                state.request_count += int(request_count)
                state.total_prompt_tokens += int(prompt_tokens)
                state.total_completion_tokens += int(completion_tokens)
                state.total_tokens += int(total_tokens)
                state.total_latency_ms += int(latency_ms)
                key = (backend, model)
                model_state = state.models.get(key)
                if model_state is None:
                    model_state = ModelCostState(backend=backend, model=model)
                    state.models[key] = model_state
                model_state.total_cost += float(total_cost)
                model_state.request_count += int(request_count)
                model_state.total_prompt_tokens += int(prompt_tokens)
                model_state.total_completion_tokens += int(completion_tokens)
                model_state.total_tokens += int(total_tokens)
                model_state.total_latency_ms += int(latency_ms)

        log.info("Restored accounting state for %d app/model combinations", len(rows))

    def _reset_connection(self) -> None:
        if self._db_conn is not None:
            try:
                self._db_conn.close()
            except Exception:  # pragma: no cover - defensive close
                pass
        self._db_conn = None
        self._queries_since_connect = 0
        self._last_activity = monotonic()

    def _connection_kwargs(self) -> Dict[str, Any]:
        if self._db_config is None:
            raise RuntimeError("Database configuration missing")
        cfg = self._db_config
        kwargs: Dict[str, Any] = {
            "host": cfg.host,
            "port": cfg.port,
            "dbname": cfg.database,
            "user": cfg.username,
            "password": cfg.password,
            "connect_timeout": int(cfg.connect_timeout),
        }
        if cfg.ssl_mode:
            kwargs["sslmode"] = cfg.ssl_mode
            if cfg.ssl_cert:
                kwargs["sslcert"] = cfg.ssl_cert
            if cfg.ssl_key:
                kwargs["sslkey"] = cfg.ssl_key
            if cfg.ssl_ca:
                kwargs["sslrootcert"] = cfg.ssl_ca
        if self._db_driver_name in {"psycopg", "psycopg2"}:
            idle_seconds = max(int(cfg.max_inactive_time), 1)
            interval = max(int(idle_seconds / 3), 1)
            kwargs.update(
                keepalives=1,
                keepalives_idle=idle_seconds,
                keepalives_interval=interval,
                keepalives_count=5,
            )
        return kwargs

    def _should_refresh_connection(self) -> bool:
        if self._db_config is None or self._db_conn is None:
            return False
        if self._db_config.max_queries and self._queries_since_connect >= self._db_config.max_queries:
            return True
        if self._db_config.max_inactive_time and (monotonic() - self._last_activity) >= self._db_config.max_inactive_time:
            return True
        return False

    def _ensure_connection(self) -> bool:
        if self._db_config is None or psycopg is None:
            return False
        if self._db_conn is not None:
            if not self._should_refresh_connection():
                return True
            self._reset_connection()
        try:
            self._db_conn = psycopg.connect(**self._connection_kwargs())
            self._db_conn.autocommit = True
            self._queries_since_connect = 0
            self._last_activity = monotonic()
            return True
        except Exception as exc:  # pragma: no cover - network/db errors
            log.error("Failed to connect to Postgres: %s", exc)
            self._reset_connection()
            return False



def compute_cost(limit: Optional[CostLimitConfig], usage_tokens: Optional[int], cost_per_unit: float) -> float:
    if limit is None or usage_tokens is None:
        return 0.0
    return (usage_tokens / 1000.0) * cost_per_unit


__all__ = ["AccountingRecorder", "AccountingRecord", "compute_cost"]

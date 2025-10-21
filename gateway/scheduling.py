"""Backend scheduling with per-backend concurrency and sequence groups."""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from typing import AsyncIterator, Dict, Optional

from .config import BackendConfig, BackendDefinition
from .backends import BackendClient


class SchedulingError(RuntimeError):
    pass

class SequenceGroup:
    def __init__(self, name: str):
        self.name = name
        self._queue: deque[asyncio.Future[None]] = deque()
        self._lock = asyncio.Lock()

    @property
    def queue_length(self) -> int:
        qlen = len(self._queue)
        return max(0, qlen - 1) if qlen else 0

    async def acquire(self) -> "_SequenceTicket":
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._queue.append(fut)
        if len(self._queue) == 1:
            fut.set_result(None)
        await fut
        await self._lock.acquire()
        return _SequenceTicket(self)

    async def _release(self) -> None:
        self._lock.release()
        if self._queue:
            self._queue.popleft()
        if self._queue:
            next_fut = self._queue[0]
            if not next_fut.done():
                next_fut.set_result(None)

class _SequenceTicket:
    def __init__(self, group: SequenceGroup):
        self._group = group

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self._group._release()

@dataclass
class BackendRuntime:
    definition: BackendDefinition
    client: BackendClient
    semaphore: asyncio.Semaphore
    sequence_group: Optional[SequenceGroup]
    in_flight: int = 0
    waiters: int = 0

    async def acquire(self) -> "BackendTicket":
        seq_ticket = None
        if self.sequence_group is not None:
            seq_ticket = await self.sequence_group.acquire()
        self.waiters += 1
        await self.semaphore.acquire()
        self.waiters -= 1
        self.in_flight += 1
        return BackendTicket(self, seq_ticket)

    def release(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        self.semaphore.release()

class BackendTicket:
    def __init__(self, runtime: BackendRuntime, sequence_ticket: Optional[_SequenceTicket]):
        self._runtime = runtime
        self._sequence_ticket = sequence_ticket

    async def __aenter__(self):
        return self._runtime

    async def __aexit__(self, exc_type, exc, tb):
        self._runtime.release()
        if self._sequence_ticket is not None:
            await self._sequence_ticket.__aexit__(exc_type, exc, tb)

class Scheduler:
    """Coordinates backend execution and exposes stats."""

    def __init__(self, config: BackendConfig, backends: Dict[str, BackendClient]):
        self._groups: Dict[str, SequenceGroup] = {}
        for name in config.sequence_groups.keys():
            self._groups[name] = SequenceGroup(name)
        self._runtimes: Dict[str, BackendRuntime] = {}
        for backend in config.backends:
            sequence_group = self._groups.get(backend.sequence_group) if backend.sequence_group else None
            semaphore = asyncio.Semaphore(max(1, backend.concurrency))
            client = backends[backend.name]
            self._runtimes[backend.name] = BackendRuntime(
                definition=backend,
                client=client,
                semaphore=semaphore,
                sequence_group=sequence_group,
            )

    def runtime(self, backend_name: str) -> BackendRuntime:
        try:
            return self._runtimes[backend_name]
        except KeyError as exc:
            raise SchedulingError(f"Unknown backend '{backend_name}'") from exc

    async def acquire(self, backend_name: str) -> BackendTicket:
        runtime = self.runtime(backend_name)
        return await runtime.acquire()

    def stats(self) -> Dict[str, Dict[str, int]]:
        per_backend = {
            name: {"in_flight": runtime.in_flight, "queue_len": runtime.waiters}
            for name, runtime in self._runtimes.items()
        }
        per_group = {
            name: {"queue_len": group.queue_length}
            for name, group in self._groups.items()
        }
        return {"per_backend": per_backend, "per_group": per_group}

__all__ = ["Scheduler", "SchedulingError"]

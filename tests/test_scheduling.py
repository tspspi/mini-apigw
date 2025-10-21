import asyncio

import pytest

from gateway.backends.base import BackendClient
from gateway.config import BackendConfig, BackendDefinition, BackendSupports
from gateway.scheduling import Scheduler


class DummyBackend(BackendClient):
    async def chat(self, model, payload, stream=False):  # pragma: no cover - not used
        raise NotImplementedError

    async def completions(self, model, payload):  # pragma: no cover - not used
        raise NotImplementedError

    async def embeddings(self, model, payload):  # pragma: no cover - not used
        raise NotImplementedError

    async def images(self, model, payload):  # pragma: no cover - not used
        raise NotImplementedError

    async def models(self):  # pragma: no cover - not used
        raise NotImplementedError


@pytest.mark.asyncio
async def test_scheduler_respects_concurrency():
    backend_def = BackendDefinition(
        type="openai",
        name="openai",
        base_url="https://api",
        supports=BackendSupports(chat=["gpt"]),
        concurrency=1,
    )
    config = BackendConfig(aliases={}, sequence_groups={}, backends=[backend_def])
    scheduler = Scheduler(config, {"openai": DummyBackend(backend_def)})

    order = []

    async def worker(tag: str):
        async with await scheduler.acquire("openai"):
            order.append(f"start-{tag}")
            await asyncio.sleep(0.01)
            order.append(f"end-{tag}")

    await asyncio.gather(worker("a"), worker("b"))
    assert order[0] == "start-a"
    assert order[1] == "end-a"
    assert order[2] == "start-b"
    assert order[3] == "end-b"

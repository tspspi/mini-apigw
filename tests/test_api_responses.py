from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gateway.accounting import AccountingRecord
from gateway.api import v1
from gateway.backends.base import BackendResult, BackendUsage


class DummyRuntime:
    def __init__(self):
        self.recorded = None
        self.last_operation = None
        self.last_payload = None

    async def authenticate(self, api_key: str):
        assert api_key == "test-key"
        return SimpleNamespace(app=SimpleNamespace(app_id="app-test"))

    async def check_policy(self, app, model: str):
        return SimpleNamespace(allowed=True, reason=None)

    async def app_over_limit(self, app) -> bool:
        return False

    async def execute_operation(self, *, operation: str, model: str, payload, stream: bool = False):
        self.last_operation = operation
        self.last_payload = payload
        usage = BackendUsage(input_tokens=3, output_tokens=4, total_tokens=7)
        body = {
            "id": "resp_test",
            "output": [
                {"type": "message", "content": [{"type": "text", "text": "hello"}]}
            ],
        }
        result = BackendResult(body=body, usage=usage)
        record = AccountingRecord(
            app_id="<unassigned>",
            backend="openai",
            model=model,
            operation=operation,
            cost=0.0,
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
            latency_ms=0,
        )
        return result, record, "openai"

    async def record_usage(self, record: AccountingRecord, app_id: str):
        record.app_id = app_id
        self.recorded = (record, app_id)

    def trace_manager(self):  # pragma: no cover - no tracing in tests
        return None


def create_test_app(runtime: DummyRuntime) -> TestClient:
    app = FastAPI()
    app.include_router(v1.router)
    app.state.runtime = runtime
    return TestClient(app)


def test_responses_proxy_returns_backend_payload():
    runtime = DummyRuntime()
    with create_test_app(runtime) as client:
        response = client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer test-key"},
            json={"model": "gpt-4o-mini", "input": "hi there"},
        )
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "resp_test"
    assert runtime.last_operation == "responses"
    assert runtime.last_payload["model"] == "gpt-4o-mini"
    assert runtime.recorded[1] == "app-test"

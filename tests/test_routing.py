from gateway.config import BackendConfig, BackendDefinition, BackendSupports, ResponsesShimBackendConfig
from gateway.routing import ModelRouter


def test_alias_resolution():
    supports = BackendSupports(chat=["openai:gpt-4o-mini"])
    backend = BackendDefinition(type="openai", name="openai", base_url="https://api", supports=supports)
    config = BackendConfig(aliases={"gpt-4o-mini": "openai:gpt-4o-mini"}, sequence_groups={}, backends=[backend])
    router = ModelRouter(config)
    candidates = router.candidates("gpt-4o-mini", "chat")
    assert candidates[0].backend.name == "openai"

def test_responses_operation_resolves_backend():
    supports = BackendSupports(responses=["openai:gpt-4o"], chat=["openai:gpt-4o"])
    backend = BackendDefinition(type="openai", name="openai", base_url="https://api", supports=supports)
    config = BackendConfig(aliases={}, sequence_groups={}, backends=[backend])
    router = ModelRouter(config)
    candidates = router.candidates("openai:gpt-4o", "responses")
    assert candidates[0].backend.name == "openai"


def test_responses_operation_falls_back_to_shim():
    supports = BackendSupports(chat=["ollama:gpt-oss"], responses=[])
    shim_cfg = ResponsesShimBackendConfig(enabled=True, operation="chat")
    backend = BackendDefinition(
        type="ollama",
        name="ollama",
        base_url="http://localhost",
        supports=supports,
        responses_shim=shim_cfg,
    )
    config = BackendConfig(aliases={"gpt-oss": "ollama:gpt-oss"}, sequence_groups={}, backends=[backend])
    router = ModelRouter(config)
    candidates = router.candidates("gpt-oss", "responses")
    assert candidates[0].backend.name == "ollama"
    assert candidates[0].shim_operation == "chat"


from gateway.config import BackendConfig, BackendDefinition, BackendSupports
from gateway.routing import ModelRouter


def test_alias_resolution():
    supports = BackendSupports(chat=["openai:gpt-4o-mini"])
    backend = BackendDefinition(type="openai", name="openai", base_url="https://api", supports=supports)
    config = BackendConfig(aliases={"gpt-4o-mini": "openai:gpt-4o-mini"}, sequence_groups={}, backends=[backend])
    router = ModelRouter(config)
    candidates = router.candidates("gpt-4o-mini", "chat")
    assert candidates[0].backend.name == "openai"

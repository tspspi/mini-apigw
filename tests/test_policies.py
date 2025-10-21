from gateway.config import AppDefinition, AppPolicy
from gateway.policies import PolicyEngine


def make_app(allow=None, deny=None):
    return AppDefinition(app_id="app", api_keys=["key"], policy=AppPolicy(allow=allow or [], deny=deny or []))


def test_allow_list_matches():
    engine = PolicyEngine()
    app = make_app(allow=["gpt-4*"], deny=[])
    decision = engine.is_allowed(app, "gpt-4o-mini")
    assert decision.allowed


def test_deny_list_blocks():
    engine = PolicyEngine()
    app = make_app(allow=[], deny=["ollama*"])
    decision = engine.is_allowed(app, "ollama:phi")
    assert not decision.allowed
    assert "denied" in decision.reason

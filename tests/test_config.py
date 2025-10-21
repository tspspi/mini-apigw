import json
from pathlib import Path

import pytest

from gateway.config import (
    ConfigError,
    load_apps_config,
    load_backends_config,
    load_daemon_config,
)


def test_load_daemon_config(tmp_path: Path):
    data = {
        "listen": {"host_v4": "0.0.0.0", "host_v6": "::", "port": 9090},
        "admin": {"bind": ["127.0.0.1:8081"], "stats_networks": []},
        "logging": {"level": "DEBUG", "redact_prompts": False, "access_log": False},
        "reload": {"enable_sighup": False},
        "timeouts": {"default_connect_s": 10, "default_read_s": 20}
    }
    path = tmp_path / "daemon.json"
    path.write_text(json.dumps(data))
    cfg = load_daemon_config(path)
    assert cfg.listen.port == 9090
    assert cfg.reload.enable_sighup is False


def test_load_backends_config(tmp_path: Path):
    data = {
        "aliases": {"chat-default": "gpt-4"},
        "sequence_groups": {"gpu": {"description": "GPU-bound"}},
        "backends": [
            {
                "type": "openai",
                "name": "openai-primary",
                "base_url": "https://api.openai.com/v1",
                "api_key": "sk-key",
                "concurrency": 2,
                "supports": {"chat": ["gpt-4"]}
            }
        ]
    }
    path = tmp_path / "backends.json"
    path.write_text(json.dumps(data))
    cfg = load_backends_config(path)
    assert cfg.backends[0].name == "openai-primary"
    assert cfg.backends[0].supports.chat == ["gpt-4"]


def test_load_apps_config_requires_key(tmp_path: Path):
    data = {"apps": [{"app_id": "test", "api_keys": []}]}
    path = tmp_path / "apps.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ConfigError):
        load_apps_config(path)

"""Configuration loading and validation for mini-apigw."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

log = logging.getLogger(__name__)


class ConfigError(RuntimeError):
    """Configuration error exception

    This exception is raised whenever the configuration
    files are invalid
    """

@dataclass(slots=True)
class ListenConfig:
    """Listener configuration for the public API endpoint.

    When neither ``host_v4`` nor ``host_v6`` are provided the service falls back to
    listening on a Unix domain socket. The default socket path mirrors the systemd
    style layout under ``/var/llmgw`` but can be overridden through configuration or
    CLI flags."""

    host_v4: Optional[str] = None
    host_v6: Optional[str] = None
    port: Optional[int] = None
    unix_socket: Optional[str] = None


@dataclass(slots=True)
class AdminConfig:
    """Administration interface configuration

    The admin interface allows to trigger server shutdown as well
    as configuratoin reloading the same way as signals do (if enabled).
    It defaults to port 8081 on both IPv4 and IPv6"""

    bind: List[str] = field(default_factory=lambda: ["127.0.0.1:8081", "[::1]:8081"])
    stats_networks: List[str] = field(default_factory=list)


@dataclass(slots=True)
class LoggingConfig:
    """Logging configuration

    Specifies the log level, if we should trim the prompts and if we should
    run an access log"""

    level: str = "INFO"
    redact_prompts: bool = True
    access_log: bool = True
    file: Optional[str] = None


@dataclass(slots=True)
class ReloadConfig:
    """SIGHUP handler

    If enabled the SIGHUP handler allows to trigger reloading of configuration
    files"""

    enable_sighup: bool = True


@dataclass(slots=True)
class TimeoutConfig:
    default_connect_s: float = 60.0
    default_read_s: float = 600.0


@dataclass(slots=True)
class DatabaseConfig:
    host: str
    database: str
    username: str
    password: str
    port: int = 5432
    max_queries: int = 50000
    max_inactive_time: float = 300.0
    connect_timeout: float = 10.0
    command_timeout: float = 60.0
    ssl_mode: Optional[str] = None
    ssl_cert: Optional[str] = None
    ssl_key: Optional[str] = None
    ssl_ca: Optional[str] = None


@dataclass(slots=True)
class DaemonConfig:
    listen: ListenConfig
    admin: AdminConfig
    logging: LoggingConfig
    reload: ReloadConfig
    timeouts: TimeoutConfig
    database: Optional[DatabaseConfig]


@dataclass(slots=True)
class BackendSupports:
    chat: List[str] = field(default_factory=list)
    completions: List[str] = field(default_factory=list)
    embeddings: List[str] = field(default_factory=list)
    images: List[str] = field(default_factory=list)


@dataclass(slots=True)
class ModelCostConfig:
    prompt: Optional[float] = None
    completion: Optional[float] = None
    image: Optional[float] = None

    def with_defaults(self, default: ModelCostConfig) -> ModelCostConfig:
        return ModelCostConfig(
            prompt=self.prompt if self.prompt is not None else default.prompt,
            completion=self.completion if self.completion is not None else default.completion,
            image=self.image if self.image is not None else default.image,
        )


@dataclass(slots=True)
class BackendCostConfig:
    default: ModelCostConfig = field(
        default_factory=lambda: ModelCostConfig(prompt=0.0, completion=0.0, image=0.0)
    )
    currency: str = "usd"
    unit: str = "1k_tokens"
    models: Dict[str, ModelCostConfig] = field(default_factory=dict)

    @property
    def prompt(self) -> float:
        return float(self.default.prompt or 0.0)

    @property
    def completion(self) -> float:
        return float(self.default.completion or 0.0)

    @property
    def image(self) -> float:
        return float(self.default.image or 0.0)

    def rate_for(self, model: str) -> ModelCostConfig:
        candidate = self.models.get(model)
        if candidate is None and ":" in model:
            suffix = model.split(":", 1)[1]
            candidate = self.models.get(suffix)
        if candidate is None:
            return self.default
        return candidate.with_defaults(self.default)


@dataclass(slots=True)
class BackendDefinition:
    type: str
    name: str
    base_url: str
    sequence_group: Optional[str] = None
    concurrency: int = 1
    supports: BackendSupports = field(default_factory=BackendSupports)
    cost: BackendCostConfig = field(default_factory=BackendCostConfig)
    api_key: Optional[str] = None
    request_timeout_s: Optional[float] = None
    extra_headers: Mapping[str, str] = field(default_factory=dict)
    auto_models: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendSequenceGroup:
    name: str
    description: Optional[str] = None


@dataclass(slots=True)
class BackendConfig:
    aliases: Mapping[str, str] = field(default_factory=dict)
    sequence_groups: Mapping[str, BackendSequenceGroup] = field(default_factory=dict)
    backends: List[BackendDefinition] = field(default_factory=list)


@dataclass(slots=True)
class CostLimitConfig:
    period: str = "day"
    limit: float = 0.0


@dataclass(slots=True)
class AppPolicy:
    allow: List[str] = field(default_factory=list)
    deny: List[str] = field(default_factory=list)


@dataclass(slots=True)
class AppTraceConfig:
    file: Optional[str] = None
    image_dir: Optional[str] = None
    include_prompts: bool = False
    include_response: bool = False
    include_keys: bool = False


@dataclass(slots=True)
class AppDefinition:
    app_id: str
    name: Optional[str] = None
    api_keys: List[str] = field(default_factory=list)
    policy: AppPolicy = field(default_factory=AppPolicy)
    cost_limit: Optional[CostLimitConfig] = None
    trace: Optional[AppTraceConfig] = None


@dataclass(slots=True)
class AppsConfig:
    apps: List[AppDefinition] = field(default_factory=list)

    def api_key_index(self) -> Dict[str, AppDefinition]:
        mapping: Dict[str, AppDefinition] = {}
        for app in self.apps:
            for key in app.api_keys:
                if key in mapping and mapping[key].app_id != app.app_id:
                    raise ConfigError(f"API key {key} assigned to multiple apps")
                mapping[key] = app
        return mapping


@dataclass(slots=True)
class ConfigBundle:
    daemon: DaemonConfig
    backends: BackendConfig
    apps: AppsConfig


def _expect(obj: MutableMapping[str, Any], key: str, ctx: str) -> Any:
    if key not in obj:
        raise ConfigError(f"Missing field '{key}' in {ctx}")
    return obj[key]


def _load_list(value: Any, ctx: str) -> List[Any]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ConfigError(f"Expected list for {ctx}")
    return value


def _load_mapping(value: Any, ctx: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ConfigError(f"Expected mapping for {ctx}")
    return value


def _load_backend_supports(raw: Mapping[str, Any]) -> BackendSupports:
    return BackendSupports(
        chat=_load_list(raw.get("chat"), "supports.chat"),
        completions=_load_list(raw.get("completions"), "supports.completions"),
        embeddings=_load_list(raw.get("embeddings"), "supports.embeddings"),
        images=_load_list(raw.get("images"), "supports.images"),
    )


def _load_model_cost(
    raw: Mapping[str, Any], *, allow_partial: bool = False, defaults: ModelCostConfig | None = None
) -> ModelCostConfig:
    prompt = raw.get("prompt")
    completion = raw.get("completion")
    image = raw.get("image")

    if allow_partial and defaults is not None:
        base = defaults
    else:
        base = ModelCostConfig(prompt=0.0, completion=0.0, image=0.0)

    return ModelCostConfig(
        prompt=float(prompt) if prompt is not None else base.prompt,
        completion=float(completion) if completion is not None else base.completion,
        image=float(image) if image is not None else base.image,
    )


def _load_backend_cost(raw: Mapping[str, Any]) -> BackendCostConfig:
    currency = str(raw.get("currency", "usd"))
    unit = str(raw.get("unit", "1k_tokens"))

    default_raw = raw.get("default")
    if default_raw is not None:
        default = _load_model_cost(_load_mapping(default_raw, "cost.default"))
    else:
        default = ModelCostConfig(
            prompt=float(raw.get("prompt", 0.0)),
            completion=float(raw.get("completion", 0.0)),
            image=float(raw.get("image", 0.0)),
        )

    models: Dict[str, ModelCostConfig] = {}
    models_raw = raw.get("models")
    if models_raw is not None:
        models_mapping = _load_mapping(models_raw, "cost.models")
        for model_name, model_value in models_mapping.items():
            if not isinstance(model_value, Mapping):
                raise ConfigError("Expected mapping for cost.models entries")
            models[model_name] = _load_model_cost(model_value, allow_partial=True, defaults=default)

    return BackendCostConfig(default=default, currency=currency, unit=unit, models=models)


def _load_backend_definition(raw: Mapping[str, Any]) -> BackendDefinition:
    supports = _load_backend_supports(_load_mapping(raw.get("supports"), "supports"))
    cost = _load_backend_cost(_load_mapping(raw.get("cost"), "cost"))
    extra_headers = dict(_load_mapping(raw.get("extra_headers"), "extra_headers"))

    return BackendDefinition(
        type=str(_expect(dict(raw), "type", "backend")),
        name=str(_expect(dict(raw), "name", "backend")),
        base_url=str(_expect(dict(raw), "base_url", "backend")),
        sequence_group=raw.get("sequence_group"),
        concurrency=int(raw.get("concurrency", 1)),
        supports=supports,
        cost=cost,
        api_key=raw.get("api_key"),
        request_timeout_s=float(raw["request_timeout_s"]) if raw.get("request_timeout_s") is not None else None,
        extra_headers=extra_headers,
        auto_models=bool(raw.get("auto_models", False)),
        metadata=_load_mapping(raw.get("metadata"), "metadata"),
    )


def _load_backend_sequence_group(raw: Mapping[str, Any], name: str) -> BackendSequenceGroup:
    return BackendSequenceGroup(name=name, description=raw.get("description"))


def load_backends_config(path: Path) -> BackendConfig:
    data = _load_json(path)
    if not isinstance(data, Mapping):
        raise ConfigError("backends.json must contain an object")

    aliases_raw = _load_mapping(data.get("aliases"), "aliases")
    sequence_groups_raw = _load_mapping(data.get("sequence_groups"), "sequence_groups")
    backends_raw = _load_list(data.get("backends"), "backends")

    sequence_groups = {
        name: _load_backend_sequence_group(defn, name)
        for name, defn in sequence_groups_raw.items()
    }

    backends = [_load_backend_definition(raw) for raw in backends_raw]
    names = {backend.name for backend in backends}
    if len(names) != len(backends):
        raise ConfigError("Duplicate backend names detected")

    return BackendConfig(
        aliases=dict(aliases_raw),
        sequence_groups=sequence_groups,
        backends=backends,
    )


def _load_cost_limit(raw: Mapping[str, Any]) -> CostLimitConfig:
    return CostLimitConfig(
        period=str(raw.get("period", "day")),
        limit=float(raw.get("limit", 0.0)),
    )


def _load_app_policy(raw: Mapping[str, Any]) -> AppPolicy:
    return AppPolicy(
        allow=[str(item) for item in _load_list(raw.get("allow"), "policy.allow")],
        deny=[str(item) for item in _load_list(raw.get("deny"), "policy.deny")],
    )


def _load_app_trace_config(raw: Mapping[str, Any]) -> AppTraceConfig:
    return AppTraceConfig(
        file=str(raw["file"]) if raw.get("file") is not None else None,
        image_dir=str(raw["imagedir"]) if raw.get("imagedir") is not None else None,
        include_prompts=bool(raw.get("includeprompts", False)),
        include_response=bool(raw.get("includeresponse", False)),
        include_keys=bool(raw.get("includekeys", False)),
    )


def _load_app_definition(raw: Mapping[str, Any]) -> AppDefinition:
    policy = _load_app_policy(_load_mapping(raw.get("policy"), "policy"))
    cost_limit_raw = raw.get("cost_limit")
    cost_limit = None
    if cost_limit_raw is not None:
        cost_limit = _load_cost_limit(_load_mapping(cost_limit_raw, "cost_limit"))

    trace_raw = raw.get("trace")
    trace = None
    if trace_raw is not None:
        trace = _load_app_trace_config(_load_mapping(trace_raw, "trace"))

    api_keys_raw = _load_list(raw.get("api_keys"), "api_keys")
    if not api_keys_raw:
        raise ConfigError("App definition requires at least one api_key")

    return AppDefinition(
        app_id=str(_expect(dict(raw), "app_id", "app")),
        name=raw.get("name"),
        api_keys=[str(key) for key in api_keys_raw],
        policy=policy,
        cost_limit=cost_limit,
        trace=trace,
    )


def load_apps_config(path: Path) -> AppsConfig:
    data = _load_json(path)
    if not isinstance(data, Mapping):
        raise ConfigError("apps.json must contain an object")

    apps_raw = _load_list(data.get("apps"), "apps")
    apps = [_load_app_definition(raw) for raw in apps_raw]
    if not apps:
        raise ConfigError("apps.json requires at least one app entry")
    return AppsConfig(apps=apps)


def load_daemon_config(path: Path) -> DaemonConfig:
    data = _load_json(path)
    if not isinstance(data, Mapping):
        raise ConfigError("daemon.json must contain an object")

    listen_raw = _load_mapping(_expect(dict(data), "listen", "daemon"), "listen")
    admin_raw = _load_mapping(_expect(dict(data), "admin", "daemon"), "admin")
    logging_raw = _load_mapping(_expect(dict(data), "logging", "daemon"), "logging")
    reload_raw = _load_mapping(_expect(dict(data), "reload", "daemon"), "reload")
    timeouts_raw = _load_mapping(_expect(dict(data), "timeouts", "daemon"), "timeouts")

    host_v4_value = listen_raw.get("host_v4")
    host_v6_value = listen_raw.get("host_v6")
    unix_socket_value = listen_raw.get("unix_socket")
    port_value = listen_raw.get("port")

    host_v4 = str(host_v4_value) if host_v4_value is not None else None
    host_v6 = str(host_v6_value) if host_v6_value is not None else None

    port: Optional[int]
    if port_value is not None:
        port = int(port_value)
    elif host_v4 or host_v6:
        port = 8080
    else:
        port = None

    unix_socket = str(unix_socket_value) if unix_socket_value is not None else None
    if unix_socket is None and not host_v4 and not host_v6:
        unix_socket = "/var/llmgw/llmgw.sock"

    listen = ListenConfig(
        host_v4=host_v4,
        host_v6=host_v6,
        port=port,
        unix_socket=unix_socket,
    )

    admin = AdminConfig(
        bind=[str(item) for item in _load_list(admin_raw.get("bind"), "admin.bind")],
        stats_networks=[str(item) for item in _load_list(admin_raw.get("stats_networks"), "admin.stats_networks")],
    )

    logging_cfg = LoggingConfig(
        level=str(logging_raw.get("level", "INFO")),
        redact_prompts=bool(logging_raw.get("redact_prompts", True)),
        access_log=bool(logging_raw.get("access_log", True)),
        file=(
            str(logging_raw.get("file"))
            if logging_raw.get("file") is not None
            else None
        ),
    )

    reload_cfg = ReloadConfig(enable_sighup=bool(reload_raw.get("enable_sighup", True)))

    timeouts = TimeoutConfig(
        default_connect_s=float(timeouts_raw.get("default_connect_s", 60.0)),
        default_read_s=float(timeouts_raw.get("default_read_s", 600.0)),
    )

    database_cfg = None
    if data.get("database"):
        db_raw = _load_mapping(data.get("database"), "database")
        database_cfg = DatabaseConfig(
            host=str(_expect(dict(db_raw), "host", "database")),
            port=int(db_raw.get("port", 5432)),
            database=str(_expect(dict(db_raw), "database", "database")),
            username=str(_expect(dict(db_raw), "username", "database")),
            password=str(_expect(dict(db_raw), "password", "database")),
            max_queries=int(db_raw.get("max_queries", 50000)),
            max_inactive_time=float(db_raw.get("max_inactive_time", 300.0)),
            connect_timeout=float(db_raw.get("connect_timeout", 10.0)),
            command_timeout=float(db_raw.get("command_timeout", 60.0)),
            ssl_mode=db_raw.get("ssl_mode"),
            ssl_cert=db_raw.get("ssl_cert"),
            ssl_key=db_raw.get("ssl_key"),
            ssl_ca=db_raw.get("ssl_ca"),
        )

    return DaemonConfig(
        listen=listen,
        admin=admin,
        logging=logging_cfg,
        reload=reload_cfg,
        timeouts=timeouts,
        database=database_cfg,
    )


def _load_json(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file missing: {path}") from exc
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc


class ConfigManager:
    """Thread-safe holder for active configuration bundle."""

    def __init__(self, daemon_path: Path, backends_path: Path, apps_path: Path):
        self._daemon_path = daemon_path
        self._backends_path = backends_path
        self._apps_path = apps_path
        self._lock = RLock()
        self._bundle: Optional[ConfigBundle] = None

    def load(self) -> ConfigBundle:
        """Load all configuration files and swap the active bundle."""
        with self._lock:
            log.debug("Loading configuration files")
            daemon = load_daemon_config(self._daemon_path)
            backends = load_backends_config(self._backends_path)
            apps = load_apps_config(self._apps_path)
            self._bundle = ConfigBundle(daemon=daemon, backends=backends, apps=apps)
            return self._bundle

    def current(self) -> ConfigBundle:
        with self._lock:
            if self._bundle is None:
                raise ConfigError("Configuration has not been loaded yet")
            return self._bundle


__all__ = [
    "AdminConfig",
    "AppDefinition",
    "AppPolicy",
    "AppTraceConfig",
    "AppsConfig",
    "BackendConfig",
    "BackendCostConfig",
    "BackendDefinition",
    "BackendSequenceGroup",
    "BackendSupports",
    "ConfigBundle",
    "ConfigError",
    "ConfigManager",
    "CostLimitConfig",
    "DaemonConfig",
    "DatabaseConfig",
    "ListenConfig",
    "LoggingConfig",
    "ReloadConfig",
    "TimeoutConfig",
]

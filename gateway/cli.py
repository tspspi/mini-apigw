"""Console entry point for the gateway."""

from __future__ import annotations

import argparse
import os
import secrets
import sys
import time
from multiprocessing import Process
from pathlib import Path
from typing import Optional, Sequence, Tuple

import requests
import uvicorn

from .app import create_app
from .config import ConfigError, DaemonConfig, load_daemon_config

DEFAULT_SOCKET_PATH = "/var/llmgw/llmgw.sock"
DEFAULT_HOST_FALLBACK = "127.0.0.1"


def _resolve_config_dir(config_dir: Optional[str]) -> Path:
    if config_dir:
        return Path(config_dir)
    env_dir = os.environ.get("MINIAPIGW_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)
    return Path.cwd() / "config"

def _remove_stale_socket(path: str) -> None:
    socket_path = Path(path)
    try:
        if socket_path.exists():
            socket_path.unlink()
    except OSError:
        pass

def _serve_uvicorn(
    config_dir: str,
    host: Optional[str],
    port: Optional[int],
    uds: Optional[str],
    reload_enabled: bool,
    log_level: str,
) -> None:
    cfg_path = Path(config_dir)
    try:
        app = create_app(cfg_path)
    except ConfigError as exc:  # pragma: no cover - runtime setup error
        print(f"[error] Failed to create application: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    if uds:
        _remove_stale_socket(uds)

    config = uvicorn.Config(
        app,
        host=host or DEFAULT_HOST_FALLBACK,
        port=port or 8000,
        uds=uds,
        reload=reload_enabled,
        log_level=log_level,
    )
    server = uvicorn.Server(config)
    app.state.server = server
    server.run()

def _start_background(
    cfg_dir: Path,
    host: Optional[str],
    port: Optional[int],
    uds: Optional[str],
    reload_enabled: bool,
    log_level: str,
) -> None:
    process = Process(
        target=_serve_uvicorn,
        args=(str(cfg_dir), host, port, uds, reload_enabled, log_level),
        daemon=False,
    )
    process.start()
    time.sleep(0.1)
    if process.exitcode not in (None, 0):
        print("[error] Gateway process terminated during start-up", file=sys.stderr)
        sys.exit(process.exitcode or 1)
    desc = f"unix:{uds}" if uds else f"http://{host}:{port}"
    print(f"mini-apigw started in background (PID {process.pid}) listening on {desc}")

def _determine_listen_target(
    args: argparse.Namespace, daemon_cfg: DaemonConfig
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    host_override = getattr(args, "host", None)
    port_override = getattr(args, "port", None)
    unix_socket_override = getattr(args, "unix_socket", None)

    if unix_socket_override and (host_override or port_override):
        raise ValueError("Cannot combine --unix-socket with --host/--port overrides")

    listen = daemon_cfg.listen

    if unix_socket_override:
        return None, None, unix_socket_override

    host = host_override or listen.host_v6 or listen.host_v4
    port = port_override if port_override is not None else listen.port

    if host is None and port_override is not None:
        host = DEFAULT_HOST_FALLBACK

    if host:
        if port is None:
            port = 8080
        return host, port, None

    uds = listen.unix_socket or DEFAULT_SOCKET_PATH
    return None, None, uds

def _split_bind(bind: str) -> Tuple[str, int]:
    value = bind.strip()
    if not value:
        raise ValueError("Empty bind value")
    if value.startswith("["):
        host_part, port_part = value.split("]:", 1)
        host = host_part[1:]
    else:
        if ":" not in value:
            raise ValueError("Bind value missing port")
        host, port_part = value.rsplit(":", 1)
    return host, int(port_part)

def _admin_base_url(daemon_cfg: DaemonConfig) -> str:
    bindings = daemon_cfg.admin.bind
    preferred: Sequence[str] = ["127.0.0.1", "localhost", "::1"]
    for candidate in bindings:
        try:
            host, port = _split_bind(candidate)
        except ValueError:
            continue
        if host in preferred:
            if ":" in host and not host.startswith("["):
                return f"http://[{host}]:{port}"
            return f"http://{host}:{port}"
    for candidate in bindings:
        try:
            host, port = _split_bind(candidate)
        except ValueError:
            continue
        if ":" in host and not host.startswith("["):
            return f"http://[{host}]:{port}"
        return f"http://{host}:{port}"
    raise ConfigError("No usable admin.bind entries configured")

def _perform_admin_action(url: str, timeout: float) -> None:
    try:
        response = requests.post(url, timeout=timeout)
    except requests.RequestException as exc:
        print(f"[error] Admin request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    if response.status_code >= 400:
        print(
            f"[error] Admin endpoint returned {response.status_code}: {response.text}",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        payload = response.json()
    except ValueError:
        payload = None

    if isinstance(payload, dict) and "status" in payload:
        print(payload["status"])
    else:
        print(f"Request succeeded ({response.status_code})")

def _command_start(args: argparse.Namespace) -> None:
    cfg_dir = _resolve_config_dir(getattr(args, "config_dir", None))
    try:
        daemon_cfg = load_daemon_config(cfg_dir / "daemon.json")
    except ConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        host, port, uds = _determine_listen_target(args, daemon_cfg)
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(2)

    reload_enabled = bool(getattr(args, "reload", False))
    foreground = bool(getattr(args, "foreground", False))
    if reload_enabled and not foreground:
        print("[info] --reload implies foreground mode", file=sys.stderr)
        foreground = True

    log_level = daemon_cfg.logging.level.lower()

    if foreground:
        desc = f"unix:{uds}" if uds else f"http://{host}:{port}"
        print(f"mini-apigw starting in foreground on {desc}")
        _serve_uvicorn(str(cfg_dir), host, port, uds, reload_enabled, log_level)
        return

    _start_background(cfg_dir, host, port, uds, reload_enabled, log_level)

def _command_reload(args: argparse.Namespace) -> None:
    cfg_dir = _resolve_config_dir(args.config_dir)
    try:
        daemon_cfg = load_daemon_config(cfg_dir / "daemon.json")
    except ConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    base_url = args.admin_url or _admin_base_url(daemon_cfg)
    url = f"{base_url.rstrip('/')}/admin/reload"
    _perform_admin_action(url, args.timeout)

def _command_stop(args: argparse.Namespace) -> None:
    cfg_dir = _resolve_config_dir(args.config_dir)
    try:
        daemon_cfg = load_daemon_config(cfg_dir / "daemon.json")
    except ConfigError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        sys.exit(1)

    base_url = args.admin_url or _admin_base_url(daemon_cfg)
    url = f"{base_url.rstrip('/')}/admin/shutdown"
    _perform_admin_action(url, args.timeout)


def _command_token(args: argparse.Namespace) -> None:
    if args.bytes <= 0:
        print("[error] --bytes must be a positive integer", file=sys.stderr)
        sys.exit(1)
    token = secrets.token_urlsafe(args.bytes)
    print(token)

def main() -> None:
    start_parent = argparse.ArgumentParser(add_help=False)
    start_parent.add_argument("--config-dir", help="Directory containing configuration files")
    start_parent.add_argument("--host", help="Override listen host")
    start_parent.add_argument("--port", type=int, help="Override listen port")
    start_parent.add_argument("--unix-socket", help="Override Unix domain socket path")
    start_parent.add_argument("--reload", action="store_true", help="Enable auto reload (development)")
    start_parent.add_argument("--foreground", action="store_true", help="Run in the foreground")

    parser = argparse.ArgumentParser(description="mini-apigw service management", parents=[start_parent])
    subparsers = parser.add_subparsers(dest="command")

    start_parser = subparsers.add_parser("start", help="Start the gateway service", parents=[start_parent])
    start_parser.set_defaults(func=_command_start)

    reload_parser = subparsers.add_parser("reload", help="Reload gateway configuration")
    reload_parser.add_argument("--config-dir", help="Directory containing configuration files")
    reload_parser.add_argument("--admin-url", help="Override admin base URL (e.g. http://127.0.0.1:8081)")
    reload_parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds")
    reload_parser.set_defaults(func=_command_reload)

    stop_parser = subparsers.add_parser("stop", help="Request a graceful shutdown")
    stop_parser.add_argument("--config-dir", help="Directory containing configuration files")
    stop_parser.add_argument("--admin-url", help="Override admin base URL (e.g. http://127.0.0.1:8081)")
    stop_parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout in seconds")
    stop_parser.set_defaults(func=_command_stop)

    token_parser = subparsers.add_parser("token", help="Generate bearer tokens")
    token_parser.add_argument("--bytes", type=int, default=32, help="Number of random bytes (default: 32)")
    token_parser.set_defaults(func=_command_token)

    parser.set_defaults(func=_command_start)

    args = parser.parse_args()

    args.func(args)

if __name__ == "__main__":  # pragma: no cover
    main()

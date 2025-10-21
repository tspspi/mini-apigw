"""Console entry point for the gateway."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import uvicorn

from .app import create_app
from .config import ConfigError, load_daemon_config


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-apigw service")
    parser.add_argument("--config-dir", help="Directory containing daemon.json, backends.json, apps.json")
    parser.add_argument("--host", help="Override listen host")
    parser.add_argument("--port", type=int, help="Override listen port")
    parser.add_argument("--reload", action="store_true", help="Enable auto reload (development)")
    args = parser.parse_args()

    config_dir = args.config_dir or os.environ.get("MINIAPIGW_CONFIG_DIR")
    if config_dir is None:
        config_dir = str(Path.cwd() / "config")
    cfg_path = Path(config_dir)
    daemon_cfg = load_daemon_config(cfg_path / "daemon.json")

    host = args.host or daemon_cfg.listen.host_v6 or daemon_cfg.listen.host_v4
    port = args.port or daemon_cfg.listen.port

    app = create_app(cfg_path)
    uvicorn.run(app, host=host, port=port, reload=args.reload, log_level=daemon_cfg.logging.level.lower())


if __name__ == "__main__":  # pragma: no cover
    main()

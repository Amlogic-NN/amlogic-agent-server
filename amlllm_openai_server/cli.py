# -*- coding: utf-8 -*-
"""CLI for running the AMLLLM OpenAI-compatible proxy server."""

from __future__ import annotations

import argparse
from pathlib import Path
import logging

import uvicorn

from .app import create_app


def build_parser() -> argparse.ArgumentParser:
    project_dir = Path(__file__).resolve().parents[1]
    default_config = project_dir / "config" / "server.yaml"

    parser = argparse.ArgumentParser(description="Run an OpenAI-compatible proxy on top of AMLLLM.")
    parser.add_argument("--host", default=None, help="Override bind host from server.yaml.")
    parser.add_argument("--port", type=int, default=None, help="Override bind port from server.yaml.")
    parser.add_argument("--api-key", default=None, help="Override API key from server.yaml.")
    parser.add_argument(
        "--config",
        default=str(default_config),
        help="Path to the server.yaml file.",
    )
    parser.add_argument("--log-level", default=None, help="Override uvicorn log level from server.yaml.")
    return parser


def main():
    args = build_parser().parse_args()
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    app, server_settings = create_app(
        config_path=args.config,
        api_key=args.api_key,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    uvicorn.run(
        app,
        host=server_settings.host,
        port=server_settings.port,
        log_level=server_settings.log_level,
    )


if __name__ == "__main__":
    main()

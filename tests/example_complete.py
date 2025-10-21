"""Minimal CLI example for the Mini API Gateway /v1/completions endpoint."""
from __future__ import annotations

import argparse
import sys

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a text completion through the Mini API Gateway")
    parser.add_argument("--host", default="127.0.0.1", help="Gateway host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Gateway port (default: 8080)")
    parser.add_argument(
        "--scheme",
        choices=["http", "https"],
        default="http",
        help="Scheme to use when connecting to the gateway",
    )
    parser.add_argument("--api-key", required=True, help="API key for the gateway")
    parser.add_argument(
        "--model",
        default="gpt-3.5-turbo-instruct",
        help="Completion model name to request (default: gpt-3.5-turbo-instruct)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )
    return parser.parse_args()


def build_base_url(scheme: str, host: str, port: int) -> str:
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def main() -> None:
    args = parse_args()
    base_url = build_base_url(args.scheme, args.host, args.port)
    client = OpenAI(api_key=args.api_key, base_url=f"{base_url}/v1", timeout=args.timeout)

    prompt = sys.stdin.read()
    if not prompt:
        print("[error] No input received on stdin", file=sys.stderr)
        raise SystemExit(1)

    try:
        completion = client.completions.create(model=args.model, prompt=prompt)
    except Exception as exc:  # pragma: no cover - depends on runtime errors
        print(f"[error] Request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    choices = getattr(completion, "choices", None)
    if choices is None and isinstance(completion, dict):
        choices = completion.get("choices")
    if not choices:
        print("[error] Gateway returned no choices", file=sys.stderr)
        raise SystemExit(1)

    first = choices[0]
    text = getattr(first, "text", None)
    if text is None and isinstance(first, dict):
        text = first.get("text")
    if text is None:
        print("[error] Completion missing text field", file=sys.stderr)
        raise SystemExit(1)

    if text.endswith("\n"):
        sys.stdout.write(text)
    else:
        print(text)


if __name__ == "__main__":
    main()

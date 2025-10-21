"""Read text from stdin, chunk it, and request embeddings via the Mini API Gateway."""
from __future__ import annotations

import argparse
import json
import sys
from typing import List

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings through the Mini API Gateway")
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
        default="text-embedding-3-small",
        help="Embedding model name to request (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4000,
        help="Maximum characters per chunk submitted for embedding (default: 4000)",
    )
    return parser.parse_args()


def build_base_url(scheme: str, host: str, port: int) -> str:
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def _chunk_text(text: str, chunk_size: int) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    chunks: List[str] = []
    length = len(text)
    idx = 0
    while idx < length:
        end = min(length, idx + chunk_size)
        if end < length:
            newline = text.rfind("\n", idx, end)
            space = text.rfind(" ", idx, end)
            split_at = max(newline, space)
            if split_at > idx:
                end = split_at
        chunk = text[idx:end].strip()
        if chunk:
            chunks.append(chunk)
        idx = end
        while idx < length and text[idx].isspace():
            idx += 1
    return chunks or [text]


def main() -> None:
    args = parse_args()
    base_url = build_base_url(args.scheme, args.host, args.port)
    client = OpenAI(api_key=args.api_key, base_url=f"{base_url}/v1", timeout=args.timeout)

    source = sys.stdin.read()
    if not source:
        print("[error] No input received on stdin", file=sys.stderr)
        sys.exit(1)

    chunks = _chunk_text(source, args.chunk_size)
    if not chunks:
        print("[error] Input only contained whitespace", file=sys.stderr)
        sys.exit(1)

    try:
        response = client.embeddings.create(model=args.model, input=chunks)
    except Exception as exc:  # pragma: no cover - runtime dependent
        print(f"[error] Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    data = getattr(response, "data", None)
    if data is None:
        data = response.get("data") if isinstance(response, dict) else None
    if not data:
        print("[error] Gateway returned no embeddings", file=sys.stderr)
        sys.exit(1)

    for index, item in enumerate(data):
        vector = getattr(item, "embedding", None)
        if vector is None and isinstance(item, dict):
            vector = item.get("embedding")
        if vector is None:
            print(f"[warn] Missing embedding for chunk {index}", file=sys.stderr)
            continue
        print(json.dumps({"index": index, "embedding": vector}))


if __name__ == "__main__":
    main()

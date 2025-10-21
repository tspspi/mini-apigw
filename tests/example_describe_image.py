"""Minimal CLI example for describing an image via the Mini API Gateway."""
from __future__ import annotations

import argparse
import base64
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Describe an image with the Mini API Gateway")
    parser.add_argument("image", help="Path to a PNG or JPEG image to describe")
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
        default="gpt-4o-mini",
        help="Model name to request (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant. Describe the uploaded image.",
        help="Optional system prompt to seed the conversation",
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


def read_image(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type not in {"image/png", "image/jpeg"}:
        raise ValueError("Only PNG and JPEG images are supported")
    with path.open("rb") as handle:
        encoded = base64.b64encode(handle.read()).decode("ascii")
    return mime_type, encoded


def _normalize_content(content: Any) -> str:
    if isinstance(content, list):
        fragments: List[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("value")
                if isinstance(text, str):
                    fragments.append(text)
        return "".join(fragments).strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("value")
        if isinstance(text, str):
            return text.strip()
    if isinstance(content, str):
        return content.strip()
    return ""


def main() -> None:
    args = parse_args()
    base_url = build_base_url(args.scheme, args.host, args.port)
    client = OpenAI(api_key=args.api_key, base_url=f"{base_url}/v1", timeout=args.timeout)

    image_path = Path(os.path.expanduser(args.image))
    try:
        mime_type, encoded = read_image(image_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[error] {exc}")

    image_url = f"data:{mime_type};base64,{encoded}"

    messages: List[Dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Please describe this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    )

    try:
        completion = client.chat.completions.create(model=args.model, messages=messages)
    except Exception as exc:  # pragma: no cover - depends on runtime errors
        raise SystemExit(f"[error] Request failed: {exc}") from exc

    payload = getattr(completion, "model_dump", None)
    if callable(payload):
        payload = completion.model_dump()
    else:
        payload = completion

    choices = payload.get("choices") if isinstance(payload, dict) else None
    if not choices:
        raise SystemExit("[error] Gateway returned no choices")

    message = choices[0].get("message", {})
    content = _normalize_content(message.get("content"))
    if not content:
        raise SystemExit("[error] Assistant response missing content")

    print(content)


if __name__ == "__main__":
    main()

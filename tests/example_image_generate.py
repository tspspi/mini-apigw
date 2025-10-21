"""CLI example for generating images via the Mini API Gateway."""
from __future__ import annotations

import argparse
import base64
import io
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

import requests
from openai import OpenAI


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate an image with the Mini API Gateway")
    parser.add_argument("--host", default="127.0.0.1", help="Gateway host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Gateway port (default: 8080)")
    parser.add_argument(
        "--scheme",
        choices=["http", "https"],
        default="http",
        help="Scheme to use when connecting to the gateway",
    )
    parser.add_argument("--api-key", required=True, help="API key for the gateway")
    parser.add_argument("--model", required=True, help="Image model name to request")
    parser.add_argument(
        "--system-prompt",
        default="Create a whimsical, sweet fantasy illustration set in a magical world.",
        help="Optional preamble added in front of the user prompt",
    )
    parser.add_argument("--output", required=True, help="Destination image path (PNG will be written)")
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--size",
        default=None,
        help="Optional image size (e.g. 1024x1024) forwarded to the backend",
    )
    parser.add_argument(
        "--response-format",
        dest="response_format",
        default=None,
        help="Optional response_format forwarded to the backend",
    )
    return parser.parse_args()


def build_base_url(scheme: str, host: str, port: int) -> str:
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def _compose_prompt(system_prompt: str, user_prompt: str) -> str:
    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()
    if not system_prompt:
        return user_prompt
    if not user_prompt:
        return system_prompt
    return f"{system_prompt}\n\n{user_prompt}"


def _decode_base64_image(data: str, output_path: Path) -> None:
    raw = base64.b64decode(data)
    if raw.startswith(PNG_SIGNATURE):
        output_path.write_bytes(raw)
        return

    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:  # pragma: no cover - Pillow optional
        raise RuntimeError(
            "Image data is not PNG and Pillow is not available to recode it; install pillow to proceed"
        ) from exc

    with Image.open(io.BytesIO(raw)) as image:
        image.convert("RGBA").save(output_path, format="PNG")


def _download_image(url: str) -> bytes:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return response.content


def _extract_first_image(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = payload.get("data")
    if not isinstance(data, Iterable):
        raise RuntimeError("Gateway response missing image data")
    for item in data:
        if isinstance(item, dict):
            return item
    raise RuntimeError("Gateway response did not contain a usable image payload")


def main() -> None:
    args = parse_args()
    base_url = build_base_url(args.scheme, args.host, args.port)
    client = OpenAI(api_key=args.api_key, base_url=f"{base_url}/v1", timeout=args.timeout)

    prompt = sys.stdin.read()
    if not prompt:
        print("[error] No prompt received on stdin", file=sys.stderr)
        raise SystemExit(1)

    final_prompt = _compose_prompt(args.system_prompt, prompt)

    request_kwargs: Dict[str, Any] = {"model": args.model, "prompt": final_prompt}
    if args.size:
        request_kwargs["size"] = args.size
    if args.response_format:
        request_kwargs["response_format"] = args.response_format
    elif not args.model.lower().startswith("gpt-image-1"):
        # DALL·E family defaults to URLs; request base64 unless overridden.
        request_kwargs["response_format"] = "b64_json"

    try:
        response = client.images.generate(**request_kwargs)
    except Exception as exc:  # pragma: no cover - runtime errors
        print(f"[error] Request failed: {exc}", file=sys.stderr)
        raise SystemExit(1)

    payload = getattr(response, "model_dump", None)
    if callable(payload):
        payload = response.model_dump()
    else:
        payload = response

    if not isinstance(payload, dict):
        print("[error] Unexpected response payload", file=sys.stderr)
        raise SystemExit(1)

    image_entry = _extract_first_image(payload)

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    b64_data = image_entry.get("b64_json")
    if isinstance(b64_data, str) and b64_data:
        try:
            _decode_base64_image(b64_data, output_path)
        except Exception as exc:  # pragma: no cover - conversion errors depend on runtime
            print(f"[error] Failed to decode base64 image: {exc}", file=sys.stderr)
            raise SystemExit(1)
        print(f"Image saved to {output_path}")
        return

    url = image_entry.get("url")
    if isinstance(url, str) and url:
        try:
            content = _download_image(url)
        except Exception as exc:  # pragma: no cover - network errors
            print(f"[error] Failed to download image: {exc}", file=sys.stderr)
            raise SystemExit(1)

        if content.startswith(PNG_SIGNATURE):
            output_path.write_bytes(content)
        else:
            try:
                from PIL import Image  # type: ignore
            except ImportError as exc:  # pragma: no cover - Pillow optional
                print(
                    "[error] Downloaded image is not PNG and Pillow is unavailable for conversion",
                    file=sys.stderr,
                )
                raise SystemExit(1) from exc
            with Image.open(io.BytesIO(content)) as image:
                image.convert("RGBA").save(output_path, format="PNG")
        print(f"Image saved to {output_path}")
        return

    print("[error] Gateway did not return base64 data or a URL", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()

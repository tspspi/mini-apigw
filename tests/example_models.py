"""Query the Mini API Gateway /v1/models endpoint and pretty-print the response."""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, Iterable

import httpx



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List models via the Mini API Gateway")
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
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds (default: 30)",
    )
    return parser.parse_args()



def build_base_url(scheme: str, host: str, port: int) -> str:
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"



def _print_failure(message: str, *, details: Dict[str, Any] | None = None) -> None:
    payload: Dict[str, Any] = {"status": "error", "message": message}
    if details:
        payload.update(details)
    print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)



def _iter_models(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, dict):
        inner = data.get("data")
        if isinstance(inner, list):
            return (item for item in inner if isinstance(item, dict))
    if isinstance(data, list):
        return (item for item in data if isinstance(item, dict))
    return ()



def main() -> None:
    args = parse_args()
    base_url = build_base_url(args.scheme, args.host, args.port)
    url = f"{base_url}/v1/models"
    headers = {"Authorization": f"Bearer {args.api_key}"}

    try:
        response = httpx.get(url, headers=headers, timeout=args.timeout)
    except httpx.RequestError as exc:
        _print_failure("Request failed", details={"error": str(exc), "url": url})
        sys.exit(1)

    content_type = response.headers.get("content-type", "")
    is_json = "json" in content_type

    try:
        payload: Any = response.json() if is_json else response.text
    except ValueError:
        payload = response.text
        is_json = False

    if response.status_code >= 400:
        details: Dict[str, Any] = {
            "status_code": response.status_code,
            "reason": response.reason_phrase,
        }
        if is_json:
            details["response"] = payload
        else:
            details["raw_response"] = payload
        _print_failure("Gateway returned an error", details=details)
        sys.exit(1)

    if not is_json:
        _print_failure("Gateway response is not JSON", details={"raw_response": payload})
        sys.exit(1)

    models = list(_iter_models(payload))
    if not models:
        _print_failure("Gateway returned no models", details={"response": payload})
        sys.exit(1)

    summary = {
        "status": "ok",
        "count": len(models),
        "models": [
            {
                "id": model.get("id"),
                "type": model.get("type"),
                "owned_by": model.get("owned_by"),
                "created": model.get("created"),
            }
            for model in models
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

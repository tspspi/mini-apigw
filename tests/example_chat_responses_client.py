"""Interactive example client for the Mini API Gateway Responses endpoint.

Run from the project root, for example:

    python -m tests.example_chat_responses_client --api-key mykey

The client keeps conversation context using the Responses "input" format and
streams assistant output to the console. Exit with ``:q``, ``quit``, or
``exit``; type ``:reset`` to clear the current conversation.
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, List

from openai import OpenAI

EXIT_COMMANDS = {":q", "quit", "exit"}
RESET_COMMAND = ":reset"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the Responses API")
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
        "--instructions",
        default=None,
        help="Optional developer/system instructions passed to the Responses API",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream token deltas (default: disabled)",
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


def _message(role: str, text: str, *, part_type: str = "input_text") -> Dict[str, Any]:
    return {"role": role, "content": [{"type": part_type, "text": text}]}


def _extract_text_from_output(output: List[Dict[str, Any]]) -> str:
    messages = []
    for item in output:
        if item.get("type") != "message":
            continue
        for part in item.get("content", []):
            if part.get("type") == "output_text":
                messages.append(str(part.get("text", "")))
    return "".join(messages).strip()


def _collect_from_response(resp: Any) -> str:
    """Best-effort conversion of a Responses object into plain text."""
    if resp is None:
        return ""
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    elif hasattr(resp, "to_dict"):
        data = resp.to_dict()
    elif isinstance(resp, dict):
        data = resp
    else:  # pragma: no cover - defensive fallback
        return str(resp)
    output = data.get("output")
    if isinstance(output, list):
        return _extract_text_from_output(output)
    return str(data)


def main() -> None:
    args = parse_args()
    base_url = build_base_url(args.scheme, args.host, args.port)
    client = OpenAI(api_key=args.api_key, base_url=f"{base_url}/v1", timeout=args.timeout)

    conversation: List[Dict[str, Any]] = []
    print("Responses client ready. Type your message (:q to quit, :reset to clear).")

    while True:
        try:
            user_text = input("you > ").strip()
        except EOFError:  # pragma: no cover - interactive only
            print()
            break
        if not user_text:
            continue
        if user_text.lower() in EXIT_COMMANDS:
            break
        if user_text.lower() == RESET_COMMAND:
            conversation.clear()
            print("[conversation cleared]")
            continue

        user_message = _message("user", user_text, part_type="input_text")
        conversation.append(user_message)

        payload: Dict[str, Any] = {
            "model": args.model,
            "input": conversation,
        }
        if args.instructions:
            payload["instructions"] = args.instructions

        if args.stream:
            text_parts: List[str] = []
            print("assistant > ", end="", flush=True)
            final_response = None
            with client.responses.stream(**payload) as stream:
                for event in stream:
                    event_type = getattr(event, "type", None)
                    if event_type == "response.output_text.delta":
                        delta = getattr(event, "delta", "")
                        text = str(delta)
                        text_parts.append(text)
                        print(text, end="", flush=True)
                    elif event_type == "response.completed":
                        final_response = getattr(event, "response", None)
                if final_response is None:
                    final_response = stream.get_final_response()
            print()
            assistant_text = "".join(text_parts).strip()
            if not assistant_text:
                assistant_text = _collect_from_response(final_response)
        else:
            response = client.responses.create(**payload)
            assistant_text = _collect_from_response(response)
            print(f"assistant > {assistant_text}")

        if assistant_text:
            conversation.append(_message("assistant", assistant_text, part_type="output_text"))
        else:
            print("[warning] empty assistant response")


if __name__ == "__main__":
    main()

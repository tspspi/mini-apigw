"""Interactive example client for the Mini API Gateway chat endpoint.

Run from the project root, for example:

    python -m tests.example_chat_client --api-key mykey

The client keeps the conversation context, exposes simple todo list tools,
and prints responses from the gateway. Exit with ``:q``, ``quit``, or ``exit``.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from openai import OpenAI

# User can type any of these commands (case-insensitive) to end the session.
EXIT_COMMANDS = {":q", "quit", "exit"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "add_todo",
            "description": "Add a new entry to the shared todo list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "item": {
                        "type": "string",
                        "description": "Todo item description",
                    }
                },
                "required": ["item"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_todo",
            "description": "Remove a todo entry by its 1-based index.",
            "parameters": {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "integer",
                        "description": "1-based index of the todo to remove",
                        "minimum": 1,
                    }
                },
                "required": ["index"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_todos",
            "description": "Return all current todo entries with their indices.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chat with the Mini API Gateway")
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
        default=None,
        help="Optional system prompt to seed the conversation",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Request structured JSON responses (OpenAI response_format / Ollama format)",
    )
    parser.add_argument(
        "--json-schema",
        default=None,
        help="Optional JSON schema (as JSON string or @path) for structured responses",
    )
    return parser.parse_args()


def _load_json_schema(arg: str | None) -> Dict[str, Any] | None:
    if not arg:
        return None
    if arg.startswith("@"):
        path = arg[1:]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except OSError as exc:
            print(f"[error] Failed to read schema file: {exc}")
            return None
        except json.JSONDecodeError as exc:
            print(f"[error] Invalid JSON schema in file: {exc}")
            return None
    try:
        return json.loads(arg)
    except json.JSONDecodeError as exc:
        print(f"[error] Invalid JSON schema string: {exc}")
        return None


def build_base_url(scheme: str, host: str, port: int) -> str:
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


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
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        text = content.get("text") or content.get("value")
        if isinstance(text, str):
            return text.strip()
        return json.dumps(content)
    return ""


def _execute_tool_call(call: Dict[str, Any], todos: List[str]) -> str:
    function = call.get("function", {})
    name = function.get("name") or ""
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as exc:
            return f"error: invalid JSON arguments ({exc})"
    elif isinstance(arguments, dict):
        args = arguments
    else:
        args = {}

    if name == "add_todo":
        item = args.get("item")
        if not item:
            return "error: missing 'item'"
        todos.append(str(item))
        return f"added todo #{len(todos)}: {item}"

    if name == "remove_todo":
        index = args.get("index")
        if not isinstance(index, int) or index < 1 or index > len(todos):
            return "error: invalid index"
        removed = todos.pop(index - 1)
        return f"removed todo #{index}: {removed}"

    if name == "list_todos":
        if not todos:
            return "[]"
        data = [{"index": idx + 1, "item": item} for idx, item in enumerate(todos)]
        return json.dumps(data)

    return f"error: unknown tool '{name}'"


def main() -> None:
    args = parse_args()
    base_url = build_base_url(args.scheme, args.host, args.port)
    client = OpenAI(api_key=args.api_key, base_url=f"{base_url}/v1", timeout=args.timeout)

    response_format = None
    if args.json:
        schema = _load_json_schema(args.json_schema)
        if schema:
            response_format = {"type": "json_schema", "json_schema": {"name": "structured", "schema": schema}}
        else:
            response_format = {"type": "json_object"}

    messages: List[Dict[str, Any]] = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})

    print("Mini API Gateway chat client. Type ':q' to exit.")

    todos: List[str] = []

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input.lower() in EXIT_COMMANDS:
            break

        messages.append({"role": "user", "content": user_input})

        followup_attempts = 0
        while True:
            followup_attempts += 1
            if followup_attempts > 6:
                print("[error] Aborting after repeated tool interactions")
                break

            try:
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    tools=TOOLS,
                    response_format=response_format,
                )
            except Exception as exc:  # pragma: no cover - depends on runtime errors
                print(f"[error] Request failed: {exc}")
                break

            try:
                payload = completion.model_dump()
            except AttributeError:
                payload = completion

            choices = payload.get("choices") or []
            if not choices:
                print("[error] Empty response from gateway")
                break

            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls") or []

            if tool_calls:
                messages.append(
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": tool_calls,
                    }
                )
                for call in tool_calls:
                    result = _execute_tool_call(call, todos)
                    tool_name = call.get("function", {}).get("name", "<unknown>")
                    print(f"tool[{tool_name}]> {result}")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call.get("id"),
                            "name": tool_name,
                            "content": result,
                        }
                    )
                continue

            content = message.get("content")
            content = _normalize_content(content)
            if not content:
                print("[error] Assistant response missing content")
                break

            label = payload.get("model") or payload.get("provider") or "gateway"
            print(f"{label}> {content}")

            messages.append({"role": "assistant", "content": content})
            break


if __name__ == "__main__":
    main()

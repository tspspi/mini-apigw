"""Asynchronous streaming example client for the Mini API Gateway chat endpoint.

Run from the project root, for example:

    python -m tests.example_async_chat_client --api-key mykey

This variant mirrors ``tests.example_chat_client`` but leverages the
``AsyncOpenAI`` client to demonstrate streaming responses. The script keeps the
conversation context, exposes the same todo tools, and prints streamed tokens
as they arrive. Exit with ``:q``, ``quit``, or ``exit``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any, Callable, Dict, List, Optional
import sys

from openai import AsyncOpenAI

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
    parser = argparse.ArgumentParser(description="Async chat with the Mini API Gateway")
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
    parser.add_argument(
        "--debug-chunks",
        action="store_true",
        help="Print each streamed chunk payload (JSON) to stderr for debugging",
    )
    return parser.parse_args()


def _load_json_schema(arg: Optional[str]) -> Optional[Dict[str, Any]]:
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


def _normalize_content(content: Any, *, strip: bool = True) -> str:
    if isinstance(content, list):
        fragments: List[str] = []
        for item in content:
            if isinstance(item, str):
                fragments.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("value")
                if isinstance(text, str):
                    fragments.append(text)
        combined = "".join(fragments)
        return combined.strip() if strip else combined
    if isinstance(content, str):
        return content.strip() if strip else content
    if isinstance(content, dict):
        text = content.get("text") or content.get("value")
        if isinstance(text, str):
            return text.strip() if strip else text
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


def _append_tool_call_fragment(
    fragments: Dict[int, Dict[str, Any]],
    entry: Dict[str, Any],
    *,
    default_index: Optional[int] = None,
) -> None:
    index = entry.get("index")
    if not isinstance(index, int):
        if isinstance(default_index, int):
            index = default_index
        else:
            index = max(fragments.keys(), default=-1) + 1
    builder = fragments.setdefault(
        index,
        {
            "id": entry.get("id"),
            "type": entry.get("type") or "function",
            "function": {"name": "", "arguments": ""},
        },
    )
    if entry.get("id"):
        builder["id"] = entry["id"]
    entry_type = entry.get("type")
    if isinstance(entry_type, str) and entry_type:
        builder["type"] = entry_type
    elif not builder.get("type"):
        builder["type"] = "function"
    fn = entry.get("function") or {}
    function_builder = builder.setdefault("function", {"name": "", "arguments": ""})
    if fn.get("name"):
        function_builder["name"] = fn["name"]
    if fn.get("arguments"):
        args_fragment = fn["arguments"]
        if not isinstance(args_fragment, str):
            args_fragment = json.dumps(args_fragment)
        existing = function_builder.get("arguments", "")
        function_builder["arguments"] = existing + args_fragment


class _StreamAccumulator:
    def __init__(self, model: str, on_text_chunk: Optional[Callable[[str], None]]):
        self._model = model
        self._on_text_chunk = on_text_chunk
        self._content_parts: List[str] = []
        self._tool_calls: Dict[int, Dict[str, Any]] = {}
        self._final_tool_calls: Optional[List[Dict[str, Any]]] = None
        self._finish_reason: Optional[str] = None
        self._provider_model: Optional[str] = None
        self._saw_streaming_text = False
        self._last_message_content: Any = None
        self._last_tool_index: Optional[int] = None

    def feed(self, data: Dict[str, Any]) -> None:
        self._provider_model = data.get("model") or self._provider_model
        for choice in data.get("choices", []):
            delta = choice.get("delta") or {}
            message_obj = choice.get("message") or {}
            finish_reason = choice.get("finish_reason")
            if isinstance(finish_reason, str):
                self._finish_reason = finish_reason

            content_delta = delta.get("content")
            if content_delta is not None:
                text = _normalize_content(content_delta, strip=False)
                if text:
                    self._content_parts.append(text)
                    if text.strip():
                        self._saw_streaming_text = True
                    if self._on_text_chunk:
                        self._on_text_chunk(text)

            if not self._saw_streaming_text:
                fallback_content = message_obj.get("content")
                if fallback_content is not None:
                    self._last_message_content = fallback_content
                    text = _normalize_content(fallback_content, strip=False)
                    if text:
                        self._content_parts.append(text)
                        if text.strip():
                            self._saw_streaming_text = True
                        if self._on_text_chunk:
                            self._on_text_chunk(text)
            else:
                fallback_content = message_obj.get("content")
                if fallback_content is not None:
                    self._last_message_content = fallback_content

            tool_delta = delta.get("tool_calls") or []
            for entry in tool_delta:
                if isinstance(entry, dict):
                    entry_index = entry.get("index")
                    if isinstance(entry_index, int):
                        self._last_tool_index = entry_index
                        index_hint = entry_index
                    else:
                        index_hint = self._last_tool_index
                    _append_tool_call_fragment(self._tool_calls, entry, default_index=index_hint)

            message_tools = message_obj.get("tool_calls") or []
            if message_tools:
                valid_tools = [entry for entry in message_tools if isinstance(entry, dict)]
                if valid_tools:
                    self._final_tool_calls = valid_tools
                    last_index = max(
                        (item.get("index") for item in valid_tools if isinstance(item.get("index"), int)),
                        default=None,
                    )
                    if isinstance(last_index, int):
                        self._last_tool_index = last_index

    def finalize(self) -> Dict[str, Any]:
        ordered_calls = self._ordered_tool_calls()
        raw_text = "".join(self._content_parts)
        normalized_text = raw_text.strip()
        if not normalized_text and self._last_message_content is not None:
            normalized_text = _normalize_content(self._last_message_content)

        message: Dict[str, Any] = {"role": "assistant", "content": normalized_text or ""}
        if ordered_calls:
            message["tool_calls"] = ordered_calls

        finish_reason = self._finish_reason
        if finish_reason is None:
            finish_reason = "tool_calls" if ordered_calls else "stop"

        return {
            "model": self._provider_model or self._model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _ordered_tool_calls(self) -> List[Dict[str, Any]]:
        def _normalize_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
            normalized = dict(entry)
            normalized["type"] = normalized.get("type") or "function"
            fn = normalized.get("function")
            if not isinstance(fn, dict):
                normalized["function"] = {"name": "", "arguments": ""}
            else:
                fn.setdefault("arguments", "")
            return normalized

        if self._final_tool_calls is not None:
            return [_normalize_entry(entry) for entry in self._final_tool_calls if isinstance(entry, dict)]

        ordered: List[Dict[str, Any]] = []
        for idx in sorted(self._tool_calls.keys()):
            ordered.append(_normalize_entry(self._tool_calls[idx]))
        return ordered


async def _stream_chat_completion(
    client: AsyncOpenAI,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    response_format: Optional[Dict[str, Any]],
    on_text_chunk: Optional[Callable[[str], None]] = None,
    debug_chunks: bool = False,
) -> Dict[str, Any]:
    stream = await client.chat.completions.create(
        model=model,
        messages=messages,
        tools=TOOLS,
        response_format=response_format,
        stream=True,
    )

    accumulator = _StreamAccumulator(model, on_text_chunk)
    chunk_index = 0
    async for chunk in stream:
        if hasattr(chunk, "model_dump"):
            data = chunk.model_dump()
        else:
            data = chunk
        if debug_chunks:
            compact = json.dumps(data, ensure_ascii=False)
            print(f"[debug] chunk[{chunk_index}]: {compact}", file=sys.stderr)
        chunk_index += 1
        accumulator.feed(data)

    if hasattr(stream, "get_final_response"):
        final_response = await stream.get_final_response()
        if hasattr(final_response, "model_dump"):
            return final_response.model_dump()
        return final_response

    return accumulator.finalize()


async def _prompt_user(prompt: str) -> str:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, input, prompt)


async def main() -> None:
    args = parse_args()
    base_url = build_base_url(args.scheme, args.host, args.port)
    client = AsyncOpenAI(api_key=args.api_key, base_url=f"{base_url}/v1", timeout=args.timeout)

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

    todos: List[str] = []
    print("Mini API Gateway async streaming chat client. Type ':q' to exit.")

    while True:
        try:
            user_input = (await _prompt_user("you> ")).strip()
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

            printed_label = False
            label_hint = args.model

            def on_text_chunk(chunk: str) -> None:
                nonlocal printed_label
                if not printed_label:
                    print(f"{label_hint}> ", end="", flush=True)
                    printed_label = True
                print(chunk, end="", flush=True)

            try:
                payload = await _stream_chat_completion(
                    client,
                    model=args.model,
                    messages=messages,
                    response_format=response_format,
                    on_text_chunk=on_text_chunk,
                    debug_chunks=args.debug_chunks,
                )
                if printed_label:
                    print()
            except Exception as exc:  # pragma: no cover - depends on runtime errors
                print(f"[error] Request failed: {exc}")
                break

            choices = payload.get("choices") or []
            if not choices:
                print("[error] Empty response from gateway")
                break

            choice = choices[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason")
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
                followup_attempts -= 1
                continue

            content = _normalize_content(message.get("content"))
            if not content and finish_reason == "tool_calls":
                followup_attempts -= 1
                continue
            if not content:
                print("[error] Assistant response missing content")
                break

            if not printed_label:
                label = payload.get("model") or payload.get("provider") or "gateway"
                print(f"{label}> {content}")
            messages.append({"role": "assistant", "content": content})
            break


if __name__ == "__main__":
    asyncio.run(main())

"""Structured recipe generator client for the Mini API Gateway.

Usage example:

    python -m tests.example_recipie_json --api-key sk-demo \\
        --model llama3.2 < prompt.txt

Reads a single prompt from stdin and requests a JSON recipe object.
"""
from __future__ import annotations

import argparse
import json
import sys
from enum import Enum
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError


class Unit(str, Enum):
    KG = "kg"
    G = "g"
    L = "l"
    ML = "ml"
    PCS = "pcs"
    TBSP = "tbsp"
    TSP = "tsp"


class Ingredient(BaseModel):
    name: str = Field(..., description="Ingredient name")
    amount: float = Field(..., description="Quantity needed")
    unit: Unit = Field(..., description="Measurement unit")



class Recipe(BaseModel):
    title: str = Field(..., description="Recipe title")
    description: str = Field(..., description="Short overview")
    ingredients: list[Ingredient] = Field(
        ..., description="List of ingredients", min_length=1
    )
    steps: list[str] = Field(..., description="Preparation steps", min_length=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate structured recipes via the Mini API Gateway")
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
        default="You are a helpful chef that creates detailed recipes as structured JSON.",
        help="Optional system prompt",
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

    prompt = sys.stdin.read().strip()
    if not prompt:
        print("[error] No input received on stdin", file=sys.stderr)
        sys.exit(1)

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": prompt})

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "recipe",
            "schema": Recipe.model_json_schema(),
        },
    }

    try:
        completion = client.chat.completions.create(
            model=args.model,
            messages=messages,
            response_format=response_format,
        )
    except Exception as exc:  # pragma: no cover - networking/runtime errors
        print(f"[error] Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    payload = completion.model_dump()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        print("[error] Unexpected response shape", file=sys.stderr)
        sys.exit(1)

    try:
        recipe = Recipe.model_validate_json(content)
    except ValidationError as exc:
        print(f"[error] Response did not match recipe schema: {exc}\nRaw content: {content}", file=sys.stderr)
        sys.exit(1)

    json.dump(recipe.model_dump(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

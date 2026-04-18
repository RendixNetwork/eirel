from __future__ import annotations

import argparse
import json
import urllib.request
from collections.abc import Awaitable, Callable
from typing import Any

from eirel.models import ChatCompletionResponse


PUBLIC_COMPLIANCE_CASES = [
    {
        "name": "chat-basic",
        "request": {
            "messages": [{"role": "user", "content": "Say hello and mention help."}],
            "temperature": 0,
            "seed": 7,
        },
        "validator": lambda payload: "help" in (
            payload.choices[0].message.content or ""
        ).lower(),
    },
    {
        "name": "tool-call-shape",
        "request": {
            "messages": [{"role": "user", "content": "What is 2 + 2? Use tools if needed."}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "math",
                        "parameters": {
                            "type": "object",
                            "properties": {"expression": {"type": "string"}},
                            "required": ["expression"],
                        },
                    },
                }
            ],
            "tool_choice": "auto",
            "temperature": 0,
            "seed": 7,
        },
        "validator": lambda payload: bool(payload.choices[0].message.tool_calls),
    },
]


def run_public_compliance_suite(
    send_request: Callable[[dict[str, Any]], dict[str, Any]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in PUBLIC_COMPLIANCE_CASES:
        response_payload = send_request(case["request"])
        response = ChatCompletionResponse.model_validate(response_payload)
        passed = bool(case["validator"](response))
        results.append({"name": case["name"], "passed": passed})
    return results


async def run_public_compliance_suite_async(
    send_request: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in PUBLIC_COMPLIANCE_CASES:
        response_payload = await send_request(case["request"])
        response = ChatCompletionResponse.model_validate(response_payload)
        passed = bool(case["validator"](response))
        results.append({"name": case["name"], "passed": passed})
    return results


def _http_sender(base_url: str, path: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    endpoint = f"{base_url.rstrip('/')}{path}"

    def send_request(payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode())

    return send_request


def configure_parser(sp: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = sp.add_parser("compliance", help="Run pre-flight compliance checks against a running miner")
    parser.add_argument("--base-url", required=True, help="Running miner base URL")
    parser.add_argument("--path", default="/v1/chat/completions", help="Chat completions endpoint path")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> None:
    results = run_public_compliance_suite(_http_sender(args.base_url, args.path))
    failed = [item for item in results if not item["passed"]]
    print(json.dumps({"results": results}, indent=2))
    raise SystemExit(1 if failed else 0)

from __future__ import annotations

import argparse
from typing import Any

import uvicorn

from eirel.app import MinerApp
from eirel.helpers import build_tool_call, content_response, tool_call_response


async def _handle(payload: dict[str, Any]) -> dict[str, Any]:
    user_text = " ".join(
        str(message.get("content", ""))
        for message in payload.get("messages", [])
        if message.get("role") == "user"
    ).lower()
    tool_names = {
        item.get("function", {}).get("name")
        for item in payload.get("tools") or []
        if isinstance(item, dict)
    }
    if "calculator" in tool_names and ("19 * 7" in user_text or "2 + 2" in user_text):
        return tool_call_response(
            [
                build_tool_call(
                    tool_id="call-calculator-1",
                    name="calculator",
                    arguments={"expression": "19 * 7" if "19 * 7" in user_text else "2 + 2"},
                )
            ]
        ).model_dump(mode="json")
    return content_response("Hello, I can help with questions.").model_dump(mode="json")


def create_app():
    return MinerApp(title="Eirel Miner Sample Server", handler=_handle).fastapi_app()


def configure_parser(sp: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = sp.add_parser("sample", help="Run the bundled reference miner sample server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> None:
    uvicorn.run(create_app(), host=args.host, port=args.port)

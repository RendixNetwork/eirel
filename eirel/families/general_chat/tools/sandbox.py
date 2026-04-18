from __future__ import annotations

from typing import Any

from eirel.families.general_chat.tools import GeneralChatTool
from eirel.families.general_chat.tools._service_client import ToolServiceClient


class SandboxTool(GeneralChatTool):
    """Owner-api-routed Python sandbox.

    Executes short Python snippets server-side and returns the captured
    stdout / stderr / exit code / duration. Intended for verifiable
    computation — math, date arithmetic, statistics, unit conversions,
    parsing, regex, JSON/CSV processing — not for long-running scripts
    or anything that needs network / filesystem access.

    The server-side sandbox blocks: network imports (socket, urllib,
    httpx, requests, ssl), subprocess creation, ctypes, dangerous os
    attributes (system, popen, exec*, kill, chmod, chown, rename).
    Resource limits: 5s default wall clock, 128 MB default memory,
    64 KB max code size.

    Miners should use this tool whenever the response contains a number
    derived from arithmetic — the trace integrity gate rewards verified
    computation via the recorded stdout, same way it rewards verified
    URL citations via the recorded tool call.
    """

    def __init__(self, client: ToolServiceClient) -> None:
        self._client = client

    @property
    def name(self) -> str:
        return "sandbox"

    @property
    def description(self) -> str:
        return (
            "Execute a Python snippet in a server-side sandbox. Returns "
            "stdout, stderr, exit code, and duration. Use for math, "
            "date arithmetic, statistics, unit conversions, parsing, "
            "regex, JSON/CSV processing. Network / subprocess / file "
            "I/O are blocked. 5s default wall clock, 128 MB memory."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python code to execute. Must use print() to "
                        "produce output (no implicit return). Example: "
                        "'print(round((47000/30000)**(1/5) - 1, 4))'."
                    ),
                    "maxLength": 65536,
                },
                "timeout_seconds": {
                    "type": "number",
                    "minimum": 0.1,
                    "maximum": 30.0,
                    "description": "Wall clock limit in seconds (default 5).",
                },
                "memory_mb": {
                    "type": "integer",
                    "minimum": 16,
                    "maximum": 1024,
                    "description": "Memory limit in MB (default 128).",
                },
            },
            "required": ["code"],
        }

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        code = str(kwargs["code"])
        payload: dict[str, Any] = {"code": code}
        timeout = kwargs.get("timeout_seconds")
        if isinstance(timeout, (int, float)) and timeout > 0:
            payload["timeout_seconds"] = float(timeout)
        memory = kwargs.get("memory_mb")
        if isinstance(memory, int) and memory > 0:
            payload["memory_mb"] = memory
        return await self._client.request(
            path="/v1/tools/sandbox",
            payload=payload,
        )


__all__ = ["SandboxTool"]

from __future__ import annotations

import secrets
from typing import Any

from eirel.families.general_chat.tools import GeneralChatTool
from eirel.families.general_chat.tools._service_client import ToolServiceClient


class SandboxTool(GeneralChatTool):
    """Owner-api-routed Python sandbox.

    Executes short Python snippets server-side and returns the captured
    stdout / stderr / exit code / duration. Intended for verifiable
    computation — math, date arithmetic, statistics, unit conversions,
    parsing, regex, JSON/CSV processing — not for long-running scripts
    or anything that needs network access.

    The server-side sandbox blocks: network imports (socket, urllib,
    httpx, requests, ssl), subprocess creation, ctypes, dangerous os
    attributes (system, popen, exec*, kill, chmod, chown, rename).
    Resource limits: 5s default wall clock, 128 MB default memory,
    64 KB max code size.

    **Sessions and files.** Pass the same ``session_id`` across calls to
    share kernel state (variables, imports, working directory) between
    runs — useful for multi-step analysis where call 2 builds on call
    1. Pass ``attachments`` to mount pre-existing files (typically the
    user's uploaded CSVs/PDFs) into the sandbox working directory.
    Files written during execution come back in the ``files`` field.
    Sessions expire after 5 minutes idle; a fresh ``session_id``
    starts a clean kernel.

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
            "stdout, stderr, exit code, duration, and any files written. "
            "Use for math, date arithmetic, statistics, unit conversions, "
            "parsing, regex, JSON/CSV processing, and analyzing user-"
            "uploaded files (pass them via attachments). Reuse session_id "
            "to keep variables / imports across calls. Network and "
            "subprocess are blocked. 5s default wall clock, 128 MB memory."
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
                "session_id": {
                    "type": "string",
                    "maxLength": 128,
                    "description": (
                        "Optional. Reuse the same session_id across "
                        "calls to share kernel state (variables, "
                        "imports, working directory) between runs. "
                        "Sessions expire after 5 minutes idle; a fresh "
                        "session_id starts a clean kernel."
                    ),
                },
                "attachments": {
                    "type": "array",
                    "maxItems": 16,
                    "description": (
                        "Optional pre-existing files to mount into the "
                        "sandbox working directory before execution. "
                        "Each item is {filename, content_b64} where "
                        "content_b64 is base64-encoded raw bytes."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "filename": {
                                "type": "string",
                                "maxLength": 255,
                            },
                            "content_b64": {"type": "string"},
                        },
                        "required": ["filename", "content_b64"],
                    },
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
        session_id = kwargs.get("session_id")
        if isinstance(session_id, str) and session_id.strip():
            payload["session_id"] = session_id.strip()
        attachments = kwargs.get("attachments")
        if isinstance(attachments, list) and attachments:
            cleaned: list[dict[str, str]] = []
            for att in attachments:
                if not isinstance(att, dict):
                    continue
                filename = att.get("filename")
                content_b64 = att.get("content_b64")
                if isinstance(filename, str) and isinstance(content_b64, str):
                    cleaned.append(
                        {"filename": filename, "content_b64": content_b64}
                    )
            if cleaned:
                payload["attachments"] = cleaned
        return await self._client.request(
            path="/v1/tools/sandbox",
            payload=payload,
        )


class SandboxSession:
    """Convenience wrapper that threads a fixed ``session_id`` across calls.

    Use when an agent wants kernel state to persist across several
    ``execute()`` calls in the same turn (or across turns, when the
    orchestrator passes the session_id back). Call ``.execute(code)``
    instead of ``tool.execute(code=..., session_id=...)``.

    The session_id is auto-allocated if not provided; access via
    ``.session_id`` to surface it back to the caller (e.g. for
    metadata or follow-up turns).
    """

    def __init__(
        self,
        tool: SandboxTool,
        *,
        session_id: str | None = None,
    ) -> None:
        self._tool = tool
        self._session_id = (
            session_id.strip() if isinstance(session_id, str) and session_id.strip()
            else secrets.token_urlsafe(12)
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    async def execute(
        self,
        code: str,
        *,
        timeout_seconds: float | None = None,
        memory_mb: int | None = None,
        attachments: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "code": code,
            "session_id": self._session_id,
        }
        if timeout_seconds is not None:
            kwargs["timeout_seconds"] = timeout_seconds
        if memory_mb is not None:
            kwargs["memory_mb"] = memory_mb
        if attachments:
            kwargs["attachments"] = attachments
        return await self._tool.execute(**kwargs)


__all__ = ["SandboxSession", "SandboxTool"]

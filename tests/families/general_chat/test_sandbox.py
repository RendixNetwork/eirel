from __future__ import annotations

import json

import httpx

from eirel.families.general_chat.tools._service_client import (
    ToolServiceClient,
    ToolServiceConfig,
)
from eirel.families.general_chat.tools.sandbox import SandboxTool


def _client(handler) -> ToolServiceClient:
    return ToolServiceClient(
        ToolServiceConfig(base_url="http://owner-api.local", api_token="tkn"),
        transport=httpx.MockTransport(handler),
    )


async def test_sandbox_calls_owner_api_endpoint_with_payload():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "stdout": "4\n",
                "stderr": "",
                "exit_code": 0,
                "duration_ms": 12,
            },
        )

    tool = SandboxTool(_client(handler))
    result = await tool.execute(code="print(2 + 2)")
    assert result["stdout"] == "4\n"
    assert result["exit_code"] == 0
    assert captured["url"] == "http://owner-api.local/v1/tools/sandbox"
    assert captured["body"] == {"code": "print(2 + 2)"}


async def test_sandbox_forwards_optional_limits():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    tool = SandboxTool(_client(handler))
    await tool.execute(code="print(1)", timeout_seconds=10.0, memory_mb=256)
    assert captured["body"] == {
        "code": "print(1)",
        "timeout_seconds": 10.0,
        "memory_mb": 256,
    }


async def test_sandbox_ignores_zero_or_negative_limits():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    tool = SandboxTool(_client(handler))
    await tool.execute(code="print(1)", timeout_seconds=0, memory_mb=-1)
    # Invalid limits are dropped so the server uses its defaults.
    assert captured["body"] == {"code": "print(1)"}


def test_sandbox_metadata():
    tool = SandboxTool(_client(lambda r: httpx.Response(200, json={})))
    assert tool.name == "sandbox"
    assert "sandbox" in tool.description.lower()
    schema = tool.parameters_schema
    assert "code" in schema["properties"]
    assert "timeout_seconds" in schema["properties"]
    assert "memory_mb" in schema["properties"]
    assert schema["required"] == ["code"]

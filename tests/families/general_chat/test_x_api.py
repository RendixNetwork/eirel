from __future__ import annotations

import json

import httpx

from eirel.families.general_chat.tools._service_client import (
    ToolServiceClient,
    ToolServiceConfig,
)
from eirel.families.general_chat.tools.x_api import XApiTool


def _client(handler) -> ToolServiceClient:
    return ToolServiceClient(
        ToolServiceConfig(base_url="http://owner-api.local", api_token="tkn"),
        transport=httpx.MockTransport(handler),
    )


async def test_x_api_calls_owner_api_endpoint_with_payload():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"posts": []})

    tool = XApiTool(_client(handler))
    result = await tool.execute(query="bittensor", max_results=5)
    assert result == {"posts": []}
    assert captured["url"] == "http://owner-api.local/v1/tools/x_api"
    assert captured["body"] == {"query": "bittensor", "max_results": 5}


async def test_x_api_clamps_max_results():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    tool = XApiTool(_client(handler))
    await tool.execute(query="x", max_results=9999)
    assert captured["body"] == {"query": "x", "max_results": 50}


def test_x_api_metadata():
    tool = XApiTool(_client(lambda r: httpx.Response(200, json={})))
    assert tool.name == "x_api"
    assert "1 call per task" in tool.description.lower() or "1 call" in tool.description.lower()

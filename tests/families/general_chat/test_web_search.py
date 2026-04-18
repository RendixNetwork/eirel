from __future__ import annotations

import json

import httpx

from eirel.families.general_chat.tools._service_client import (
    ToolServiceClient,
    ToolServiceConfig,
)
from eirel.families.general_chat.tools.web_search import WebSearchTool


def _client(handler) -> ToolServiceClient:
    transport = httpx.MockTransport(handler)
    return ToolServiceClient(
        ToolServiceConfig(base_url="http://owner-api.local", api_token="tkn", job_id="j-1"),
        transport=transport,
    )


async def test_web_search_calls_owner_api_endpoint_with_payload():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["method"] = request.method
        captured["body"] = json.loads(request.content.decode())
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"results": [{"url": "https://a.com"}]})

    tool = WebSearchTool(_client(handler))
    result = await tool.execute(query="nvidia revenue", max_results=4)

    assert result == {"results": [{"url": "https://a.com"}]}
    assert captured["url"] == "http://owner-api.local/v1/search"
    assert captured["method"] == "POST"
    assert captured["body"] == {"query": "nvidia revenue", "top_k": 4}
    headers = captured["headers"]  # type: ignore[index]
    assert headers["authorization"] == "Bearer tkn"
    assert headers["x-eirel-job-id"] == "j-1"


async def test_web_search_default_max_results():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    tool = WebSearchTool(_client(handler))
    await tool.execute(query="hello")
    assert captured["body"] == {"query": "hello", "top_k": 5}


async def test_web_search_clamps_max_results():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    tool = WebSearchTool(_client(handler))
    await tool.execute(query="x", max_results=999)
    assert captured["body"] == {"query": "x", "top_k": 20}


def test_web_search_schema_shape():
    tool = WebSearchTool(_client(lambda r: httpx.Response(200, json={})))
    schema = tool.parameters_schema
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert schema["required"] == ["query"]


def test_web_search_metadata():
    tool = WebSearchTool(_client(lambda r: httpx.Response(200, json={})))
    assert tool.name == "web_search"
    assert "web" in tool.description.lower()

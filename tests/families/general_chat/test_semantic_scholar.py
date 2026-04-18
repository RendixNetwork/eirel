from __future__ import annotations

import json

import httpx

from eirel.families.general_chat.tools._service_client import (
    ToolServiceClient,
    ToolServiceConfig,
)
from eirel.families.general_chat.tools.semantic_scholar import SemanticScholarTool


def _client(handler) -> ToolServiceClient:
    return ToolServiceClient(
        ToolServiceConfig(base_url="http://owner-api.local", api_token="tkn"),
        transport=httpx.MockTransport(handler),
    )


async def test_semantic_scholar_calls_owner_api_endpoint_with_payload():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={"papers": []})

    tool = SemanticScholarTool(_client(handler))
    result = await tool.execute(query="transformer scaling", max_results=3)
    assert result == {"papers": []}
    assert captured["url"] == "http://owner-api.local/v1/tools/semantic_scholar"
    assert captured["body"] == {"query": "transformer scaling", "max_results": 3}


async def test_semantic_scholar_default_max_results():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    tool = SemanticScholarTool(_client(handler))
    await tool.execute(query="LLM")
    assert captured["body"] == {"query": "LLM", "max_results": 5}


async def test_semantic_scholar_forwards_optional_filters():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    tool = SemanticScholarTool(_client(handler))
    await tool.execute(
        query="retrieval augmented generation",
        max_results=10,
        year="2023-",
        fields_of_study=["Computer Science"],
        open_access_only=True,
    )
    assert captured["body"] == {
        "query": "retrieval augmented generation",
        "max_results": 10,
        "year": "2023-",
        "fields_of_study": ["Computer Science"],
        "open_access_only": True,
    }


async def test_semantic_scholar_max_results_clamped_to_100():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content.decode())
        return httpx.Response(200, json={})

    tool = SemanticScholarTool(_client(handler))
    await tool.execute(query="x", max_results=9999)
    assert captured["body"]["max_results"] == 100


def test_semantic_scholar_metadata():
    tool = SemanticScholarTool(_client(lambda r: httpx.Response(200, json={})))
    assert tool.name == "semantic_scholar"
    assert "semantic scholar" in tool.description.lower()
    schema = tool.parameters_schema
    assert "query" in schema["properties"]
    assert schema["properties"]["max_results"]["maximum"] == 100

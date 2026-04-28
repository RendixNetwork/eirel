from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from eirel import (
    AgentCapabilityMetadata,
    AgentInvocationRequest,
    AgentInvocationResponse,
    BaseAgent,
    build_agent_app,
)
from eirel.schemas import StreamChunk


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("EIREL_DISABLE_REQUEST_AUTH", "1")


class _NonStreamingAgent(BaseAgent):
    """Override only `infer()` — exercises the default `infer_stream()` fallback."""

    async def infer(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        # Slim 0.3.0 shape: executed tool calls live in metadata, not at
        # the response top level.
        return AgentInvocationResponse(
            task_id=request.task_id,
            family_id=request.family_id,
            output={"answer": f"hello {request.subtask}"},
            citations=["https://example.com/a"],
            metadata={
                "handled": True,
                "executed_tool_calls": [{"name": "search", "args": {"q": "x"}}],
            },
        )


class _StreamingAgent(BaseAgent):
    """Real streaming impl that yields multiple deltas before done."""

    async def infer(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        # Should not be hit by stream tests — but BaseAgent requires it.
        return AgentInvocationResponse(
            task_id=request.task_id, family_id=request.family_id,
        )

    async def infer_stream(  # type: ignore[override]
        self, request: AgentInvocationRequest,
    ) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(event="delta", text="part-one ")
        yield StreamChunk(event="delta", text="part-two")
        yield StreamChunk(
            event="citation",
            citation={"url": "https://example.com/x", "title": "X"},
        )
        yield StreamChunk(
            event="done",
            output={"answer": "part-one part-two"},
            citations=["https://example.com/x"],
            status="completed",
        )


def _make_request_payload() -> dict:
    return {
        "task_id": "t-1",
        "primary_goal": "say hi",
        "subtask": "world",
        "family_id": "general_chat",
    }


def _agent_app(agent_cls: type[BaseAgent]):
    agent = agent_cls(
        hotkey="hk",
        endpoint="http://127.0.0.1:9000",
        version="1.0.0",
        capabilities=AgentCapabilityMetadata(
            family_id="general_chat", description="test agent",
        ),
    )
    return build_agent_app(agent)


def _parse_ndjson(body: bytes) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def test_default_infer_stream_emits_single_delta_then_done():
    """Agents that only implement infer() get a working stream by default."""
    client = TestClient(_agent_app(_NonStreamingAgent))
    resp = client.post("/v1/agent/infer/stream", json=_make_request_payload())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    chunks = _parse_ndjson(resp.content)
    assert chunks[0]["event"] == "delta"
    assert chunks[0]["text"] == "hello world"
    assert chunks[-1]["event"] == "done"
    assert chunks[-1]["status"] == "completed"
    # Citations on the done chunk; executed tool calls live in metadata.
    assert chunks[-1]["citations"] == ["https://example.com/a"]
    assert chunks[-1]["metadata"]["executed_tool_calls"][0]["name"] == "search"


def test_streaming_agent_yields_multiple_deltas_in_order():
    client = TestClient(_agent_app(_StreamingAgent))
    resp = client.post("/v1/agent/infer/stream", json=_make_request_payload())
    assert resp.status_code == 200

    chunks = _parse_ndjson(resp.content)
    events = [c["event"] for c in chunks]
    assert events == ["delta", "delta", "citation", "done"]

    deltas = [c["text"] for c in chunks if c["event"] == "delta"]
    assert "".join(deltas) == "part-one part-two"

    assert chunks[2]["citation"] == {"url": "https://example.com/x", "title": "X"}
    assert chunks[3]["status"] == "completed"


def test_infer_stream_reports_validation_error_as_400():
    client = TestClient(_agent_app(_NonStreamingAgent))
    resp = client.post("/v1/agent/infer/stream", json={"task_id": "t-1"})
    assert resp.status_code == 400


def test_non_stream_endpoint_still_works():
    """Adding the streaming route must not regress the existing endpoint."""
    client = TestClient(_agent_app(_NonStreamingAgent))
    resp = client.post("/v1/agent/infer", json=_make_request_payload())
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "completed"
    assert body["output"] == {"answer": "hello world"}

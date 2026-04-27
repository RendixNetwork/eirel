"""Tests for MinerApp's /v1/agent/infer/stream route added in 0.2.4."""
from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from eirel.app import MinerApp
from eirel.schemas import StreamChunk


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("EIREL_DISABLE_REQUEST_AUTH", "1")


def _agent_request_payload() -> dict:
    return {
        "task_id": "t-1",
        "primary_goal": "say hi",
        "subtask": "world",
        "family_id": "general_chat",
    }


def _parse_ndjson(body: bytes) -> list[dict]:
    return [json.loads(line) for line in body.splitlines() if line.strip()]


async def _passthrough_handler(payload: dict) -> dict:
    return {
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "hi"}}
        ]
    }


async def _agent_handler(payload: dict) -> dict:
    return {
        "task_id": payload["task_id"],
        "family_id": payload["family_id"],
        "status": "completed",
        "output": {"answer": "hello world"},
        "citations": ["https://example.com/x"],
        "tool_calls": [{"name": "search"}],
    }


def test_default_stream_falls_back_to_unary_handler():
    """No agent_stream_handler → MinerApp emits one big delta + done."""
    app = MinerApp(
        title="test", handler=_passthrough_handler, agent_handler=_agent_handler,
    ).fastapi_app()
    client = TestClient(app)

    resp = client.post("/v1/agent/infer/stream", json=_agent_request_payload())
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-ndjson")

    chunks = _parse_ndjson(resp.content)
    events = [c["event"] for c in chunks]
    assert events == ["delta", "done"]
    assert chunks[0]["text"] == "hello world"
    assert chunks[1]["status"] == "completed"
    assert chunks[1]["citations"] == ["https://example.com/x"]


def test_real_stream_handler_yields_multiple_deltas():
    async def _stream(payload: dict) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(event="delta", text="part-")
        yield StreamChunk(event="delta", text="one")
        yield StreamChunk(
            event="done",
            output={"answer": "part-one"},
            citations=[],
            tool_calls=[],
            status="completed",
        )

    app = MinerApp(
        title="test",
        handler=_passthrough_handler,
        agent_handler=_agent_handler,
        agent_stream_handler=_stream,
    ).fastapi_app()
    client = TestClient(app)

    resp = client.post("/v1/agent/infer/stream", json=_agent_request_payload())
    assert resp.status_code == 200
    chunks = _parse_ndjson(resp.content)
    events = [c["event"] for c in chunks]
    assert events == ["delta", "delta", "done"]
    deltas = "".join(c["text"] for c in chunks if c["event"] == "delta")
    assert deltas == "part-one"


def test_stream_handler_exception_emits_failed_done():
    async def _stream(payload: dict) -> AsyncIterator[StreamChunk]:
        yield StreamChunk(event="delta", text="boom")
        raise RuntimeError("upstream LLM 500")

    app = MinerApp(
        title="test",
        handler=_passthrough_handler,
        agent_handler=_agent_handler,
        agent_stream_handler=_stream,
    ).fastapi_app()
    client = TestClient(app)

    resp = client.post("/v1/agent/infer/stream", json=_agent_request_payload())
    assert resp.status_code == 200
    chunks = _parse_ndjson(resp.content)
    assert chunks[0]["event"] == "delta"
    assert chunks[-1]["event"] == "done"
    assert chunks[-1]["status"] == "failed"
    assert "upstream LLM 500" in chunks[-1]["error"]


def test_stream_handler_accepts_dict_chunks():
    """Plain dicts are coerced to StreamChunk (matches build_agent_app)."""
    async def _stream(payload: dict):
        yield {"event": "delta", "text": "ok"}
        yield {"event": "done", "status": "completed"}

    app = MinerApp(
        title="test",
        handler=_passthrough_handler,
        agent_handler=_agent_handler,
        agent_stream_handler=_stream,
    ).fastapi_app()
    client = TestClient(app)

    resp = client.post("/v1/agent/infer/stream", json=_agent_request_payload())
    assert resp.status_code == 200
    chunks = _parse_ndjson(resp.content)
    assert [c["event"] for c in chunks] == ["delta", "done"]


def test_unary_endpoint_still_works():
    app = MinerApp(
        title="test", handler=_passthrough_handler, agent_handler=_agent_handler,
    ).fastapi_app()
    client = TestClient(app)
    resp = client.post("/v1/agent/infer", json=_agent_request_payload())
    assert resp.status_code == 200
    assert resp.json()["output"]["answer"] == "hello world"


def test_stream_validation_error_returns_400():
    app = MinerApp(
        title="test", handler=_passthrough_handler, agent_handler=_agent_handler,
    ).fastapi_app()
    client = TestClient(app)
    resp = client.post("/v1/agent/infer/stream", json={"task_id": "t-1"})
    assert resp.status_code == 400

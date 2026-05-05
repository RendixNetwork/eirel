from __future__ import annotations

"""Tests for Item 8: A2A protocol models and converters."""

import pytest

from eirel.a2a import (
    AgentCard,
    A2ATaskRequest,
    A2ATaskResponse,
    a2a_request_from_invocation,
    invocation_response_from_a2a,
)
from eirel.schemas import AgentInvocationRequest, AgentInvocationResponse


def _invocation_request() -> AgentInvocationRequest:
    return AgentInvocationRequest(
        task_id="t1",
        session_id="s1",
        prompt="Find recent papers on AI agents",
        family_id="general_chat",
        episode_id="e1",
    )


# ── A2A Models ──────────────────────────────────────────────────────────────


def test_agent_card_creation():
    card = AgentCard(name="TestAgent", url="http://agent.example.com")
    assert card.name == "TestAgent"
    assert card.version == "1.0"


def test_a2a_task_request():
    req = A2ATaskRequest(id="t1", message={"role": "user", "parts": [{"type": "text", "text": "Hello"}]})
    assert req.id == "t1"
    assert req.message["parts"][0]["text"] == "Hello"


def test_a2a_task_response_defaults():
    resp = A2ATaskResponse(id="t1")
    assert resp.status == "submitted"
    assert resp.result == {}


def test_a2a_task_response_completed():
    resp = A2ATaskResponse(id="t1", status="completed", result={"answer": "42"})
    assert resp.status == "completed"


# ── Converters ──────────────────────────────────────────────────────────────


def test_a2a_request_from_invocation():
    request = _invocation_request()
    a2a_req = a2a_request_from_invocation(request)
    assert a2a_req.id == "t1"
    assert a2a_req.message["parts"][0]["text"] == "Find recent papers on AI agents"
    assert a2a_req.metadata["family_id"] == "general_chat"


def test_a2a_request_uses_prompt():
    request = AgentInvocationRequest(
        task_id="t2",
        prompt="Research AI trends",
        family_id="general_chat",
    )
    a2a_req = a2a_request_from_invocation(request)
    assert a2a_req.message["parts"][0]["text"] == "Research AI trends"


def test_invocation_response_from_a2a_completed():
    a2a_resp = A2ATaskResponse(
        id="t1",
        status="completed",
        result={"summary": "AI is evolving rapidly"},
    )
    response = invocation_response_from_a2a(a2a_resp, task_id="t1", family_id="general_chat")
    assert response.status == "completed"
    assert response.output["summary"] == "AI is evolving rapidly"
    assert response.metadata["a2a_task_id"] == "t1"


def test_invocation_response_from_a2a_failed():
    a2a_resp = A2ATaskResponse(id="t1", status="failed")
    response = invocation_response_from_a2a(a2a_resp, task_id="t1", family_id="general_chat")
    assert response.status == "failed"


def test_invocation_response_from_a2a_working():
    a2a_resp = A2ATaskResponse(id="t1", status="working")
    response = invocation_response_from_a2a(a2a_resp, task_id="t1", family_id="general_chat")
    assert response.status == "deferred"


def test_invocation_response_from_a2a_canceled():
    a2a_resp = A2ATaskResponse(id="t1", status="canceled")
    response = invocation_response_from_a2a(a2a_resp, task_id="t1", family_id="general_chat")
    assert response.status == "failed"


def test_invocation_response_extracts_artifact_text():
    a2a_resp = A2ATaskResponse(
        id="t1",
        status="completed",
        result={},
        artifacts=[{"parts": [{"type": "text", "text": "Extracted content"}]}],
    )
    response = invocation_response_from_a2a(a2a_resp, task_id="t1", family_id="general_chat")
    assert response.output.get("summary") == "Extracted content"


def test_round_trip_preserves_task_id():
    request = _invocation_request()
    a2a_req = a2a_request_from_invocation(request)
    a2a_resp = A2ATaskResponse(id=a2a_req.id, status="completed", result={"done": True})
    response = invocation_response_from_a2a(a2a_resp, task_id=request.task_id, family_id=request.family_id)
    assert response.task_id == request.task_id
    assert response.family_id == request.family_id

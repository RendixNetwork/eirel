from __future__ import annotations

"""Tests for Item 4: MCP tool support in the agent invocation protocol."""

import pytest

from eirel.models import ToolDefinition, ToolFunctionDefinition
from eirel.schemas import AgentInvocationRequest


def _tool_def(name: str = "retrieval_search", desc: str = "Search retrieval index") -> ToolDefinition:
    return ToolDefinition(
        type="function",
        function=ToolFunctionDefinition(
            name=name,
            description=desc,
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        ),
    )


# ── Schema field existence ───────────────────────────────────────────────────


def test_request_has_tools_field():
    """AgentInvocationRequest should have a 'tools' field."""
    assert "tools" in AgentInvocationRequest.model_fields


def test_request_has_tool_choice_field():
    assert "tool_choice" in AgentInvocationRequest.model_fields


# ── Backward compatibility ──────────────────────────────────────────────────


def test_tools_defaults_to_none():
    """Old miners that don't send tools should still work."""
    req = AgentInvocationRequest(
        task_id="t1",
        primary_goal="test",
        subtask="test",
        family_id="general_chat",
    )
    assert req.tools is None
    assert req.tool_choice is None


def test_tools_none_serializes_cleanly():
    req = AgentInvocationRequest(
        task_id="t1",
        primary_goal="test",
        subtask="test",
        family_id="general_chat",
    )
    data = req.model_dump(mode="json")
    assert data["tools"] is None


# ── Serialization with tools ────────────────────────────────────────────────


def test_request_with_tools_serializes():
    tools = [_tool_def("retrieval_search"), _tool_def("browser_open", "Open a web page")]
    req = AgentInvocationRequest(
        task_id="t1",
        primary_goal="test",
        subtask="test",
        family_id="general_chat",
        tools=tools,
        tool_choice="auto",
    )
    data = req.model_dump(mode="json")
    assert len(data["tools"]) == 2
    assert data["tools"][0]["function"]["name"] == "retrieval_search"
    assert data["tool_choice"] == "auto"


def test_round_trip_serialization():
    tools = [_tool_def()]
    req = AgentInvocationRequest(
        task_id="t1",
        primary_goal="test",
        subtask="test",
        family_id="general_chat",
        tools=tools,
        tool_choice={"type": "function", "function": {"name": "retrieval_search"}},
    )
    data = req.model_dump(mode="json")
    restored = AgentInvocationRequest.model_validate(data)
    assert len(restored.tools) == 1
    assert restored.tools[0].function.name == "retrieval_search"
    assert restored.tool_choice == {"type": "function", "function": {"name": "retrieval_search"}}


def test_empty_tools_list():
    req = AgentInvocationRequest(
        task_id="t1",
        primary_goal="test",
        subtask="test",
        family_id="general_chat",
        tools=[],
    )
    assert req.tools == []


# ── chat_payload_from_agent_request integration ─────────────────────────────


def test_chat_payload_includes_tools_when_present():
    from eirel.helpers import chat_payload_from_agent_request

    tools = [_tool_def()]
    req = AgentInvocationRequest(
        task_id="t1",
        primary_goal="Research aspirin",
        subtask="Find studies",
        family_id="general_chat",
        tools=tools,
        tool_choice="auto",
    )
    payload = chat_payload_from_agent_request(req)
    assert "tools" in payload
    assert len(payload["tools"]) == 1
    assert payload["tools"][0]["function"]["name"] == "retrieval_search"
    assert payload["tool_choice"] == "auto"


def test_chat_payload_omits_tools_when_none():
    from eirel.helpers import chat_payload_from_agent_request

    req = AgentInvocationRequest(
        task_id="t1",
        primary_goal="test",
        subtask="test",
        family_id="general_chat",
    )
    payload = chat_payload_from_agent_request(req)
    assert "tools" not in payload
    assert "tool_choice" not in payload


# ── ToolDefinition validation ────────────────────────────────────────────────


def test_tool_definition_requires_function_type():
    with pytest.raises(Exception):
        ToolDefinition(type="invalid", function=ToolFunctionDefinition(name="x"))


def test_tool_definition_requires_name():
    with pytest.raises(Exception):
        ToolDefinition(type="function", function=ToolFunctionDefinition())

from __future__ import annotations

import pytest
from pydantic import ValidationError

from eirel.families.general_chat.response import (
    Citation,
    ConversationResponse,
    GeneralChatResponse,
    ToolCall,
    TraceRecorder,
)


def test_citation_round_trip():
    c = Citation(url="https://example.com/a", snippet="hello", tool_name="web_search")
    data = c.model_dump(mode="json")
    restored = Citation.model_validate(data)
    assert restored.url == "https://example.com/a"
    assert restored.snippet == "hello"
    assert restored.tool_name == "web_search"


def test_citation_optional_snippet():
    c = Citation(url="https://example.com/a", tool_name="web_search")
    assert c.snippet is None


def test_citation_requires_tool_name():
    with pytest.raises(ValidationError):
        Citation(url="https://x.com")  # type: ignore[call-arg]


def test_tool_call_round_trip():
    tc = ToolCall(
        tool_name="web_search",
        args={"query": "nvidia"},
        result_digest="abc",
        latency_ms=125,
    )
    data = tc.model_dump(mode="json")
    restored = ToolCall.model_validate(data)
    assert restored.tool_name == "web_search"
    assert restored.args == {"query": "nvidia"}
    assert restored.result_digest == "abc"
    assert restored.latency_ms == 125


def test_tool_call_defaults():
    tc = ToolCall(tool_name="sandbox")
    assert tc.args == {}
    assert tc.result_digest is None
    assert tc.latency_ms == 0


def test_tool_call_negative_latency_rejected():
    with pytest.raises(ValidationError):
        ToolCall(tool_name="sandbox", latency_ms=-1)


def test_general_chat_response_round_trip():
    resp = GeneralChatResponse(
        content="hello there",
        citations=[
            Citation(url="https://example.com", tool_name="web_search"),
        ],
        tool_calls=[ToolCall(tool_name="web_search", args={"q": "x"})],
        metadata={"mode": "instant"},
    )
    data = resp.model_dump(mode="json")
    restored = GeneralChatResponse.model_validate(data)
    assert restored.content == "hello there"
    assert len(restored.citations) == 1
    assert len(restored.tool_calls) == 1
    assert restored.metadata == {"mode": "instant"}


def test_general_chat_response_minimal():
    resp = GeneralChatResponse(content="hi")
    assert resp.citations == []
    assert resp.tool_calls == []
    assert resp.metadata == {}


def test_conversation_response_round_trip():
    cr = ConversationResponse(
        turns=[
            GeneralChatResponse(content="hi"),
            GeneralChatResponse(content="bye"),
        ]
    )
    data = cr.model_dump(mode="json")
    restored = ConversationResponse.model_validate(data)
    assert len(restored.turns) == 2
    assert restored.turns[0].content == "hi"


# -- TraceRecorder ------------------------------------------------------------


def test_trace_recorder_freeze_empty():
    rec = TraceRecorder()
    response = rec.freeze()
    assert response.content == ""
    assert response.citations == []
    assert response.tool_calls == []


def test_trace_recorder_records_content_and_tool_calls():
    rec = TraceRecorder()
    rec.add_content("Hello, ")
    rec.add_content("world.")
    rec.record_tool_call(ToolCall(tool_name="web_search", args={"q": "x"}))
    rec.record_citation(
        Citation(url="https://a.com", snippet="abc", tool_name="web_search")
    )
    rec.set_metadata("mode", "instant")
    rec.set_metadata("web_search", True)

    response = rec.freeze()
    assert response.content == "Hello, world."
    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "web_search"
    assert len(response.citations) == 1
    assert response.metadata == {"mode": "instant", "web_search": True}


def test_trace_recorder_freeze_returns_independent_copy():
    rec = TraceRecorder()
    rec.add_content("hi")
    response = rec.freeze()
    rec.add_content("more")
    rec.record_tool_call(ToolCall(tool_name="sandbox"))
    # The previously frozen response is unaffected by later mutations.
    assert response.content == "hi"
    assert response.tool_calls == []


def test_trace_recorder_output_token_accounting():
    rec = TraceRecorder()
    rec.record_output_tokens(10)
    rec.record_output_tokens(5)
    assert rec.output_tokens == 15


def test_trace_recorder_negative_tokens_rejected():
    rec = TraceRecorder()
    with pytest.raises(ValueError):
        rec.record_output_tokens(-1)

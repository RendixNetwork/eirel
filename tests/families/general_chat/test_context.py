from __future__ import annotations

import pytest

from eirel.families.general_chat.budget import (
    INSTANT_BUDGET,
    INSTANT_WEB_SEARCH_BUDGET,
    THINKING_BUDGET,
    THINKING_WEB_SEARCH_BUDGET,
)
from eirel.families.general_chat.context import (
    ConversationTurn,
    GeneralChatContext,
    context_from_request,
)


def test_conversation_turn_basic():
    turn = ConversationTurn(role="user", content="hello")
    assert turn.role == "user"
    assert turn.content == "hello"
    assert turn.metadata is None


def test_conversation_turn_with_metadata():
    turn = ConversationTurn(role="assistant", content="hi", metadata={"k": "v"})
    assert turn.metadata == {"k": "v"}


def test_conversation_turn_invalid_role():
    with pytest.raises(Exception):
        ConversationTurn(role="system", content="x")  # type: ignore[arg-type]


def test_context_from_request_instant_default():
    payload = {"task_id": "t1"}
    ctx = context_from_request(payload)
    assert ctx.mode == "instant"
    assert ctx.web_search_enabled is False
    assert ctx.budget is INSTANT_BUDGET
    assert ctx.conversation_id == "t1"
    assert ctx.hotkey is None
    assert ctx.conversation_history == ()


def test_context_from_request_instant_web_search():
    payload = {
        "task_id": "t1",
        "inputs": {"mode": "instant", "web_search": True},
    }
    ctx = context_from_request(payload)
    assert ctx.mode == "instant"
    assert ctx.web_search_enabled is True
    assert ctx.budget is INSTANT_WEB_SEARCH_BUDGET


def test_context_from_request_thinking():
    payload = {"task_id": "t2", "inputs": {"mode": "thinking"}}
    ctx = context_from_request(payload)
    assert ctx.mode == "thinking"
    assert ctx.budget is THINKING_BUDGET


def test_context_from_request_thinking_web_search():
    payload = {
        "task_id": "t3",
        "inputs": {"mode": "thinking", "web_search": True},
    }
    ctx = context_from_request(payload)
    assert ctx.mode == "thinking"
    assert ctx.web_search_enabled is True
    assert ctx.budget is THINKING_WEB_SEARCH_BUDGET


def test_context_from_request_invalid_mode():
    payload = {"task_id": "t1", "inputs": {"mode": "deep_research"}}
    with pytest.raises(ValueError, match="unsupported mode"):
        context_from_request(payload)


def test_context_from_request_history_from_messages():
    payload = {
        "task_id": "t1",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello there"},
            {"role": "user", "content": "tell me more"},
        ],
    }
    ctx = context_from_request(payload)
    assert len(ctx.conversation_history) == 3
    assert ctx.conversation_history[0].role == "user"
    assert ctx.conversation_history[1].role == "assistant"
    assert ctx.conversation_history[2].content == "tell me more"


def test_context_from_request_history_skips_invalid_roles():
    payload = {
        "task_id": "t1",
        "messages": [
            {"role": "system", "content": "ignored"},
            {"role": "user", "content": "kept"},
        ],
    }
    ctx = context_from_request(payload)
    assert len(ctx.conversation_history) == 1
    assert ctx.conversation_history[0].content == "kept"


def test_context_from_request_history_via_inputs():
    payload = {
        "task_id": "t1",
        "inputs": {
            "conversation_history": [
                {"role": "user", "content": "x"},
            ],
        },
    }
    ctx = context_from_request(payload)
    assert len(ctx.conversation_history) == 1


def test_context_from_request_session_id_fallback():
    payload = {"session_id": "s-123"}
    ctx = context_from_request(payload)
    assert ctx.conversation_id == "s-123"


def test_context_from_request_hotkey():
    payload = {"task_id": "t1", "hotkey": "5HotKey"}
    ctx = context_from_request(payload)
    assert ctx.hotkey == "5HotKey"


def test_context_is_frozen():
    ctx = GeneralChatContext(
        hotkey=None,
        conversation_id="c1",
        mode="instant",
        web_search_enabled=False,
        budget=INSTANT_BUDGET,
        conversation_history=(),
    )
    with pytest.raises(AttributeError):
        ctx.mode = "thinking"  # type: ignore[misc]


def test_context_history_is_tuple():
    ctx = GeneralChatContext(
        hotkey=None,
        conversation_id="c1",
        mode="instant",
        web_search_enabled=False,
        budget=INSTANT_BUDGET,
        conversation_history=(ConversationTurn(role="user", content="x"),),
    )
    assert isinstance(ctx.conversation_history, tuple)

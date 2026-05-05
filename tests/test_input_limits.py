from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from eirel import models as eirel_models
from eirel import schemas as eirel_schemas


# ── per-message content size cap ────────────────────────────────────────────


def test_message_content_within_limit_is_accepted():
    payload = {"role": "user", "content": "hello"}
    eirel_models.Message.model_validate(payload)


def test_message_content_exceeding_limit_is_rejected(monkeypatch):
    monkeypatch.setattr(eirel_models, "MAX_MESSAGE_CONTENT_BYTES", 64)
    with pytest.raises(ValidationError, match="exceeds"):
        eirel_models.Message.model_validate(
            {"role": "user", "content": "x" * 128}
        )


def test_message_utf8_byte_counting(monkeypatch):
    monkeypatch.setattr(eirel_models, "MAX_MESSAGE_CONTENT_BYTES", 4)
    # "あ" is 3 UTF-8 bytes; two of them exceed the 4-byte cap.
    with pytest.raises(ValidationError, match="exceeds"):
        eirel_models.Message.model_validate({"role": "user", "content": "ああ"})


# ── max messages count ─────────────────────────────────────────────────────


def test_chat_completion_request_rejects_too_many_messages(monkeypatch):
    monkeypatch.setattr(eirel_models, "MAX_MESSAGES", 3)
    with pytest.raises(ValidationError, match="EIREL_MAX_MESSAGES"):
        eirel_models.ChatCompletionRequest.model_validate(
            {
                "messages": [
                    {"role": "user", "content": "m1"},
                    {"role": "assistant", "content": "m2"},
                    {"role": "user", "content": "m3"},
                    {"role": "assistant", "content": "m4"},
                ]
            }
        )


def test_chat_completion_request_accepts_within_limit():
    eirel_models.ChatCompletionRequest.model_validate(
        {"messages": [{"role": "user", "content": "hi"}]}
    )


# ── metadata depth ──────────────────────────────────────────────────────────


def _base_request(**extras) -> dict:
    base = {
        "task_id": "t1",
        "prompt": "s",
        "family_id": "general_chat",
    }
    base.update(extras)
    return base


def test_metadata_depth_within_limit_accepted():
    eirel_schemas.AgentInvocationRequest.model_validate(
        _base_request(metadata={"a": {"b": {"c": 1}}})
    )


def test_metadata_depth_exceeding_limit_rejected(monkeypatch):
    monkeypatch.setattr(eirel_schemas, "MAX_METADATA_DEPTH", 3)
    # 4 levels of nesting under metadata → too deep.
    deep = {"l1": {"l2": {"l3": {"l4": "x"}}}}
    with pytest.raises(ValidationError, match="metadata"):
        eirel_schemas.AgentInvocationRequest.model_validate(
            _base_request(metadata=deep)
        )


def test_inputs_depth_also_enforced(monkeypatch):
    monkeypatch.setattr(eirel_schemas, "MAX_METADATA_DEPTH", 2)
    deep = {"a": {"b": {"c": 1}}}
    with pytest.raises(ValidationError, match="inputs"):
        eirel_schemas.AgentInvocationRequest.model_validate(
            _base_request(inputs=deep)
        )


def test_lists_count_as_depth(monkeypatch):
    monkeypatch.setattr(eirel_schemas, "MAX_METADATA_DEPTH", 2)
    with pytest.raises(ValidationError, match="metadata"):
        eirel_schemas.AgentInvocationRequest.model_validate(
            _base_request(metadata={"a": [[[{"deep": 1}]]]})
        )


# ── context_history message content cap ────────────────────────────────────


def test_context_message_content_cap(monkeypatch):
    monkeypatch.setattr(eirel_models, "MAX_MESSAGE_CONTENT_BYTES", 8)
    # Reimport schemas so it picks up the patched constant via module-level import.
    with pytest.raises(ValidationError, match="history content exceeds"):
        eirel_schemas.ContextMessage.model_validate(
            {"role": "user", "content": "x" * 32}
        )

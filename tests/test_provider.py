from __future__ import annotations

import json

import pytest

from eirel.provider import AgentProviderClient, MinerProviderConfig


def _config(**overrides) -> MinerProviderConfig:
    defaults = {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "mode": "direct",
        "api_key": "test-key",
    }
    return MinerProviderConfig(**{**defaults, **overrides})


# ── _anthropic_to_openai conversion ──────────────────────────────────────────


def test_anthropic_to_openai_text_blocks():
    client = AgentProviderClient(_config(provider="anthropic"))
    payload = {
        "content": [
            {"type": "text", "text": "Hello "},
            {"type": "text", "text": "world"},
        ],
        "stop_reason": "end_turn",
    }
    result = client._anthropic_to_openai(payload)
    assert result["choices"][0]["message"]["content"] == "Hello world"
    assert result["choices"][0]["finish_reason"] == "end_turn"
    assert "tool_calls" not in result["choices"][0]["message"]


def test_anthropic_to_openai_tool_use_blocks():
    client = AgentProviderClient(_config(provider="anthropic"))
    payload = {
        "content": [
            {"type": "text", "text": "I will search for that."},
            {
                "type": "tool_use",
                "id": "call_123",
                "name": "retrieval_search",
                "input": {"query": "EU AI Act"},
            },
        ],
        "stop_reason": "tool_use",
    }
    result = client._anthropic_to_openai(payload)
    message = result["choices"][0]["message"]
    assert message["content"] == "I will search for that."
    assert len(message["tool_calls"]) == 1
    tool_call = message["tool_calls"][0]
    assert tool_call["id"] == "call_123"
    assert tool_call["type"] == "function"
    assert tool_call["function"]["name"] == "retrieval_search"
    assert json.loads(tool_call["function"]["arguments"]) == {"query": "EU AI Act"}


def test_anthropic_to_openai_empty_content():
    client = AgentProviderClient(_config(provider="anthropic"))
    payload = {"content": [], "stop_reason": "stop"}
    result = client._anthropic_to_openai(payload)
    assert result["choices"][0]["message"]["content"] == ""
    assert "tool_calls" not in result["choices"][0]["message"]


def test_anthropic_to_openai_non_dict_content_items_ignored():
    client = AgentProviderClient(_config(provider="anthropic"))
    payload = {"content": ["stray string", 42, {"type": "text", "text": "real"}]}
    result = client._anthropic_to_openai(payload)
    assert result["choices"][0]["message"]["content"] == "real"


# ── _openai_to_anthropic conversion ──────────────────────────────────────────


def test_openai_to_anthropic_extracts_system():
    client = AgentProviderClient(_config(provider="anthropic"))
    payload = {
        "messages": [
            {"role": "system", "content": "You are a researcher."},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 256,
    }
    result = client._openai_to_anthropic(payload)
    assert result["system"] == "You are a researcher."
    assert len(result["messages"]) == 1
    assert result["messages"][0]["role"] == "user"


# ── Token signing key rotation ───────────────────────────────────────────────


def test_token_signing_rotation():
    from eirel.token_signing import sign_resume_token, verify_resume_token

    old_secret = "old-secret"
    new_secret = "new-secret"
    token = sign_resume_token("payload-data", old_secret)

    # Verify with list of secrets (new first, old second)
    result = verify_resume_token(token, [new_secret, old_secret])
    assert result == "payload-data"


def test_token_signing_rotation_rejects_unknown_secret():
    from eirel.token_signing import sign_resume_token, verify_resume_token, InvalidResumeToken

    token = sign_resume_token("payload-data", "known-secret")
    with pytest.raises(InvalidResumeToken):
        verify_resume_token(token, ["wrong-secret-1", "wrong-secret-2"])

"""Tests for StructuredOutputNode."""
from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel

from eirel.structured import StructuredOutputError, StructuredOutputNode


class _Person(BaseModel):
    name: str
    age: int


class _StubProvider:
    """Mimics AgentProviderClient.chat_completions; returns a queue of replies."""

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if not self._replies:
            raise AssertionError("StubProvider exhausted")
        text = self._replies.pop(0)
        return {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {},
        }


async def test_structured_output_parses_clean_json():
    provider = _StubProvider([
        '{"name": "Alice", "age": 30}',
    ])
    node = StructuredOutputNode(provider, _Person)()
    update = await node({"messages": [{"role": "user", "content": "describe Alice"}]})
    assert update["structured_output"] == {"name": "Alice", "age": 30}
    assert len(provider.calls) == 1


async def test_structured_output_strips_markdown_fences():
    provider = _StubProvider([
        '```json\n{"name": "Bob", "age": 22}\n```',
    ])
    node = StructuredOutputNode(provider, _Person)()
    update = await node({"messages": []})
    assert update["structured_output"] == {"name": "Bob", "age": 22}


async def test_structured_output_extracts_object_from_prose():
    provider = _StubProvider([
        'Sure! Here is the JSON:\n{"name": "Carol", "age": 41}\nLet me know if anything else.',
    ])
    node = StructuredOutputNode(provider, _Person)()
    update = await node({"messages": []})
    assert update["structured_output"]["name"] == "Carol"


async def test_structured_output_retries_on_validation_failure():
    provider = _StubProvider([
        '{"name": "Dora"}',  # missing age — fails validation
        '{"name": "Dora", "age": 27}',
    ])
    node = StructuredOutputNode(provider, _Person, max_attempts=2)()
    update = await node({"messages": []})
    assert update["structured_output"] == {"name": "Dora", "age": 27}
    assert len(provider.calls) == 2
    # Retry payload includes the validation error inline so the model can self-correct.
    second_system = provider.calls[1]["messages"][0]
    assert "previous reply failed validation" in second_system["content"]


async def test_structured_output_raises_after_exhausting_retries():
    provider = _StubProvider([
        "not json at all",
        "still not json",
        "definitely not json",
    ])
    node = StructuredOutputNode(provider, _Person, max_attempts=3)()
    with pytest.raises(StructuredOutputError) as excinfo:
        await node({"messages": []})
    assert excinfo.value.attempts == 3
    assert excinfo.value.last_error
    assert excinfo.value.last_payload == "definitely not json"


def test_structured_output_rejects_zero_attempts():
    with pytest.raises(ValueError):
        StructuredOutputNode(_StubProvider([]), _Person, max_attempts=0)

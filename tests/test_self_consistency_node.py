"""Tests for SelfConsistencyNode."""
from __future__ import annotations

import asyncio
from typing import Any

from eirel.families.general_chat.response import TraceRecorder
from eirel.graph.patterns import SelfConsistencyNode, majority_vote
from eirel.graph.runtime import RunConfig, RunContext, _RUN_CONTEXT


class _StubProvider:
    """Returns a queue of canned reply texts, in order."""

    def __init__(self, replies: list[str]) -> None:
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


def test_majority_vote_picks_most_common():
    assert majority_vote(["yes", "no", "yes"]) == "yes"
    assert majority_vote(["YES", "yes  ", " yes"]) == "YES"  # first-seen original kept
    assert majority_vote([]) == ""


def test_majority_vote_first_seen_wins_on_tie():
    assert majority_vote(["a", "b"]) == "a"


async def test_self_consistency_picks_majority_answer():
    # Three samples — two say "42", one says "41". Majority wins.
    provider = _StubProvider(["42", "41", "42"])
    node = SelfConsistencyNode(provider, n=3)()
    update = await node({"messages": [{"role": "user", "content": "what is 6*7?"}]})
    msg = update["messages"]
    assert msg["role"] == "assistant"
    assert msg["content"] == "42"
    assert len(provider.calls) == 3


async def test_self_consistency_uses_temperature_in_payload():
    provider = _StubProvider(["x", "x", "x"])
    node = SelfConsistencyNode(provider, n=3, temperature=0.9)()
    await node({"messages": []})
    assert all(c["temperature"] == 0.9 for c in provider.calls)


async def test_self_consistency_passes_per_sample_seeds():
    provider = _StubProvider(["a", "b", "c"])
    node = SelfConsistencyNode(provider, n=3, seeds=[10, 20, 30])()
    await node({"messages": []})
    assert [c["seed"] for c in provider.calls] == [10, 20, 30]


async def test_self_consistency_raises_when_seeds_length_mismatches_n():
    provider = _StubProvider([])
    try:
        SelfConsistencyNode(provider, n=3, seeds=[1, 2])
    except ValueError as exc:
        assert "length" in str(exc)
    else:
        raise AssertionError("expected ValueError on seed length mismatch")


async def test_self_consistency_runs_samples_in_parallel():
    """All samples should fire concurrently — total time ≈ one call's worth."""

    class _SlowProvider:
        def __init__(self) -> None:
            self.calls = 0

        async def chat_completions(self, payload):
            self.calls += 1
            await asyncio.sleep(0.05)
            return {"choices": [{"message": {"content": "ok"}}]}

    provider = _SlowProvider()
    node = SelfConsistencyNode(provider, n=5)()
    import time

    t0 = time.monotonic()
    await node({"messages": []})
    elapsed = time.monotonic() - t0
    # 5×0.05s sequential = 0.25s; parallel should land ≪ 0.15s.
    assert elapsed < 0.15, f"samples did not run in parallel: {elapsed:.3f}s"
    assert provider.calls == 5


async def test_self_consistency_records_substeps_in_trace():
    provider = _StubProvider(["yes", "yes", "no"])
    trace = TraceRecorder()
    ctx = RunContext(config=RunConfig(), trace=trace)
    token = _RUN_CONTEXT.set(ctx)
    try:
        node = SelfConsistencyNode(provider, n=3, name="sc_test")()
        await node({"messages": []})
    finally:
        _RUN_CONTEXT.reset(token)
    tool_names = [tc.tool_name for tc in trace.tool_calls]
    assert "sc_test:sample_0" in tool_names
    assert "sc_test:sample_1" in tool_names
    assert "sc_test:sample_2" in tool_names
    assert "sc_test:aggregate" in tool_names


async def test_self_consistency_custom_aggregator_runs():
    provider = _StubProvider(["a", "bb", "ccc"])
    node = SelfConsistencyNode(
        provider, n=3, aggregator=lambda samples: max(samples, key=len)
    )()
    update = await node({"messages": []})
    assert update["messages"]["content"] == "ccc"

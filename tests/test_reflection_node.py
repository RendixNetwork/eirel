"""Tests for ReflectionNode."""
from __future__ import annotations

from typing import Any

from eirel.families.general_chat.response import TraceRecorder
from eirel.graph.patterns import ReflectionNode
from eirel.graph.runtime import RunConfig, RunContext, _RUN_CONTEXT


class _StubProvider:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(payload)
        if not self._replies:
            raise AssertionError("StubProvider exhausted")
        text = self._replies.pop(0)
        return {"choices": [{"message": {"content": text}}]}


async def test_reflection_stops_on_no_change():
    """Critic returns NO_CHANGE on first iteration → only 2 calls (gen + crit)."""
    gen = _StubProvider([
        "Paris is the capital of France.",  # generate
    ])
    crit = _StubProvider([
        "NO_CHANGE",  # critique 0 → stop
    ])
    node = ReflectionNode(gen, critic_provider=crit, max_iterations=2)()
    update = await node({"messages": [{"role": "user", "content": "capital of France?"}]})
    assert update["messages"]["content"] == "Paris is the capital of France."
    assert len(gen.calls) == 1
    assert len(crit.calls) == 1


async def test_reflection_revises_when_critic_finds_issues():
    """First critique non-NO_CHANGE → revise; second critique NO_CHANGE → stop."""
    gen = _StubProvider([
        "Paris.",  # generate
        "Paris is the capital of France.",  # revise 0
    ])
    crit = _StubProvider([
        "Be more complete — give a full sentence.",  # critique 0
        "NO_CHANGE",  # critique 1 (after revise)
    ])
    node = ReflectionNode(gen, critic_provider=crit, max_iterations=2)()
    update = await node({"messages": [{"role": "user", "content": "capital of France?"}]})
    assert update["messages"]["content"] == "Paris is the capital of France."
    assert len(gen.calls) == 2  # generate + revise
    assert len(crit.calls) == 2  # critique twice


async def test_reflection_respects_max_iterations_cap():
    """Critic always disagrees → cap fires after max_iterations rounds."""
    gen = _StubProvider([
        "draft 0",
        "revision 1",
        "revision 2",
    ])
    crit = _StubProvider([
        "needs work",
        "still needs work",
        # Third critique would have fired but max_iterations=2 — stops here.
    ])
    node = ReflectionNode(gen, critic_provider=crit, max_iterations=2)()
    update = await node({"messages": []})
    assert update["messages"]["content"] == "revision 2"
    assert len(gen.calls) == 3  # 1 generate + 2 revisions
    assert len(crit.calls) == 2  # 2 critiques


async def test_reflection_zero_iterations_skips_loop():
    gen = _StubProvider(["just the draft"])
    crit = _StubProvider([])  # never called
    node = ReflectionNode(gen, critic_provider=crit, max_iterations=0)()
    update = await node({"messages": []})
    assert update["messages"]["content"] == "just the draft"
    assert len(gen.calls) == 1
    assert len(crit.calls) == 0


async def test_reflection_records_substeps_in_trace():
    gen = _StubProvider(["draft", "revised"])
    crit = _StubProvider(["fix it", "NO_CHANGE"])
    trace = TraceRecorder()
    ctx = RunContext(config=RunConfig(), trace=trace)
    token = _RUN_CONTEXT.set(ctx)
    try:
        node = ReflectionNode(gen, critic_provider=crit, max_iterations=2, name="ref")()
        await node({"messages": []})
    finally:
        _RUN_CONTEXT.reset(token)
    names = [tc.tool_name for tc in trace.tool_calls]
    assert "ref:generate" in names
    assert "ref:critique_0" in names
    assert "ref:revise_0" in names
    assert "ref:critique_1" in names
    assert "ref:converged" in names


async def test_reflection_default_critic_is_generator():
    """When critic_provider is omitted, the generator is reused."""
    gen = _StubProvider(["draft", "revised"])
    # Sneak NO_CHANGE in as the second reply so the loop terminates after revise.
    # Actually: generate → critique → revise → critique → stop.
    # Reusing the same provider means the queue pulls in interleaved order.
    gen_for_both = _StubProvider(["draft", "fix it", "revised", "NO_CHANGE"])
    node = ReflectionNode(gen_for_both, max_iterations=2)()
    update = await node({"messages": []})
    assert update["messages"]["content"] == "revised"
    assert len(gen_for_both.calls) == 4

"""Tests for ReActNode."""
from __future__ import annotations

from typing import Any

from eirel.families.general_chat.budget import INSTANT_BUDGET, BudgetTracker
from eirel.families.general_chat.response import TraceRecorder
from eirel.families.general_chat.tools import GeneralChatTool, GeneralChatToolCatalog
from eirel.graph.patterns import ReActNode


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


class _StaticTool(GeneralChatTool):
    def __init__(self, name: str, result: dict[str, Any]) -> None:
        self._name = name
        self._result = result
        self.received_args: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "static result"

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "additionalProperties": True}

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.received_args.append(dict(kwargs))
        return self._result


def _catalog(tools: list[GeneralChatTool]) -> GeneralChatToolCatalog:
    return GeneralChatToolCatalog(
        tools, budget=BudgetTracker(budget=INSTANT_BUDGET), trace=TraceRecorder()
    )


async def test_react_terminates_on_final_answer():
    provider = _StubProvider([
        "Thought: I have it.\nAction: final_answer\nAction Input: \"42\"\n",
    ])
    catalog = _catalog([])
    node = ReActNode(provider, catalog, max_steps=4)()
    update = await node({"messages": [{"role": "user", "content": "what is 6*7?"}]})
    assert update["messages"]["content"] == "42"
    assert len(provider.calls) == 1


async def test_react_dispatches_tool_then_finalizes():
    tool = _StaticTool("search", {"hits": ["paris", "lyon"]})
    catalog = _catalog([tool])
    provider = _StubProvider([
        # Step 0: call search.
        'Thought: need to look it up.\nAction: search\nAction Input: {"q": "cities"}\n',
        # Step 1: terminate.
        'Thought: got it.\nAction: final_answer\nAction Input: "Paris and Lyon"\n',
    ])
    node = ReActNode(provider, catalog, max_steps=4)()
    update = await node({"messages": [{"role": "user", "content": "list cities"}]})
    assert update["messages"]["content"] == "Paris and Lyon"
    assert len(tool.received_args) == 1
    assert tool.received_args[0]["q"] == "cities"


async def test_react_observation_gets_appended_to_scratchpad():
    """The second LLM call should see the prior assistant turn + observation."""
    tool = _StaticTool("ping", {"echo": "pong"})
    catalog = _catalog([tool])
    provider = _StubProvider([
        'Thought: ping it.\nAction: ping\nAction Input: {}\n',
        'Thought: done.\nAction: final_answer\nAction Input: "ok"\n',
    ])
    node = ReActNode(provider, catalog, max_steps=4)()
    await node({"messages": [{"role": "user", "content": "ping please"}]})
    # Second call's payload should include the observation as a user msg
    # AFTER the original user prompt. Skip the original prompt by matching
    # the "Observation:" prefix.
    second = provider.calls[1]["messages"]
    obs_messages = [
        m for m in second
        if m["role"] == "user" and m.get("content", "").startswith("Observation:")
    ]
    assert obs_messages, "no Observation message in second LLM call"
    assert "pong" in obs_messages[0]["content"]


async def test_react_handles_tool_failure_gracefully():
    class _BoomTool(GeneralChatTool):
        @property
        def name(self):
            return "boom"

        @property
        def description(self):
            return "fails"

        @property
        def parameters_schema(self):
            return {"type": "object", "additionalProperties": True}

        async def execute(self, **kwargs):
            raise RuntimeError("kaboom")

    catalog = _catalog([_BoomTool()])
    provider = _StubProvider([
        'Thought: try.\nAction: boom\nAction Input: {}\n',
        'Thought: it failed.\nAction: final_answer\nAction Input: "I could not complete the task."\n',
    ])
    node = ReActNode(provider, catalog, max_steps=4)()
    update = await node({"messages": []})
    assert update["messages"]["content"] == "I could not complete the task."
    # Second call should see the error in the Observation user message.
    second = provider.calls[1]["messages"]
    obs = next(
        m for m in second
        if m["role"] == "user" and m.get("content", "").startswith("Observation:")
    )
    assert "kaboom" in obs["content"]


async def test_react_step_cap_returns_last_thought():
    provider = _StubProvider([
        'Thought: step 1.\nAction: search\nAction Input: {"q": "a"}\n',
        'Thought: step 2.\nAction: search\nAction Input: {"q": "b"}\n',
    ])
    catalog = _catalog([_StaticTool("search", {"hits": []})])
    node = ReActNode(provider, catalog, max_steps=2)()
    update = await node({"messages": []})
    # Cap fires after 2 steps; last thought surfaces as fallback.
    assert "step 2" in update["messages"]["content"]


async def test_react_reprompts_on_missing_action_label():
    """Malformed completion → node nudges with a correction message."""
    provider = _StubProvider([
        "Thought: rambling without an Action label",  # malformed
        'Thought: now formatted.\nAction: final_answer\nAction Input: "ok"\n',
    ])
    node = ReActNode(provider, _catalog([]), max_steps=4)()
    update = await node({"messages": []})
    assert update["messages"]["content"] == "ok"
    # Second call should include the correction nudge.
    second = provider.calls[1]["messages"]
    nudges = [m for m in second if "missing 'Action:'" in m.get("content", "")]
    assert nudges, "expected a correction nudge after malformed step"


async def test_react_records_substeps_in_trace():
    from eirel.graph.runtime import RunConfig, RunContext, _RUN_CONTEXT

    provider = _StubProvider([
        'Thought: t.\nAction: ping\nAction Input: {}\n',
        'Thought: done.\nAction: final_answer\nAction Input: "ok"\n',
    ])
    catalog = _catalog([_StaticTool("ping", {"v": 1})])
    trace = TraceRecorder()
    ctx = RunContext(config=RunConfig(), trace=trace)
    token = _RUN_CONTEXT.set(ctx)
    try:
        node = ReActNode(provider, catalog, max_steps=4, name="rx")()
        await node({"messages": []})
    finally:
        _RUN_CONTEXT.reset(token)
    names = [tc.tool_name for tc in trace.tool_calls]
    assert "rx:step_0" in names
    assert "rx:observation_0" in names
    assert "rx:step_1" in names

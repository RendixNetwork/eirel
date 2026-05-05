"""Tests for PlannerExecutorNode."""
from __future__ import annotations

import json
from typing import Any

from eirel.families.general_chat.budget import INSTANT_BUDGET, BudgetTracker
from eirel.families.general_chat.response import TraceRecorder
from eirel.families.general_chat.tools import GeneralChatTool, GeneralChatToolCatalog
from eirel.graph.patterns import PlannerExecutorNode


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


class _EchoTool(GeneralChatTool):
    def __init__(self, name: str, *, fail: bool = False) -> None:
        self._name = name
        self._fail = fail

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"echoes {self._name} back"

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": True}

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        if self._fail:
            raise RuntimeError(f"{self._name} blew up")
        return {"echoed": kwargs, "tool": self._name}


def _catalog(tools: list[GeneralChatTool]) -> GeneralChatToolCatalog:
    return GeneralChatToolCatalog(
        tools, budget=BudgetTracker(budget=INSTANT_BUDGET), trace=TraceRecorder()
    )


async def test_planner_executor_runs_two_step_plan():
    plan = json.dumps([
        {"id": "a", "tool": "search", "args": {"q": "weather paris"}},
        {"id": "b", "tool": "lookup", "args": {"ref": "$a"}, "depends_on": ["a"]},
    ])
    planner = _StubProvider([plan])
    executor = _StubProvider(["The weather in Paris is mild."])
    catalog = _catalog([_EchoTool("search"), _EchoTool("lookup")])
    node = PlannerExecutorNode(planner, executor, catalog)()
    update = await node({"messages": [{"role": "user", "content": "weather?"}]})
    assert update["messages"]["content"] == "The weather in Paris is mild."
    # Planner called once, executor (synthesis) called once.
    assert len(planner.calls) == 1
    assert len(executor.calls) == 1


async def test_planner_executor_substitutes_dollar_refs():
    """Step b's args should receive step a's result via $a."""
    plan = json.dumps([
        {"id": "a", "tool": "search", "args": {"q": "x"}},
        {"id": "b", "tool": "lookup", "args": {"prev": "$a"}, "depends_on": ["a"]},
    ])
    planner = _StubProvider([plan])
    executor = _StubProvider(["done"])

    captured: list[dict[str, Any]] = []

    class _Capture(GeneralChatTool):
        @property
        def name(self) -> str:
            return "lookup"

        @property
        def description(self) -> str:
            return "captures args"

        @property
        def parameters_schema(self) -> dict:
            return {"type": "object", "additionalProperties": True}

        async def execute(self, **kwargs: Any) -> dict[str, Any]:
            captured.append(kwargs)
            return {"ok": True}

    catalog = _catalog([_EchoTool("search"), _Capture()])
    node = PlannerExecutorNode(planner, executor, catalog)()
    await node({"messages": []})
    assert len(captured) == 1
    # The 'prev' arg was substituted with the result of step a.
    prev = captured[0]["prev"]
    assert isinstance(prev, dict)
    assert prev["tool"] == "search"


async def test_planner_executor_replans_on_step_failure():
    bad_plan = json.dumps([{"id": "a", "tool": "boom", "args": {}}])
    good_plan = json.dumps([{"id": "a", "tool": "search", "args": {}}])
    planner = _StubProvider([bad_plan, good_plan])
    executor = _StubProvider(["recovered"])
    catalog = _catalog([_EchoTool("boom", fail=True), _EchoTool("search")])
    node = PlannerExecutorNode(planner, executor, catalog, max_replans=1)()
    update = await node({"messages": []})
    assert update["messages"]["content"] == "recovered"
    assert len(planner.calls) == 2  # initial plan + replan


async def test_planner_executor_synthesizes_with_failed_step_log():
    """When all replans fail, synthesis still runs and sees the failure log."""
    bad_plan = json.dumps([{"id": "a", "tool": "boom", "args": {}}])
    planner = _StubProvider([bad_plan, bad_plan])
    executor = _StubProvider(["I tried but the tool failed."])
    catalog = _catalog([_EchoTool("boom", fail=True)])
    node = PlannerExecutorNode(planner, executor, catalog, max_replans=1)()
    update = await node({"messages": []})
    assert update["messages"]["role"] == "assistant"
    assert update["messages"]["content"] == "I tried but the tool failed."
    # Synthesis prompt should have included the failure log.
    synth_payload = executor.calls[0]
    log_msg = synth_payload["messages"][-1]["content"]
    assert "Tool results:" in log_msg
    assert "boom blew up" in log_msg


async def test_planner_executor_unparseable_plan_returns_error_message():
    planner = _StubProvider(["this is not JSON at all"] * 2)
    executor = _StubProvider([])
    catalog = _catalog([_EchoTool("search")])
    node = PlannerExecutorNode(planner, executor, catalog, max_replans=1)()
    update = await node({"messages": []})
    assert "wasn't able to draft" in update["messages"]["content"]


async def test_planner_executor_records_substeps_in_trace():
    from eirel.graph.runtime import RunConfig, RunContext, _RUN_CONTEXT

    plan = json.dumps([{"id": "a", "tool": "search", "args": {}}])
    planner = _StubProvider([plan])
    executor = _StubProvider(["done"])
    catalog = _catalog([_EchoTool("search")])
    trace = TraceRecorder()
    ctx = RunContext(config=RunConfig(), trace=trace)
    token = _RUN_CONTEXT.set(ctx)
    try:
        node = PlannerExecutorNode(
            planner, executor, catalog, name="pe"
        )()
        await node({"messages": []})
    finally:
        _RUN_CONTEXT.reset(token)
    names = [tc.tool_name for tc in trace.tool_calls]
    assert "pe:plan_0" in names
    assert "pe:synthesize" in names

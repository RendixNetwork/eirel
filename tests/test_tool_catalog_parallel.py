"""Tests for ``GeneralChatToolCatalog.execute_many`` parallel dispatch."""
from __future__ import annotations

import asyncio
import time

import pytest

from eirel.families.general_chat.budget import (
    INSTANT_BUDGET,
    BudgetExhaustedError,
    BudgetTracker,
)
from eirel.families.general_chat.response import TraceRecorder
from eirel.families.general_chat.tools import GeneralChatTool, GeneralChatToolCatalog
from eirel.graph.node import PendingToolCall


class _SleepTool(GeneralChatTool):
    def __init__(self, name: str, sleep_s: float, *, raises: Exception | None = None):
        self._name = name
        self._sleep_s = sleep_s
        self._raises = raises

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"sleep tool {self._name}"

    @property
    def parameters_schema(self) -> dict:
        return {"type": "object", "properties": {}, "additionalProperties": True}

    async def execute(self, **kwargs):
        await asyncio.sleep(self._sleep_s)
        if self._raises is not None:
            raise self._raises
        return {"name": self._name, "kwargs": dict(kwargs)}


def _catalog(tools):
    return GeneralChatToolCatalog(
        tools,
        budget=BudgetTracker(budget=INSTANT_BUDGET),
        trace=TraceRecorder(),
    )


async def test_execute_many_runs_in_parallel():
    catalog = _catalog([_SleepTool("a", 0.05), _SleepTool("b", 0.05), _SleepTool("c", 0.05)])
    calls = [
        PendingToolCall(name="a", call_id="1"),
        PendingToolCall(name="b", call_id="2"),
        PendingToolCall(name="c", call_id="3"),
    ]
    t0 = time.monotonic()
    results = await catalog.execute_many(calls)
    elapsed = time.monotonic() - t0
    assert elapsed < 0.12, f"branches did not run in parallel: {elapsed:.3f}s"
    assert [r["call_id"] for r in results] == ["1", "2", "3"]
    assert all(r["status"] == "ok" for r in results)


async def test_execute_many_captures_per_call_errors():
    catalog = _catalog([
        _SleepTool("ok", 0.01),
        _SleepTool("boom", 0.01, raises=RuntimeError("kaboom")),
    ])
    calls = [PendingToolCall(name="ok"), PendingToolCall(name="boom")]
    results = await catalog.execute_many(calls)
    assert results[0]["status"] == "ok"
    assert results[1]["status"] == "error"
    assert "kaboom" in results[1]["error"]


async def test_execute_many_unknown_tool_returns_error_entry():
    catalog = _catalog([_SleepTool("known", 0.01)])
    calls = [PendingToolCall(name="ghost")]
    results = await catalog.execute_many(calls)
    assert results[0]["status"] == "error"
    assert "unknown" in results[0]["error"]


async def test_execute_many_empty_returns_empty_list():
    catalog = _catalog([_SleepTool("a", 0.01)])
    results = await catalog.execute_many([])
    assert results == []


async def test_execute_many_propagates_budget_exhaustion():
    """If the catalog budget is already exhausted, no calls should fire."""
    catalog = _catalog([_SleepTool("a", 0.01)])
    # Push the tracker over the cap directly. The dataclass is mutable;
    # this avoids needing to throw from a record_* call before the test
    # body even runs.
    catalog._budget.output_tokens_used = catalog._budget.budget.output_tokens + 1

    # The catalog calls _budget.check() before dispatching.
    with pytest.raises(BudgetExhaustedError):
        await catalog.execute_many([PendingToolCall(name="a")])


async def test_tool_node_dispatches_pending_calls_via_catalog():
    from eirel.graph import StateField, StateSpec, replace, ToolNode

    catalog = _catalog([_SleepTool("a", 0.01), _SleepTool("b", 0.01)])
    spec = StateSpec({
        "pending_tool_calls": StateField(reducer=replace, default_factory=list),
        "tool_results": StateField(reducer=replace, default_factory=list),
    })
    node = ToolNode(catalog)()

    state = spec.init(pending_tool_calls=[
        {"name": "a", "arguments": {}, "call_id": "1"},
        PendingToolCall(name="b", call_id="2"),
    ])
    update = await node(state)
    assert update["pending_tool_calls"] == []
    assert len(update["tool_results"]) == 2
    assert {r["call_id"] for r in update["tool_results"]} == {"1", "2"}

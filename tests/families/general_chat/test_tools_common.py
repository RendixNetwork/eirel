from __future__ import annotations

from typing import Any

import pytest

from eirel.families.general_chat.budget import (
    INSTANT_BUDGET,
    BudgetTracker,
)
from eirel.families.general_chat.response import TraceRecorder
from eirel.families.general_chat.tools import GeneralChatTool, GeneralChatToolCatalog


class _FakeTool(GeneralChatTool):
    def __init__(self, tool_name: str = "fake_tool") -> None:
        self._name = tool_name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A fake tool for testing."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> dict[str, Any]:
        return {"ok": True, **kwargs}


def _catalog(tools: list[GeneralChatTool]) -> tuple[GeneralChatToolCatalog, BudgetTracker, TraceRecorder]:
    budget = BudgetTracker(budget=INSTANT_BUDGET)
    trace = TraceRecorder()
    return GeneralChatToolCatalog(tools, budget=budget, trace=trace), budget, trace


async def test_catalog_dispatch_records_tool_call():
    catalog, budget, trace = _catalog([_FakeTool()])
    result = await catalog.execute("fake_tool", query="abc")
    assert result == {"ok": True, "query": "abc"}
    assert len(trace.tool_calls) == 1
    assert trace.tool_calls[0].tool_name == "fake_tool"
    assert trace.tool_calls[0].args == {"query": "abc"}


async def test_catalog_unknown_tool_raises():
    catalog, _, _ = _catalog([_FakeTool()])
    with pytest.raises(ValueError, match="unknown general_chat tool"):
        await catalog.execute("missing_tool")


async def test_catalog_available_lists_tool_names():
    catalog, _, _ = _catalog([_FakeTool("alpha"), _FakeTool("beta")])
    assert sorted(catalog.available) == ["alpha", "beta"]


async def test_catalog_definitions_emit_tool_definition_objects():
    catalog, _, _ = _catalog([_FakeTool("search")])
    defs = catalog.definitions()
    assert len(defs) == 1
    assert defs[0].type == "function"
    assert defs[0].function.name == "search"


async def test_catalog_get_returns_tool_or_none():
    tool = _FakeTool("alpha")
    catalog, _, _ = _catalog([tool])
    assert catalog.get("alpha") is tool
    assert catalog.get("nothing") is None


async def test_catalog_allows_unlimited_tool_calls():
    catalog, _, trace = _catalog([_FakeTool()])
    for _ in range(10):
        await catalog.execute("fake_tool")
    assert len(trace.tool_calls) == 10

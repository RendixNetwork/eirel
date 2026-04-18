from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from eirel.families.general_chat.budget import BudgetTracker
from eirel.families.general_chat.response import ToolCall, TraceRecorder
from eirel.models import ToolDefinition, ToolFunctionDefinition


class GeneralChatTool(ABC):
    """ABC for owner-api-routed tool clients used by general_chat agents."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def parameters_schema(self) -> dict[str, Any]: ...

    @abstractmethod
    async def execute(self, **kwargs: Any) -> dict[str, Any]: ...


class GeneralChatToolCatalog:
    """Bundles tools for a single conversation turn.

    Calling ``execute(tool_name, **kwargs)`` records the call against the
    supplied :class:`BudgetTracker` (which may raise ``BudgetExhaustedError``)
    and appends a :class:`ToolCall` entry to the trace recorder.
    """

    def __init__(
        self,
        tools: list[GeneralChatTool],
        *,
        budget: BudgetTracker,
        trace: TraceRecorder,
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self._budget = budget
        self._trace = trace

    @property
    def available(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> GeneralChatTool | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                type="function",
                function=ToolFunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.parameters_schema,
                ),
            )
            for tool in self._tools.values()
        ]

    async def execute(self, tool_name: str, **kwargs: Any) -> dict[str, Any]:
        self._budget.check()
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"unknown general_chat tool: {tool_name!r}")
        start = time.monotonic()
        result = await tool.execute(**kwargs)
        latency_ms = int((time.monotonic() - start) * 1000)
        self._trace.record_tool_call(
            ToolCall(
                tool_name=tool_name,
                args=dict(kwargs),
                result_digest=str(result)[:500] if result else None,
                latency_ms=latency_ms,
            )
        )
        return result


__all__ = ["GeneralChatTool", "GeneralChatToolCatalog"]

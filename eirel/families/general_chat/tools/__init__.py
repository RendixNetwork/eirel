from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from eirel.families.general_chat.budget import BudgetTracker
from eirel.families.general_chat.response import ToolCall, TraceRecorder
from eirel.models import ToolDefinition, ToolFunctionDefinition

if TYPE_CHECKING:
    from eirel.graph.node import PendingToolCall


_logger = logging.getLogger(__name__)


# -- Retry / fallback policy --------------------------------------------------


def _is_empty_result(result: Any) -> bool:
    """Default empty-predicate: a falsy mapping or sequence of zero hits.

    Treats ``None``, ``{}``, ``[]``, ``""`` as empty. For dict results
    that wrap a list (e.g. ``{"results": []}``), checks the common
    list-bearing keys ``results``, ``hits``, ``items``, ``entries``.
    """
    if result is None:
        return True
    if isinstance(result, str):
        return not result.strip()
    if isinstance(result, (list, tuple, set)):
        return len(result) == 0
    if isinstance(result, dict):
        if not result:
            return True
        for key in ("results", "hits", "items", "entries"):
            if key in result:
                value = result[key]
                if isinstance(value, (list, tuple, set)) and len(value) == 0:
                    return True
        return False
    return False


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """Per-call retry policy used by :meth:`GeneralChatToolCatalog.execute_with_policy`.

    Re-runs the tool ``max_attempts`` times when ``execute`` raises one
    of ``retry_on``. ``backoff`` controls inter-attempt sleep.

    Parameters
    ----------
    max_attempts
        Total tries (initial + retries). Default 2 (one retry).
    backoff_seconds
        Base sleep between attempts.
    backoff
        ``"linear"`` → sleep ``backoff_seconds * attempt``.
        ``"exponential"`` → sleep ``backoff_seconds * 2**(attempt-1)``.
    retry_on
        Exception types that trigger retry. Defaults to
        ``(Exception,)`` — anything that surfaces from the tool. Narrow
        this with a custom transient-error tuple in production.
    """

    max_attempts: int = 2
    backoff_seconds: float = 0.0
    backoff: str = "linear"
    retry_on: tuple[type[BaseException], ...] = (Exception,)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.backoff not in ("linear", "exponential"):
            raise ValueError(
                f"backoff must be 'linear' or 'exponential', got {self.backoff!r}"
            )
        if self.backoff_seconds < 0:
            raise ValueError("backoff_seconds must be non-negative")

    def sleep_seconds(self, attempt: int) -> float:
        """Seconds to wait BEFORE attempt number ``attempt`` (1-indexed).

        Returns 0 for the first attempt; only kicks in on retries.
        """
        if attempt <= 1 or self.backoff_seconds == 0:
            return 0.0
        if self.backoff == "exponential":
            return self.backoff_seconds * (2 ** (attempt - 2))
        return self.backoff_seconds * (attempt - 1)


@dataclass(frozen=True, slots=True)
class FallbackChain:
    """Cascade through tools, taking the first result that ``predicate`` accepts.

    Wraps a primary tool name plus 0+ fallbacks. Each link gets the same
    args. The first link whose result passes ``predicate`` (default:
    "result is non-empty AND no exception") wins. If every link fails
    or returns empty, the last exception (or empty result) is returned.

    Pair with :class:`RetryPolicy` to retry each link before failing
    over to the next.
    """

    primary: str
    fallbacks: tuple[str, ...] = ()
    predicate: Callable[[Any], bool] | None = None
    retry_policy: "RetryPolicy | None" = None

    @property
    def chain(self) -> tuple[str, ...]:
        return (self.primary, *self.fallbacks)


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

    async def execute_with_policy(
        self,
        tool_name: str,
        *,
        policy: RetryPolicy,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run a single tool with a retry policy.

        Re-runs the tool when ``execute`` raises one of
        ``policy.retry_on``, up to ``policy.max_attempts`` times, with
        the configured backoff between attempts. The final exception
        (after retries exhausted) is re-raised. Each attempt is
        recorded in the trace as a separate :class:`ToolCall` with
        ``tool_name`` suffixed ``":attempt_N"`` so eval can attribute
        retry latency.

        Budget is checked once up front (matches :meth:`execute`).
        """
        self._budget.check()
        tool = self._tools.get(tool_name)
        if tool is None:
            raise ValueError(f"unknown general_chat tool: {tool_name!r}")
        last_exc: BaseException | None = None
        for attempt in range(1, policy.max_attempts + 1):
            sleep_s = policy.sleep_seconds(attempt)
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            start = time.monotonic()
            try:
                result = await tool.execute(**kwargs)
            except policy.retry_on as exc:  # type: ignore[misc]
                latency_ms = int((time.monotonic() - start) * 1000)
                self._trace.record_tool_call(
                    ToolCall(
                        tool_name=f"{tool_name}:attempt_{attempt}",
                        args=dict(kwargs),
                        result_digest=f"error: {exc}"[:500],
                        latency_ms=latency_ms,
                    )
                )
                last_exc = exc
                continue
            latency_ms = int((time.monotonic() - start) * 1000)
            self._trace.record_tool_call(
                ToolCall(
                    tool_name=(
                        f"{tool_name}:attempt_{attempt}"
                        if attempt > 1
                        else tool_name
                    ),
                    args=dict(kwargs),
                    result_digest=str(result)[:500] if result else None,
                    latency_ms=latency_ms,
                )
            )
            return result
        # All attempts exhausted — re-raise the last exception. The
        # ``retry_on`` constraint guarantees ``last_exc`` is set when
        # we reach here.
        assert last_exc is not None
        raise last_exc

    async def execute_chain(
        self,
        chain: FallbackChain,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Cascade through ``chain`` until one link returns a passing result.

        Each link runs through :meth:`execute_with_policy` when
        ``chain.retry_policy`` is set; otherwise via :meth:`execute`.
        Acceptance is decided by ``chain.predicate`` (defaults to
        "non-empty result, no exception" — see :func:`_is_empty_result`).

        On every-link failure, the last exception is re-raised. If
        every link returned an "empty" result (per the predicate), the
        last empty result is returned — the caller can inspect it.

        Budget is checked at chain start; a fallback fire after a
        primary failure does NOT re-check (matches the "one logical
        tool call, multiple physical attempts" model).
        """
        self._budget.check()
        predicate = chain.predicate or (
            lambda r: not _is_empty_result(r)
        )
        last_exc: BaseException | None = None
        last_empty: Any = None
        for idx, name in enumerate(chain.chain):
            tool = self._tools.get(name)
            if tool is None:
                self._trace.record_tool_call(
                    ToolCall(
                        tool_name=f"{name}:chain_unknown",
                        args=dict(kwargs),
                        result_digest=f"error: unknown tool",
                        latency_ms=0,
                    )
                )
                last_exc = ValueError(f"unknown general_chat tool: {name!r}")
                continue
            try:
                if chain.retry_policy is not None:
                    # The retry path checks budget itself; we already did.
                    # Skip its check by inlining the loop here.
                    result = await self._run_with_policy_no_budget_check(
                        name, tool, chain.retry_policy, kwargs
                    )
                else:
                    start = time.monotonic()
                    result = await tool.execute(**kwargs)
                    latency_ms = int((time.monotonic() - start) * 1000)
                    self._trace.record_tool_call(
                        ToolCall(
                            tool_name=(
                                name if idx == 0 else f"{name}:fallback_{idx}"
                            ),
                            args=dict(kwargs),
                            result_digest=str(result)[:500] if result else None,
                            latency_ms=latency_ms,
                        )
                    )
            except Exception as exc:  # noqa: BLE001 — capture per-link
                last_exc = exc
                continue
            if predicate(result):
                return result
            last_empty = result
        if last_exc is not None:
            raise last_exc
        return last_empty if last_empty is not None else {}

    async def _run_with_policy_no_budget_check(
        self,
        tool_name: str,
        tool: "GeneralChatTool",
        policy: RetryPolicy,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        last_exc: BaseException | None = None
        for attempt in range(1, policy.max_attempts + 1):
            sleep_s = policy.sleep_seconds(attempt)
            if sleep_s > 0:
                await asyncio.sleep(sleep_s)
            start = time.monotonic()
            try:
                result = await tool.execute(**kwargs)
            except policy.retry_on as exc:  # type: ignore[misc]
                latency_ms = int((time.monotonic() - start) * 1000)
                self._trace.record_tool_call(
                    ToolCall(
                        tool_name=f"{tool_name}:attempt_{attempt}",
                        args=dict(kwargs),
                        result_digest=f"error: {exc}"[:500],
                        latency_ms=latency_ms,
                    )
                )
                last_exc = exc
                continue
            latency_ms = int((time.monotonic() - start) * 1000)
            self._trace.record_tool_call(
                ToolCall(
                    tool_name=(
                        f"{tool_name}:attempt_{attempt}"
                        if attempt > 1
                        else tool_name
                    ),
                    args=dict(kwargs),
                    result_digest=str(result)[:500] if result else None,
                    latency_ms=latency_ms,
                )
            )
            return result
        assert last_exc is not None
        raise last_exc

    async def execute_many(
        self, calls: "list[PendingToolCall]"
    ) -> list[dict[str, Any]]:
        """Run N tool calls concurrently and return result dicts in the same order.

        Each result has the shape::

            {
                "name": <tool_name>,
                "call_id": <opaque id or None>,
                "result": <tool output dict>  # only if status == "ok"
                "status": "ok" | "error",
                "error": <exception message>  # only if status == "error"
                "latency_ms": <int>,
            }

        Per-call exceptions are captured into the result entry so a
        partial fan-out failure doesn't kill the whole graph node.
        Budget violations propagate — the catalog's :class:`BudgetTracker`
        raises :class:`~eirel.families.general_chat.budget.BudgetExhaustedError`,
        which the graph runtime treats as terminal. This is checked
        once before dispatch (``self._budget.check()``) so all branches
        either run or none do; per-branch budget enforcement (e.g. token
        accounting per LLM tool call) lives inside individual tools.
        """
        self._budget.check()
        if not calls:
            return []

        async def _one(call: "PendingToolCall") -> dict[str, Any]:
            tool = self._tools.get(call.name)
            start = time.monotonic()
            if tool is None:
                latency_ms = int((time.monotonic() - start) * 1000)
                error = f"unknown general_chat tool: {call.name!r}"
                self._trace.record_tool_call(
                    ToolCall(
                        tool_name=call.name,
                        args=dict(call.arguments),
                        result_digest=error,
                        latency_ms=latency_ms,
                    )
                )
                return {
                    "name": call.name,
                    "call_id": call.call_id,
                    "status": "error",
                    "error": error,
                    "latency_ms": latency_ms,
                }
            try:
                result = await tool.execute(**call.arguments)
            except Exception as exc:  # noqa: BLE001 — capture per-branch
                latency_ms = int((time.monotonic() - start) * 1000)
                self._trace.record_tool_call(
                    ToolCall(
                        tool_name=call.name,
                        args=dict(call.arguments),
                        result_digest=f"error: {exc}"[:500],
                        latency_ms=latency_ms,
                    )
                )
                return {
                    "name": call.name,
                    "call_id": call.call_id,
                    "status": "error",
                    "error": str(exc),
                    "latency_ms": latency_ms,
                }
            latency_ms = int((time.monotonic() - start) * 1000)
            self._trace.record_tool_call(
                ToolCall(
                    tool_name=call.name,
                    args=dict(call.arguments),
                    result_digest=str(result)[:500] if result else None,
                    latency_ms=latency_ms,
                )
            )
            return {
                "name": call.name,
                "call_id": call.call_id,
                "status": "ok",
                "result": result,
                "latency_ms": latency_ms,
            }

        return await asyncio.gather(*[_one(c) for c in calls])


__all__ = [
    "FallbackChain",
    "GeneralChatTool",
    "GeneralChatToolCatalog",
    "RetryPolicy",
]

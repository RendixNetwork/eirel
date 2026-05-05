"""Node primitives: the executable units of a state graph.

A node is, fundamentally, an async callable ``(state) -> partial_update``.
The runtime applies the partial update via the per-field reducers on the
:class:`~eirel.graph.state.StateSpec`. Returning ``None`` is a no-op.

For most miners writing a graph by hand, ``async def`` functions are
enough — the graph builder accepts them directly. The classes here are
*factories* that build such functions for the two most common cases:

  * :class:`LLMNode` — wraps an :class:`~eirel.provider.AgentProviderClient`
    so a node can call the LLM with the current ``messages`` field and
    append the assistant reply.
  * :class:`ToolNode` — wraps a
    :class:`~eirel.families.general_chat.tools.GeneralChatToolCatalog`
    so a node can fan out N tool calls in parallel and gather their
    results back into state.
  * :class:`BranchNode` — a no-op node whose only purpose is to host a
    conditional-edge router. State is unchanged; the edge's router does
    the real work. Useful when you need a named decision point in the
    graph for tracing/observability.

These factories are a thin convenience layer; you're free to ignore them
and write nodes by hand.
"""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from eirel.families.general_chat.tools import GeneralChatToolCatalog
    from eirel.provider import AgentProviderClient

__all__ = [
    "NodeFn",
    "PendingToolCall",
    "BranchNode",
    "LLMNode",
    "ToolNode",
]


NodeFn = Callable[[dict[str, Any]], Awaitable[dict[str, Any] | None]]


# -- Pending-tool-call payload ------------------------------------------------


@dataclass(frozen=True, slots=True)
class PendingToolCall:
    """One tool invocation queued for a :class:`ToolNode` to dispatch.

    Carry these in a state field (default key ``"pending_tool_calls"``)
    when a planner/LLM node decides which tools to run.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    call_id: str | None = None


# -- LLM node -----------------------------------------------------------------


class LLMNode:
    """Factory: builds a node that calls an LLM with the current messages.

    The factory is opinionated about the state shape:

      * ``messages_key`` (default ``"messages"``) holds an OpenAI-style list
        of ``{role, content}`` dicts. The node calls ``provider.chat_completions``
        with this payload and a ``max_tokens`` allowance derived from
        ``max_tokens`` (passed at construction time).
      * The assistant reply (``content`` only — tool-calls live elsewhere
        in the graph by design) is appended back onto the same key so the
        history grows turn by turn.

    Custom payload shaping (system prompts, tool definitions, structured
    output) lives in the planner node *before* the LLM node, not inside
    it. Keeping LLMNode dumb makes it composable.
    """

    __slots__ = (
        "_provider",
        "_messages_key",
        "_max_tokens",
        "_extra_payload",
    )

    def __init__(
        self,
        provider: "AgentProviderClient",
        *,
        messages_key: str = "messages",
        max_tokens: int = 2048,
        extra_payload: dict[str, Any] | None = None,
    ) -> None:
        self._provider = provider
        self._messages_key = messages_key
        self._max_tokens = max_tokens
        self._extra_payload = dict(extra_payload) if extra_payload else None

    def __call__(self) -> NodeFn:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            messages = state.get(self._messages_key) or []
            payload: dict[str, Any] = {
                "messages": list(messages),
                "max_tokens": self._max_tokens,
            }
            if self._extra_payload:
                for key, value in self._extra_payload.items():
                    payload.setdefault(key, value)
            response = await self._provider.chat_completions(payload)
            content = ""
            try:
                content = response["choices"][0]["message"].get("content") or ""
            except (KeyError, IndexError, TypeError):
                content = ""
            return {self._messages_key: {"role": "assistant", "content": content}}

        run.__name__ = "llm_node"
        return run


# -- Tool node ----------------------------------------------------------------


class ToolNode:
    """Factory: builds a node that dispatches pending tool calls in parallel.

    Reads ``pending_calls_key`` (default ``"pending_tool_calls"``), runs
    every entry through the
    :class:`~eirel.families.general_chat.tools.GeneralChatToolCatalog`
    via :meth:`execute_many`, and writes results to ``results_key`` (default
    ``"tool_results"``). After dispatch the pending list is cleared by
    emitting an empty list under ``pending_calls_key`` (the field's
    reducer must accept that — :func:`~eirel.graph.state.replace` is the
    natural choice for this key).

    Errors in any branch are captured into the result entry rather than
    raising, so a partially failed fan-out doesn't kill the whole graph.
    Budget violations propagate (the catalog's ``BudgetTracker`` raises
    :class:`~eirel.families.general_chat.budget.BudgetExhaustedError`,
    which the runtime treats as terminal).
    """

    __slots__ = ("_catalog", "_pending_key", "_results_key")

    def __init__(
        self,
        catalog: "GeneralChatToolCatalog",
        *,
        pending_calls_key: str = "pending_tool_calls",
        results_key: str = "tool_results",
    ) -> None:
        self._catalog = catalog
        self._pending_key = pending_calls_key
        self._results_key = results_key

    def __call__(self) -> NodeFn:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            pending = state.get(self._pending_key) or []
            if not pending:
                return {self._pending_key: []}
            calls: list[PendingToolCall] = []
            for entry in pending:
                if isinstance(entry, PendingToolCall):
                    calls.append(entry)
                elif isinstance(entry, dict):
                    calls.append(
                        PendingToolCall(
                            name=str(entry["name"]),
                            arguments=dict(entry.get("arguments") or {}),
                            call_id=entry.get("call_id"),
                        )
                    )
                else:
                    raise TypeError(
                        f"ToolNode expected PendingToolCall or dict, "
                        f"got {type(entry).__name__}"
                    )
            results = await self._catalog.execute_many(calls)
            return {
                self._pending_key: [],
                self._results_key: results,
            }

        run.__name__ = "tool_node"
        return run


# -- Branch node --------------------------------------------------------------


class BranchNode:
    """Factory: builds a no-op node for hosting a conditional-edge router.

    Useful when you want a named decision point in the graph for
    tracing/observability but the routing logic lives entirely in the
    edge's router function.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str = "branch") -> None:
        self._name = name

    def __call__(self) -> NodeFn:
        async def run(state: dict[str, Any]) -> None:  # noqa: ARG001
            return None

        run.__name__ = self._name
        return run


# -- Helpers ------------------------------------------------------------------


def _ensure_async(fn: Callable[..., Any]) -> NodeFn:
    """Normalize a sync or async callable to a NodeFn.

    Used by the graph builder to accept plain ``def`` functions for
    convenience; the executor always awaits.
    """
    if asyncio.iscoroutinefunction(fn):
        return fn  # type: ignore[return-value]

    async def wrapped(state: dict[str, Any]) -> dict[str, Any] | None:
        result = fn(state)
        if asyncio.iscoroutine(result):
            return await result
        return result

    wrapped.__name__ = getattr(fn, "__name__", "node")
    return wrapped

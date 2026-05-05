"""Rolling summary: collapse long chat history into a single summary message.

Used as a graph node to keep ``messages`` bounded so checkpoint state
stays under the 256 KB cap. The summarizer LLM is supplied by the miner
(typically the same provider client as the main LLM, possibly with a
cheaper/faster model).

Strategy
--------

When ``len(messages) > every_n_turns``, the oldest ``every_n_turns - keep_recent``
messages are sent to the summarizer LLM with a "summarize the
conversation so far" prompt. The result becomes one ``role=system``
message tagged ``[summary]`` placed at the front of the new messages
list, followed by the most recent ``keep_recent`` turns.

This is intentionally simple. Sophisticated approaches (key-value
extraction, vector retrieval over old turns) live behind the
``VectorStore`` interface — pair them with this if needed.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

__all__ = ["RollingSummary", "SummarizerFn"]


SummarizerFn = Callable[[list[dict[str, Any]]], Awaitable[str]]


@dataclass(slots=True)
class RollingSummary:
    """Summarize old turns once history exceeds ``every_n_turns``.

    Configure with a ``summarizer`` callable that takes a list of
    messages and returns a summary string. Call :meth:`maybe_summarize`
    inside a graph node:

    .. code-block:: python

        rolling = RollingSummary(summarizer=my_summary_fn, every_n_turns=20, keep_recent=4)

        async def memory_node(state):
            new_messages = await rolling.maybe_summarize(state["messages"])
            return {"messages": new_messages} if new_messages is not None else None

    The summarizer node should return the new messages list under the
    state field, with the field's reducer set to :func:`replace` so the
    rolled-up list overwrites the old one. ``add_messages`` is wrong
    here — it would append the summary instead of replacing history.
    """

    summarizer: SummarizerFn
    every_n_turns: int = 20
    keep_recent: int = 4
    summary_role: str = "system"
    summary_prefix: str = "[summary] "

    def __post_init__(self) -> None:
        if self.every_n_turns < 2:
            raise ValueError("every_n_turns must be at least 2")
        if self.keep_recent < 1:
            raise ValueError("keep_recent must be at least 1")
        if self.keep_recent >= self.every_n_turns:
            raise ValueError("keep_recent must be less than every_n_turns")

    async def maybe_summarize(
        self, messages: list[dict[str, Any]]
    ) -> list[dict[str, Any]] | None:
        """Return a collapsed messages list, or ``None`` if no work was needed."""
        if len(messages) <= self.every_n_turns:
            return None
        # Preserve the leading summary if one already exists — re-summarize
        # everything older than the keep_recent window so summaries don't
        # stack indefinitely.
        old_messages = messages[: -self.keep_recent]
        recent = messages[-self.keep_recent :]
        summary_text = await self.summarizer(old_messages)
        summary_message = {
            "role": self.summary_role,
            "content": f"{self.summary_prefix}{summary_text}",
        }
        return [summary_message] + recent

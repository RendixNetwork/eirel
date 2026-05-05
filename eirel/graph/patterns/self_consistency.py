"""Self-consistency: sample N completions, aggregate to one answer.

Pattern from Wang et al. 2022 (arXiv:2203.11171). For tasks where the
LLM tends to be right-on-average but noisy in any single sample,
sampling N times at temperature > 0 and taking the majority answer
beats a single greedy decode.

This node is opinionated only about the I/O shape:

  * Reads ``messages_key`` (default ``"messages"``).
  * Calls ``provider.chat_completions`` ``n`` times with ``temperature``
    + (optionally) per-sample ``seed`` overrides merged into the
    payload.
  * Aggregates the N response texts via ``aggregator`` (default:
    :func:`majority_vote`) into a single consensus string.
  * Writes the consensus back to ``messages_key`` as one assistant
    message — the rest of the graph sees the same shape an
    :class:`~eirel.graph.node.LLMNode` would have produced.

Each sample is recorded as a :class:`ToolCall` sub-span on the active
:class:`TraceRecorder` so the eval pipeline can surface the per-sample
cost and the post-aggregation winner.

Cost note: ``n=5`` quintuples LLM spend per turn. Default to ``n=3``;
miners that overshoot lose to ones that don't on the per-run USD
ceiling.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import Counter
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from eirel.graph.node import NodeFn
from eirel.graph.patterns._trace import record_substep

if TYPE_CHECKING:
    from eirel.provider import AgentProviderClient

__all__ = ["SelfConsistencyNode", "majority_vote"]

_logger = logging.getLogger(__name__)


_WHITESPACE = re.compile(r"\s+")


def _normalize(text: str) -> str:
    """Casefold + strip + collapse whitespace for vote bucketing."""
    return _WHITESPACE.sub(" ", (text or "").strip().casefold())


def majority_vote(samples: Sequence[str]) -> str:
    """Return the most-common sample by normalized text.

    Tiebreak rule: first-seen wins (matches ``Counter.most_common``
    when frequencies tie). Returns the empty string for an empty input.
    Returns the original-cased version of the winning bucket — not the
    normalized form — so the consumer sees a readable response.
    """
    if not samples:
        return ""
    buckets: dict[str, str] = {}  # normalized -> first-seen original
    counts: Counter[str] = Counter()
    for raw in samples:
        norm = _normalize(raw)
        counts[norm] += 1
        buckets.setdefault(norm, raw)
    winner_norm, _ = counts.most_common(1)[0]
    return buckets[winner_norm]


class SelfConsistencyNode:
    """Factory: builds a self-consistency node over ``provider``.

    Parameters
    ----------
    provider
        :class:`AgentProviderClient` to call.
    n
        Number of samples per turn (default 3). Costs ``n`` × a single
        LLM call.
    aggregator
        Callable ``list[str] -> str``. Defaults to :func:`majority_vote`
        (good for short categorical or numeric answers). Pass a custom
        aggregator for free-form text (e.g. an LLM-judge picker).
    messages_key
        State field with the chat history. Default ``"messages"``.
    max_tokens
        Output-token allowance per sample.
    temperature
        Sampling temperature merged into the request payload. Default
        ``0.7`` — high enough to vary samples, low enough to stay on-topic.
    seeds
        Optional per-sample seed list. When provided, length must equal
        ``n``; the seed is merged into the payload as ``seed=<int>`` so
        providers that honor it produce deterministic-yet-varied samples.
    extra_payload
        Additional payload fields merged into every sample call (e.g.
        ``response_format``, ``top_p``).
    name
        Sub-span name prefix recorded in the trace. Default
        ``"self_consistency"``.
    """

    __slots__ = (
        "_provider",
        "_n",
        "_aggregator",
        "_messages_key",
        "_max_tokens",
        "_temperature",
        "_seeds",
        "_extra_payload",
        "_name",
    )

    def __init__(
        self,
        provider: "AgentProviderClient",
        *,
        n: int = 3,
        aggregator: Callable[[Sequence[str]], str] | None = None,
        messages_key: str = "messages",
        max_tokens: int = 2048,
        temperature: float = 0.7,
        seeds: Sequence[int] | None = None,
        extra_payload: dict[str, Any] | None = None,
        name: str = "self_consistency",
    ) -> None:
        if n < 1:
            raise ValueError("n must be at least 1")
        if seeds is not None and len(seeds) != n:
            raise ValueError(
                f"seeds length {len(seeds)} does not match n={n}"
            )
        self._provider = provider
        self._n = n
        self._aggregator = aggregator or majority_vote
        self._messages_key = messages_key
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._seeds = list(seeds) if seeds is not None else None
        self._extra_payload = dict(extra_payload) if extra_payload else None
        self._name = name

    def _build_payload(self, messages: list[dict[str, Any]], idx: int) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": list(messages),
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if self._seeds is not None:
            payload["seed"] = int(self._seeds[idx])
        if self._extra_payload:
            for key, value in self._extra_payload.items():
                payload.setdefault(key, value)
        return payload

    async def _one_sample(
        self, messages: list[dict[str, Any]], idx: int
    ) -> str:
        payload = self._build_payload(messages, idx)
        t0 = time.monotonic()
        response = await self._provider.chat_completions(payload)
        latency_ms = int((time.monotonic() - t0) * 1000)
        content = ""
        try:
            content = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            content = ""
        record_substep(
            tool_name=f"{self._name}:sample_{idx}",
            args={"temperature": self._temperature, "seed": payload.get("seed")},
            result_digest=content[:500] if content else None,
            latency_ms=latency_ms,
        )
        return content

    def __call__(self) -> NodeFn:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            messages = list(state.get(self._messages_key) or [])
            tasks = [self._one_sample(messages, i) for i in range(self._n)]
            samples = await asyncio.gather(*tasks)
            consensus = self._aggregator(samples)
            record_substep(
                tool_name=f"{self._name}:aggregate",
                args={"n": self._n, "aggregator": getattr(self._aggregator, "__name__", "custom")},
                result_digest=consensus[:500] if consensus else None,
                latency_ms=0,
            )
            return {
                self._messages_key: {"role": "assistant", "content": consensus},
            }

        run.__name__ = self._name
        return run

"""Pluggable safety guard ABC.

A graph compiles with an optional :class:`Guard` (default
:class:`~eirel.safety.noop.NoopGuard`). The runtime calls
:meth:`Guard.pre_input` once before the entry node and
:meth:`Guard.post_output` once before reaching ``END``. Returning a
:class:`GuardVerdict` with ``allow=False`` short-circuits the run with
a deferred-style outcome that carries the reason — same shape as an
:class:`~eirel.graph.runtime.Interrupt`, so the caller treats it
uniformly.

Pluggability matters because real safety stacks vary: some pods run
Llama-Guard locally, some shell out to NeMo, some hit a hosted moderation
API, and most production deployments chain several. The ABC + chain
pattern keeps the integration points stable while letting miners pick
their own scanners. Concrete adapters (``LlamaGuardAdapter`` /
``NeMoGuardAdapter``) live behind extras and are imported lazily.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Guard", "GuardVerdict"]


@dataclass(frozen=True, slots=True)
class GuardVerdict:
    """Outcome of one guard call.

    ``allow=False`` blocks the graph. ``redactions`` lets a guard suggest
    safe replacements for offending fields; the runtime applies them as
    a state-merge update before continuing on ``allow=True``. ``metadata``
    is forwarded to the tracer so observability tools can surface what
    fired without re-running the guard.
    """

    allow: bool
    reason: str | None = None
    redactions: Mapping[str, Any] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(cls, **metadata: Any) -> "GuardVerdict":
        return cls(allow=True, metadata=dict(metadata) if metadata else {})

    @classmethod
    def deny(cls, reason: str, **metadata: Any) -> "GuardVerdict":
        return cls(allow=False, reason=reason, metadata=dict(metadata) if metadata else {})


class Guard(ABC):
    """Two-call interface: inspect the input before any node runs, and
    inspect the output before the graph terminates.

    Implementations MUST be pure-async and side-effect free except for
    their own scanner calls. Mutating ``state`` directly is forbidden;
    return ``redactions`` instead and the runtime will merge them via
    the spec's reducers.
    """

    @abstractmethod
    async def pre_input(self, state: Mapping[str, Any]) -> GuardVerdict:
        ...

    @abstractmethod
    async def post_output(self, state: Mapping[str, Any]) -> GuardVerdict:
        ...

    async def aclose(self) -> None:
        return None

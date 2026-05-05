"""Tracer ABC + per-run helpers.

The graph runtime emits a :class:`SpanHandle` per node and per
significant event (tool call, guard verdict, interrupt, etc.). Backends
implement how to record those — stdout, Langfuse, OpenTelemetry, a
custom buffer — without the runtime caring.

Why a custom interface, not OpenTelemetry directly
--------------------------------------------------

OTEL's API is heavy (context propagation, samplers, exporters) and
production-quality OTEL collectors are not always installed where
miners run. Eirel ships a tiny ABC that any of those can adapt to
(``OpenTelemetryTracer`` is a one-file adapter), keeping the core dep
graph empty. Critically the trace *frames* this module emits are
designed to feed eiretes' KPI computation directly — that's the
contract that matters, not the underlying tracer plumbing.

Schema versioning
-----------------

:data:`EIREL_TRACE_SCHEMA_VERSION` is stamped onto every emitted frame
so eiretes can reject mismatched producers with a 422 instead of
silently misinterpreting the shape.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EIREL_TRACE_SCHEMA_VERSION",
    "SpanHandle",
    "Tracer",
]


# Bumped only on incompatible frame-shape changes. Optional metadata
# fields are forward-compat — adding them does NOT bump.
EIREL_TRACE_SCHEMA_VERSION: str = "1.0"


@dataclass(slots=True)
class SpanHandle:
    """Opaque handle returned by :meth:`Tracer.span_start`.

    The miner code never reads its fields; it just hands the handle
    back to ``span_end``. Backends may stash whatever they need —
    OTEL spans, monotonic timers, external transaction ids.
    """

    name: str
    attrs: dict[str, Any] = field(default_factory=dict)
    backend_state: Any = None


class Tracer(ABC):
    """Three-call interface: start span, end span, fire one-shot event."""

    @abstractmethod
    def span_start(
        self,
        name: str,
        *,
        attrs: Mapping[str, Any] | None = None,
        parent: SpanHandle | None = None,
    ) -> SpanHandle:
        ...

    @abstractmethod
    def span_end(
        self,
        handle: SpanHandle,
        *,
        status: str = "ok",
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        ...

    @abstractmethod
    def event(
        self,
        name: str,
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        ...

    def aclose(self) -> None:
        return None

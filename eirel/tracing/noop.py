"""No-op tracer. The runtime's default when none is configured."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from eirel.tracing.tracer import SpanHandle, Tracer

__all__ = ["NoopTracer"]


class NoopTracer(Tracer):
    def span_start(
        self,
        name: str,
        *,
        attrs: Mapping[str, Any] | None = None,
        parent: SpanHandle | None = None,
    ) -> SpanHandle:
        return SpanHandle(name=name, attrs=dict(attrs or {}))

    def span_end(
        self,
        handle: SpanHandle,
        *,
        status: str = "ok",
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        return None

    def event(
        self,
        name: str,
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        return None

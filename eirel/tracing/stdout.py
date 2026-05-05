"""Stdout tracer — emits structured JSON lines.

Useful for dev and for kicking the tracer integration around without
shipping a full observability stack. Production deployments swap in a
Langfuse/OTEL adapter via the same interface.
"""
from __future__ import annotations

import json
import sys
import time
from collections.abc import Mapping
from typing import Any, TextIO

from eirel.tracing.tracer import (
    EIREL_TRACE_SCHEMA_VERSION,
    SpanHandle,
    Tracer,
)

__all__ = ["StdoutTracer"]


class StdoutTracer(Tracer):
    """One JSON line per span_start / span_end / event.

    Lines are co-versioned with :data:`EIREL_TRACE_SCHEMA_VERSION` so a
    consumer (eiretes) can reject mismatched producers cleanly.
    """

    def __init__(self, *, stream: TextIO | None = None) -> None:
        self._stream = stream or sys.stdout

    def _emit(self, kind: str, **fields: Any) -> None:
        record = {
            "schema_version": EIREL_TRACE_SCHEMA_VERSION,
            "kind": kind,
            "ts": time.time(),
            **fields,
        }
        line = json.dumps(record, separators=(",", ":"), default=str)
        print(line, file=self._stream, flush=True)

    def span_start(
        self,
        name: str,
        *,
        attrs: Mapping[str, Any] | None = None,
        parent: SpanHandle | None = None,
    ) -> SpanHandle:
        handle = SpanHandle(
            name=name,
            attrs=dict(attrs or {}),
            backend_state={"started_monotonic": time.monotonic()},
        )
        self._emit(
            "span_start",
            name=name,
            attrs=dict(attrs or {}),
            parent=parent.name if parent else None,
        )
        return handle

    def span_end(
        self,
        handle: SpanHandle,
        *,
        status: str = "ok",
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        started = (handle.backend_state or {}).get("started_monotonic")
        latency_ms = (
            int((time.monotonic() - started) * 1000) if started is not None else None
        )
        self._emit(
            "span_end",
            name=handle.name,
            status=status,
            latency_ms=latency_ms,
            attrs=dict(attrs or {}),
        )

    def event(
        self,
        name: str,
        *,
        attrs: Mapping[str, Any] | None = None,
    ) -> None:
        self._emit("event", name=name, attrs=dict(attrs or {}))

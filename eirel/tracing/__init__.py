"""Pluggable tracing for graph runs."""
from __future__ import annotations

from eirel.tracing.noop import NoopTracer
from eirel.tracing.stdout import StdoutTracer
from eirel.tracing.tracer import (
    EIREL_TRACE_SCHEMA_VERSION,
    SpanHandle,
    Tracer,
)


def __getattr__(name: str):
    if name == "LangfuseTracer":
        from eirel.tracing.langfuse import LangfuseTracer  # type: ignore[import-not-found]
        return LangfuseTracer
    raise AttributeError(f"module 'eirel.tracing' has no attribute {name!r}")


__all__ = [
    "EIREL_TRACE_SCHEMA_VERSION",
    "LangfuseTracer",
    "NoopTracer",
    "SpanHandle",
    "StdoutTracer",
    "Tracer",
]

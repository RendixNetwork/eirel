"""Internal helper: surface pattern sub-steps in the active trace.

Each pattern node records its sub-iterations as
:class:`~eirel.families.general_chat.response.ToolCall` entries on the
active :class:`~eirel.families.general_chat.response.TraceRecorder`,
keyed by ``"{pattern_name}:{step_label}"``. This lets the eval pipeline
attribute cost and latency to the right level of nesting without a new
trace schema.
"""
from __future__ import annotations

from typing import Any

from eirel.graph.runtime import current_run_context


def record_substep(
    *,
    tool_name: str,
    args: dict[str, Any] | None,
    result_digest: str | None,
    latency_ms: int,
) -> None:
    """Append a sub-step to the run's :class:`TraceRecorder`, if any.

    Best-effort: silently no-ops when no recorder is bound (unit tests
    that drive a node directly without going through the runtime).
    """
    ctx = current_run_context()
    trace = getattr(ctx, "trace", None) if ctx is not None else None
    if trace is None:
        return
    record = getattr(trace, "record_tool_call", None)
    if record is None:
        return
    # Avoid importing ToolCall at module load — it pulls pydantic; the
    # patterns module should stay light.
    from eirel.families.general_chat.response import ToolCall

    record(
        ToolCall(
            tool_name=tool_name,
            args=dict(args) if args else {},
            result_digest=(result_digest or None),
            latency_ms=max(0, int(latency_ms)),
        )
    )

"""Tests for the Tracer ABC + StdoutTracer + runtime span emission."""
from __future__ import annotations

import io
import json

from eirel import (
    EIREL_TRACE_SCHEMA_VERSION,
    NoopTracer,
    SpanHandle,
    StdoutTracer,
    StateField,
    StateGraph,
    StateSpec,
    Tracer,
    add_messages,
    replace,
)
from eirel.graph import END


class _RecordingTracer(Tracer):
    """Captures every call into a list for assertion."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, dict]] = []

    def span_start(self, name, *, attrs=None, parent=None):
        self.events.append(("span_start", name, dict(attrs or {})))
        return SpanHandle(name=name, attrs=dict(attrs or {}))

    def span_end(self, handle, *, status="ok", attrs=None):
        self.events.append(("span_end", handle.name, {"status": status, **dict(attrs or {})}))

    def event(self, name, *, attrs=None):
        self.events.append(("event", name, dict(attrs or {})))


# -- StdoutTracer -----------------------------------------------------------


def test_stdout_tracer_emits_versioned_json():
    sink = io.StringIO()
    tracer = StdoutTracer(stream=sink)
    handle = tracer.span_start("node:planner", attrs={"task": "plan"})
    tracer.event("custom.fired", attrs={"k": "v"})
    tracer.span_end(handle, status="ok", attrs={"updated_keys": ["x"]})
    lines = [json.loads(line) for line in sink.getvalue().splitlines() if line.strip()]
    assert len(lines) == 3
    for record in lines:
        assert record["schema_version"] == EIREL_TRACE_SCHEMA_VERSION
    assert [r["kind"] for r in lines] == ["span_start", "event", "span_end"]
    assert lines[0]["name"] == "node:planner"
    assert lines[0]["attrs"] == {"task": "plan"}
    assert lines[1]["attrs"] == {"k": "v"}
    assert lines[2]["latency_ms"] >= 0


# -- Runtime emits per-node spans -------------------------------------------


async def test_runtime_emits_node_span_pair():
    spec = StateSpec({
        "messages": StateField(reducer=add_messages, default_factory=list),
    })

    async def planner(state):
        return {"messages": {"role": "assistant", "content": "p"}}

    g = StateGraph(spec)
    g.add_node("planner", planner)
    g.add_edge("planner", END)
    g.set_entry_point("planner")

    tracer = _RecordingTracer()
    compiled = g.compile(tracer=tracer)
    await compiled.ainvoke(spec.init())

    starts = [e for e in tracer.events if e[0] == "span_start"]
    ends = [e for e in tracer.events if e[0] == "span_end"]
    assert any(name == "node:planner" for _, name, _ in starts)
    assert any(name == "node:planner" for _, name, _ in ends)


async def test_runtime_emits_guard_event_when_guard_runs():
    from eirel import Guard, GuardVerdict

    spec = StateSpec({
        "messages": StateField(reducer=add_messages, default_factory=list),
    })

    class _OkGuard(Guard):
        async def pre_input(self, state):
            return GuardVerdict.ok(label="pre")

        async def post_output(self, state):
            return GuardVerdict.ok(label="post")

    async def respond(state):
        return {"messages": {"role": "assistant", "content": "ok"}}

    g = StateGraph(spec)
    g.add_node("respond", respond)
    g.add_edge("respond", END)
    g.set_entry_point("respond")

    tracer = _RecordingTracer()
    compiled = g.compile(safety=_OkGuard(), tracer=tracer)
    await compiled.ainvoke(spec.init())

    event_names = [name for kind, name, _ in tracer.events if kind == "event"]
    assert "guard.pre_input" in event_names
    assert "guard.post_output" in event_names


async def test_noop_tracer_is_silent():
    """NoopTracer accepts the calls without raising and returns a usable handle."""
    tracer = NoopTracer()
    handle = tracer.span_start("x")
    tracer.event("y")
    tracer.span_end(handle)


async def test_runtime_marks_span_as_error_when_node_raises():
    spec = StateSpec({
        "messages": StateField(reducer=add_messages, default_factory=list),
    })

    async def boom(state):
        raise RuntimeError("kaboom")

    g = StateGraph(spec)
    g.add_node("boom", boom)
    g.add_edge("boom", END)
    g.set_entry_point("boom")

    tracer = _RecordingTracer()
    compiled = g.compile(tracer=tracer)

    chunks = []
    async for chunk in compiled.astream(spec.init()):
        chunks.append(chunk)

    boom_ends = [
        attrs for kind, name, attrs in tracer.events
        if kind == "span_end" and name == "node:boom"
    ]
    assert boom_ends and boom_ends[0]["status"] == "error"
    assert "kaboom" in boom_ends[0].get("error", "")

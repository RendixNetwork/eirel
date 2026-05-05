from __future__ import annotations

import asyncio
import time

import pytest

from eirel.graph import (
    END,
    GraphRecursionError,
    StateField,
    StateGraph,
    StateSpec,
    add_messages,
    merge_dict,
    replace,
)


@pytest.fixture
def basic_spec():
    return StateSpec({
        "msgs": StateField(reducer=add_messages, default_factory=list),
        "next": StateField(reducer=replace, default=""),
        "count": StateField(reducer=replace, default=0),
    })


async def test_runtime_linear_path(basic_spec):
    async def hello(state):
        return {"msgs": {"role": "user", "content": "hi"}}

    async def reply(state):
        return {"msgs": {"role": "assistant", "content": "hello back"}}

    g = StateGraph(basic_spec)
    g.add_node("hello", hello)
    g.add_node("reply", reply)
    g.add_edge("hello", "reply")
    g.add_edge("reply", END)
    g.set_entry_point("hello")

    compiled = g.compile()
    final = await compiled.ainvoke(basic_spec.init())
    assert final["msgs"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello back"},
    ]


async def test_runtime_conditional_routing(basic_spec):
    async def gate(state):
        return {"next": "yes" if state["count"] >= 1 else "no", "count": state["count"] + 1}

    async def yes_node(state):
        return {"msgs": {"role": "system", "content": "took yes"}}

    async def no_node(state):
        return {"msgs": {"role": "system", "content": "took no"}}

    g = StateGraph(basic_spec)
    g.add_node("gate", gate)
    g.add_node("yes_node", yes_node)
    g.add_node("no_node", no_node)
    g.add_conditional_edges(
        "gate",
        router=lambda s: s["next"],
        mapping={"yes": "yes_node", "no": "no_node"},
    )
    g.add_edge("yes_node", END)
    g.add_edge("no_node", "gate")
    g.set_entry_point("gate")

    compiled = g.compile(recursion_limit=10)
    final = await compiled.ainvoke(basic_spec.init())
    # First gate sees count=0 → routes to no, increments to 1
    # no_node loops back to gate; gate sees count=1 → routes to yes
    # yes_node ends.
    assert final["count"] == 2
    assert any("took no" in m.get("content", "") for m in final["msgs"])
    assert any("took yes" in m.get("content", "") for m in final["msgs"])


async def test_runtime_parallel_fan_out(basic_spec):
    spec = StateSpec({
        "input": StateField(reducer=replace, default=""),
        "a_done": StateField(reducer=replace, default=False),
        "b_done": StateField(reducer=replace, default=False),
        "c_done": StateField(reducer=replace, default=False),
        "joined": StateField(reducer=replace, default=False),
    })

    async def kickoff(state):
        return {"input": "x"}

    async def a(state):
        await asyncio.sleep(0.05)
        return {"a_done": True}

    async def b(state):
        await asyncio.sleep(0.05)
        return {"b_done": True}

    async def c(state):
        await asyncio.sleep(0.05)
        return {"c_done": True}

    async def join(state):
        return {"joined": state["a_done"] and state["b_done"] and state["c_done"]}

    g = StateGraph(spec)
    g.add_node("kickoff", kickoff)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_node("c", c)
    g.add_node("join", join)
    g.add_parallel_edges("kickoff", ["a", "b", "c"], join="join")
    g.add_edge("join", END)
    g.set_entry_point("kickoff")

    compiled = g.compile()
    t0 = time.monotonic()
    final = await compiled.ainvoke(spec.init())
    elapsed = time.monotonic() - t0

    assert final["joined"] is True
    # Three branches each sleep 0.05s. If serial, elapsed >= 0.15s.
    # If parallel, ~0.05s. Allow generous slack.
    assert elapsed < 0.12, f"branches did not run in parallel: {elapsed:.3f}s"


async def test_runtime_parallel_branch_failure_cancels_siblings():
    spec = StateSpec({
        "ran": StateField(reducer=add_messages, default_factory=list),
    })

    cancelled = asyncio.Event()

    async def fast_fail(state):
        await asyncio.sleep(0.01)
        raise RuntimeError("boom")

    async def slow_branch(state):
        try:
            await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return {"ran": "slow"}

    async def kickoff(state):
        return None

    async def join(state):
        return None

    g = StateGraph(spec)
    g.add_node("kickoff", kickoff)
    g.add_node("fast", fast_fail)
    g.add_node("slow", slow_branch)
    g.add_node("join", join)
    g.add_parallel_edges("kickoff", ["fast", "slow"], join="join")
    g.add_edge("join", END)
    g.set_entry_point("kickoff")

    compiled = g.compile()
    with pytest.raises(RuntimeError, match="boom"):
        await compiled.ainvoke(spec.init())
    assert cancelled.is_set(), "slow sibling branch was not cancelled"


async def test_runtime_recursion_limit(basic_spec):
    async def loop(state):
        return {"count": state["count"] + 1, "next": "loop"}

    g = StateGraph(basic_spec)
    g.add_node("loop", loop)
    # Phantom exit so END is statically reachable; router always loops
    # in practice, triggering the recursion limit.
    g.add_conditional_edges(
        "loop",
        router=lambda s: s["next"],
        mapping={"loop": "loop", "stop": END},
    )
    g.set_entry_point("loop")

    compiled = g.compile(recursion_limit=5)
    with pytest.raises(GraphRecursionError) as excinfo:
        await compiled.ainvoke(basic_spec.init())
    assert excinfo.value.limit == 5


async def test_runtime_streaming_emits_done():
    spec = StateSpec({
        "answer": StateField(reducer=replace, default=""),
    })

    async def respond(state):
        return {"answer": "hello world"}

    g = StateGraph(spec)
    g.add_node("respond", respond)
    g.add_edge("respond", END)
    g.set_entry_point("respond")

    compiled = g.compile()
    chunks = []
    async for chunk in compiled.astream(spec.init()):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0].event == "done"
    assert chunks[0].status == "completed"
    assert chunks[0].output == {"answer": "hello world"}


async def test_runtime_streaming_emits_failed_done_on_error(basic_spec):
    async def boom(state):
        raise RuntimeError("planner exploded")

    g = StateGraph(basic_spec)
    g.add_node("boom", boom)
    g.add_edge("boom", END)
    g.set_entry_point("boom")

    compiled = g.compile()
    chunks = []
    async for chunk in compiled.astream(basic_spec.init()):
        chunks.append(chunk)

    assert chunks[-1].event == "done"
    assert chunks[-1].status == "failed"
    assert "planner exploded" in (chunks[-1].error or "")


async def test_runtime_node_returning_invalid_type_raises(basic_spec):
    async def bad(state):
        return ["not", "a", "dict"]

    g = StateGraph(basic_spec)
    g.add_node("bad", bad)
    g.add_edge("bad", END)
    g.set_entry_point("bad")

    compiled = g.compile()
    with pytest.raises(TypeError, match="dict"):
        await compiled.ainvoke(basic_spec.init())


async def test_runtime_run_context_visible_inside_node(basic_spec):
    from eirel.graph import RunConfig, RunContext, current_run_context, run_to_completion

    seen: list[str | None] = []

    async def peek(state):
        ctx = current_run_context()
        seen.append(ctx.config.thread_id if ctx else None)
        return None

    g = StateGraph(basic_spec)
    g.add_node("peek", peek)
    g.add_edge("peek", END)
    g.set_entry_point("peek")
    compiled = g.compile()

    cfg = RunConfig(thread_id="thread-42")
    ctx = RunContext(config=cfg)
    await run_to_completion(compiled, basic_spec.init(), context=ctx)

    assert seen == ["thread-42"]

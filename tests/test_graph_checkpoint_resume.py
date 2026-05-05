"""Tests for graph-runtime checkpoint emission + Interrupt resume."""
from __future__ import annotations

import os

import pytest

from eirel import (
    AgentCapabilityMetadata,
    AgentInvocationRequest,
    AgentInvocationResponse,
    GraphAgent,
    StateField,
    StateGraph,
    StateSpec,
    add_messages,
    replace,
)
from eirel.checkpoint import InMemoryCheckpointer, encode_thread_token
from eirel.graph import END, Interrupt, RunConfig


def _make_spec():
    return StateSpec({
        "messages": StateField(reducer=add_messages, default_factory=list),
        "answer": StateField(reducer=replace, default=""),
        "step": StateField(reducer=replace, default=0),
    })


# -- Per-node checkpointing --------------------------------------------------


async def test_runtime_checkpoints_each_node():
    spec = _make_spec()

    async def first(state):
        return {"step": 1}

    async def second(state):
        return {"step": 2, "answer": "done"}

    g = StateGraph(spec)
    g.add_node("first", first)
    g.add_node("second", second)
    g.add_edge("first", "second")
    g.add_edge("second", END)
    g.set_entry_point("first")

    checkpointer = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=checkpointer)
    await compiled.ainvoke(spec.init(), config=RunConfig(thread_id="t1"))

    history = await checkpointer.alist("t1")
    # One pre-node checkpoint per executed node (first, second).
    nodes = [h.node for h in history]
    assert "first" in nodes
    assert "second" in nodes
    # Newest first.
    assert history[0].node == "second"


async def test_runtime_skips_checkpoint_without_thread_id():
    """No thread_id → no checkpoints written even if checkpointer is configured."""
    spec = _make_spec()

    async def step(state):
        return {"step": 1}

    g = StateGraph(spec)
    g.add_node("step", step)
    g.add_edge("step", END)
    g.set_entry_point("step")

    checkpointer = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=checkpointer)
    await compiled.ainvoke(spec.init())  # no RunConfig → thread_id is None
    assert await checkpointer.aget("t1") is None


# -- Interrupt + resume -------------------------------------------------------


async def test_interrupt_pauses_and_resumes_at_same_node():
    spec = _make_spec()

    pause_state = {"raised": False}

    async def gate(state):
        if not pause_state["raised"]:
            pause_state["raised"] = True
            raise Interrupt(node="gate", payload={"reason": "human_review"})
        return {"step": state["step"] + 1, "answer": "after-resume"}

    g = StateGraph(spec)
    g.add_node("gate", gate)
    g.add_edge("gate", END)
    g.set_entry_point("gate")

    checkpointer = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=checkpointer)

    # First turn: gate raises Interrupt, runtime captures it as a deferred
    # streaming chunk.
    chunks = []
    async for chunk in compiled.astream(
        spec.init(), config=RunConfig(thread_id="t-resume")
    ):
        chunks.append(chunk)
    assert chunks[-1].event == "done"
    assert chunks[-1].status == "deferred"
    interrupt_meta = chunks[-1].metadata["interrupt"]
    assert interrupt_meta["node"] == "gate"
    assert interrupt_meta["payload"] == {"reason": "human_review"}
    assert interrupt_meta["thread_id"] == "t-resume"
    assert interrupt_meta["checkpoint_id"]

    # Second turn: resume via checkpoint_id, should pick up at the same
    # node and proceed past the interrupt this time.
    final = await compiled.ainvoke(
        spec.init(),
        config=RunConfig(
            thread_id="t-resume",
            checkpoint_id=interrupt_meta["checkpoint_id"],
        ),
    )
    assert final["answer"] == "after-resume"
    assert final["step"] == 1


async def test_interrupt_in_ainvoke_returns_none_path_is_handled_via_signal():
    """Sanity: ainvoke surfaces the internal _InterruptSignal as a raised exception
    (not a clean return). GraphAgent translates that into a deferred response."""
    from eirel.graph.runtime import _InterruptSignal

    spec = _make_spec()

    async def boom(state):
        raise Interrupt(node="boom", payload={"why": "test"})

    g = StateGraph(spec)
    g.add_node("boom", boom)
    g.add_edge("boom", END)
    g.set_entry_point("boom")
    compiled = g.compile(checkpointer=InMemoryCheckpointer())

    with pytest.raises(_InterruptSignal):
        await compiled.ainvoke(spec.init(), config=RunConfig(thread_id="t-bad"))


# -- GraphAgent translates Interrupt → deferred AgentInvocationResponse ------


async def test_graph_agent_returns_deferred_response_on_interrupt(monkeypatch):
    monkeypatch.setenv("EIREL_RESUME_TOKEN_SECRET", "test-resume-secret")
    spec = _make_spec()

    async def gate(state):
        raise Interrupt(node="gate", payload={"reason": "needs_input"})

    g = StateGraph(spec)
    g.add_node("gate", gate)
    g.add_edge("gate", END)
    g.set_entry_point("gate")
    compiled = g.compile(checkpointer=InMemoryCheckpointer())

    def _to_state(req):
        return spec.init()

    def _from_state(state, req):
        return AgentInvocationResponse(
            family_id="general_chat",
            status="completed",
            output={"answer": state["answer"]},
        )

    agent = GraphAgent(
        hotkey="5HotkeyTest",
        endpoint="http://localhost:9999",
        version="0.1.0",
        capabilities=AgentCapabilityMetadata(
            family_id="general_chat", description="x", supports_streaming=True
        ),
        graph=compiled,
        to_state=_to_state,
        from_state=_from_state,
    )
    response = await agent.infer(AgentInvocationRequest(prompt="x", turn_id="thread-1"))
    assert response.status == "deferred"
    assert response.resume_token  # signed when env secret is set
    interrupt_meta = response.metadata["interrupt"]
    assert interrupt_meta["node"] == "gate"
    assert interrupt_meta["payload"] == {"reason": "needs_input"}


async def test_graph_agent_resume_via_request_resume_token(monkeypatch):
    """End-to-end: turn 1 returns deferred + resume_token; turn 2 passes
    that token back and the graph resumes past the interrupt."""
    monkeypatch.setenv("EIREL_RESUME_TOKEN_SECRET", "rt-secret")
    spec = _make_spec()
    pause_state = {"raised": False}

    async def gate(state):
        if not pause_state["raised"]:
            pause_state["raised"] = True
            raise Interrupt(node="gate", payload={})
        return {"answer": "resumed"}

    g = StateGraph(spec)
    g.add_node("gate", gate)
    g.add_edge("gate", END)
    g.set_entry_point("gate")
    cp = InMemoryCheckpointer()
    compiled = g.compile(checkpointer=cp)

    agent = GraphAgent(
        hotkey="5HotkeyTest",
        endpoint="http://localhost:9999",
        version="0.1.0",
        capabilities=AgentCapabilityMetadata(
            family_id="general_chat", description="x", supports_streaming=True
        ),
        graph=compiled,
        to_state=lambda req: spec.init(),
        from_state=lambda state, req: AgentInvocationResponse(
            family_id="general_chat",
            status="completed",
            output={"answer": state["answer"]},
        ),
    )

    # Turn 1
    deferred = await agent.infer(AgentInvocationRequest(prompt="ask", turn_id="thread-r"))
    assert deferred.status == "deferred"
    token = deferred.resume_token
    assert token

    # Turn 2 — caller hands the token back.
    final = await agent.infer(
        AgentInvocationRequest(prompt="continue", turn_id="thread-r", resume_token=token)
    )
    assert final.status == "completed"
    assert final.output == {"answer": "resumed"}

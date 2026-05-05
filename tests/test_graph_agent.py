from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from eirel import (
    AgentCapabilityMetadata,
    AgentInvocationRequest,
    AgentInvocationResponse,
    GraphAgent,
    StateField,
    StateGraph,
    StateSpec,
    add_messages,
    build_agent_app,
    replace,
)
from eirel.graph import END
from eirel.schemas import StreamChunk


@pytest.fixture(autouse=True)
def _disable_auth(monkeypatch):
    monkeypatch.setenv("EIREL_DISABLE_REQUEST_AUTH", "1")


def _build_simple_graph():
    spec = StateSpec({
        "messages": StateField(reducer=add_messages, default_factory=list),
        "answer": StateField(reducer=replace, default=""),
    })

    async def respond(state):
        prompt = ""
        for msg in state["messages"]:
            if msg.get("role") == "user":
                prompt = msg.get("content", "")
        return {"answer": f"echo: {prompt}"}

    g = StateGraph(spec)
    g.add_node("respond", respond)
    g.add_edge("respond", END)
    g.set_entry_point("respond")
    return g.compile(), spec


def _to_state(spec):
    def fn(request: AgentInvocationRequest) -> dict:
        msgs = [{"role": m.role, "content": m.content} for m in request.history]
        msgs.append({"role": "user", "content": request.prompt or ""})
        return spec.init(messages=msgs)
    return fn


def _from_state(state, request):
    return AgentInvocationResponse(
        task_id=request.turn_id or request.task_id,
        family_id="general_chat",
        status="completed",
        output={"answer": state["answer"]},
    )


def _build_agent() -> GraphAgent:
    compiled, spec = _build_simple_graph()
    return GraphAgent(
        hotkey="5HotkeyTest",
        endpoint="http://localhost:9999",
        version="0.1.0",
        capabilities=AgentCapabilityMetadata(
            family_id="general_chat",
            description="graph echo agent",
            supports_streaming=True,
        ),
        graph=compiled,
        to_state=_to_state(spec),
        from_state=_from_state,
    )


async def test_graph_agent_infer_returns_response():
    agent = _build_agent()
    request = AgentInvocationRequest(prompt="hi there", turn_id="t1")
    response = await agent.infer(request)
    assert response.status == "completed"
    assert response.output == {"answer": "echo: hi there"}
    assert response.task_id == "t1"


async def test_graph_agent_infer_stream_terminates_with_done():
    agent = _build_agent()
    request = AgentInvocationRequest(prompt="ping", turn_id="t2")
    chunks: list[StreamChunk] = []
    async for chunk in agent.infer_stream(request):
        chunks.append(chunk)
    assert chunks[-1].event == "done"
    assert chunks[-1].status == "completed"
    assert chunks[-1].output == {"answer": "echo: ping"}


def test_graph_agent_serves_via_miner_app():
    """The graph agent must work with the existing FastAPI MinerApp scaffold."""
    agent = _build_agent()
    app = build_agent_app(agent)
    client = TestClient(app)

    # /healthz is unauthenticated.
    health = client.get("/healthz")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    # Unary infer.
    body = {"prompt": "hello world", "turn_id": "abc", "history": []}
    resp = client.post("/v1/agent/infer", json=body)
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "completed"
    assert payload["output"]["answer"] == "echo: hello world"

    # Streaming.
    with client.stream("POST", "/v1/agent/infer/stream", json=body) as stream:
        events: list[dict] = []
        for line in stream.iter_lines():
            if not line:
                continue
            events.append(json.loads(line))
    assert events[-1]["event"] == "done"
    assert events[-1]["status"] == "completed"


async def test_graph_agent_run_context_carries_thread_id():
    """The agent should populate RunContext.config.thread_id from request.turn_id."""
    from eirel.graph import current_run_context

    spec = StateSpec({
        "captured": StateField(reducer=replace, default=None),
    })

    async def capture(state):
        ctx = current_run_context()
        return {"captured": ctx.config.thread_id if ctx else None}

    g = StateGraph(spec)
    g.add_node("capture", capture)
    g.add_edge("capture", END)
    g.set_entry_point("capture")
    compiled = g.compile()

    agent = GraphAgent(
        hotkey="5HotkeyTest",
        endpoint="http://localhost:9999",
        version="0.1.0",
        capabilities=AgentCapabilityMetadata(
            family_id="general_chat",
            description="capture",
            supports_streaming=True,
        ),
        graph=compiled,
        to_state=lambda req: spec.init(),
        from_state=lambda state, req: AgentInvocationResponse(
            family_id="general_chat",
            status="completed",
            output={"captured": state["captured"]},
        ),
    )
    response = await agent.infer(AgentInvocationRequest(prompt="x", turn_id="thread-xyz"))
    assert response.output == {"captured": "thread-xyz"}

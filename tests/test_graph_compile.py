from __future__ import annotations

import pytest

from eirel.graph import (
    END,
    StateField,
    StateGraph,
    StateSpec,
    add_messages,
    replace,
)
from eirel.graph.compile import GraphValidationError


@pytest.fixture
def spec():
    return StateSpec({
        "msgs": StateField(reducer=add_messages, default_factory=list),
        "next": StateField(reducer=replace, default=""),
    })


async def _noop(state):
    return None


def test_compile_requires_entry_point(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_edge("a", END)
    with pytest.raises(GraphValidationError, match="entry"):
        g.compile()


def test_compile_rejects_orphan_node(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_node("orphan", _noop)
    g.add_edge("a", END)
    g.set_entry_point("a")
    with pytest.raises(GraphValidationError, match="unreachable"):
        g.compile()


def test_compile_rejects_edge_to_unknown_dst(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_edge("a", "ghost")
    g.set_entry_point("a")
    with pytest.raises(GraphValidationError, match="not a declared node"):
        g.compile()


def test_compile_rejects_dangling_node_no_outbound(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge("a", "b")
    g.set_entry_point("a")
    with pytest.raises(GraphValidationError, match="no outbound edge"):
        g.compile()


def test_compile_rejects_double_edge_from_same_source(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge("a", "b")
    g.add_edge("a", END)  # second outbound — invalid
    g.add_edge("b", END)
    g.set_entry_point("a")
    with pytest.raises(GraphValidationError, match="already has an outbound edge"):
        g.compile()


def test_compile_rejects_cycle_when_recursion_limit_zero(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge("a", "b")
    g.add_edge("b", "a")  # cycle
    g.set_entry_point("a")
    with pytest.raises(GraphValidationError, match="cycle"):
        g.compile(recursion_limit=0)


def test_compile_allows_cycle_with_recursion_limit(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_node("done", _noop)
    g.add_conditional_edges(
        "a", router=lambda s: s["next"], mapping={"loop": "a", "stop": "done"}
    )
    g.add_edge("done", END)
    g.set_entry_point("a")
    compiled = g.compile(recursion_limit=10)
    assert compiled.plan.has_cycles is True
    assert compiled.plan.recursion_limit == 10


def test_compile_rejects_no_terminating_path(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_edge("a", "b")
    g.add_edge("b", "a")  # cycle, no exit
    g.set_entry_point("a")
    with pytest.raises(GraphValidationError, match="END is not reachable"):
        g.compile(recursion_limit=10)


def test_compile_accepts_finish_point_implicit_end(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_node("done", _noop)
    g.add_edge("a", "done")
    g.set_entry_point("a")
    g.set_finish_point("done")
    compiled = g.compile()
    # "done" should now have an implicit edge to END.
    assert compiled.plan.static_edges["done"] == END


def test_compile_parallel_edge_branches_dont_need_outbound(spec):
    g = StateGraph(spec)
    g.add_node("kickoff", _noop)
    g.add_node("a", _noop)
    g.add_node("b", _noop)
    g.add_node("join", _noop)
    g.add_parallel_edges("kickoff", ["a", "b"], join="join")
    g.add_edge("join", END)
    g.set_entry_point("kickoff")
    compiled = g.compile()
    assert "kickoff" in compiled.plan.parallel_edges


def test_compile_rejects_parallel_edge_to_unknown_dst(spec):
    g = StateGraph(spec)
    g.add_node("kickoff", _noop)
    g.add_node("join", _noop)
    g.add_parallel_edges("kickoff", ["a", "b"], join="join")
    g.add_edge("join", END)
    g.set_entry_point("kickoff")
    with pytest.raises(GraphValidationError, match="parallel edge"):
        g.compile()


def test_compile_rejects_conditional_mapping_to_unknown_dst(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    g.add_conditional_edges(
        "a", router=lambda s: s["next"], mapping={"x": "ghost"}
    )
    g.set_entry_point("a")
    with pytest.raises(GraphValidationError, match="not a declared node"):
        g.compile()


def test_register_duplicate_node_rejected(spec):
    g = StateGraph(spec)
    g.add_node("a", _noop)
    with pytest.raises(ValueError, match="already registered"):
        g.add_node("a", _noop)


def test_register_reserved_node_name_rejected(spec):
    g = StateGraph(spec)
    with pytest.raises(ValueError, match="reserved"):
        g.add_node("__start__", _noop)
    with pytest.raises(ValueError, match="reserved"):
        g.add_node("__end__", _noop)

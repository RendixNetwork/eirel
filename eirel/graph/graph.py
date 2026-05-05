"""StateGraph builder API.

A miner constructs a graph by:

  1. Declaring a :class:`~eirel.graph.state.StateSpec` (the field shape +
     reducers).
  2. Adding nodes (each is an async callable ``(state) -> partial_update``).
  3. Adding edges (static, conditional, or parallel fan-out/join).
  4. Setting the entry point.
  5. Calling :meth:`StateGraph.compile` to validate and freeze the layout.

The compiled :class:`CompiledGraph` is what gets executed. It exposes
``ainvoke`` (run to completion, return final state) and ``astream`` (yield
``StreamChunk`` events as the graph runs) — both implemented in
``graph.runtime``.

Example::

    from eirel.graph import StateGraph, StateSpec, StateField, add_messages, replace, END

    spec = StateSpec({
        "messages": StateField(reducer=add_messages, default_factory=list),
        "next": StateField(reducer=replace, default=""),
    })

    graph = StateGraph(spec)
    graph.add_node("planner", planner_fn)
    graph.add_node("respond", respond_fn)
    graph.add_conditional_edges(
        "planner",
        router=lambda s: s["next"],
        mapping={"answer": "respond", "stop": END},
    )
    graph.add_edge("respond", END)
    graph.set_entry_point("planner")
    compiled = graph.compile()
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from typing import TYPE_CHECKING

from eirel.graph.compile import ExecutionPlan, compile_graph
from eirel.graph.edge import (
    END,
    START,
    AnyEdge,
    ConditionalEdge,
    Edge,
    ParallelEdge,
    Router,
)
from eirel.graph.node import NodeFn, _ensure_async
from eirel.graph.state import StateSpec

if TYPE_CHECKING:
    from eirel.checkpoint.base import Checkpointer
    from eirel.safety.guard import Guard
    from eirel.tracing.tracer import Tracer

__all__ = [
    "START",
    "END",
    "StateGraph",
    "CompiledGraph",
]


_RESERVED_NAMES = {START, END}


class StateGraph:
    """Mutable builder for a graph. Call :meth:`compile` to freeze."""

    __slots__ = ("_spec", "_nodes", "_edges", "_entry", "_finish")

    def __init__(self, state_spec: StateSpec) -> None:
        if not isinstance(state_spec, StateSpec):
            raise TypeError(
                f"StateGraph expected a StateSpec, got {type(state_spec).__name__}"
            )
        self._spec = state_spec
        self._nodes: dict[str, NodeFn] = {}
        self._edges: list[AnyEdge] = []
        self._entry: str | None = None
        self._finish: str | None = None

    @property
    def state_spec(self) -> StateSpec:
        return self._spec

    @property
    def nodes(self) -> Mapping[str, NodeFn]:
        return self._nodes

    def add_node(self, name: str, fn: Callable[..., Any]) -> "StateGraph":
        """Register ``fn`` under ``name``.

        ``fn`` may be sync or async; sync callables are wrapped to await
        a coroutine if they happen to return one. Returning ``None`` is a
        no-op (state unchanged).
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"node name must be a non-empty string: {name!r}")
        if name in _RESERVED_NAMES:
            raise ValueError(f"{name!r} is reserved")
        if name in self._nodes:
            raise ValueError(f"node {name!r} already registered")
        if not callable(fn):
            raise TypeError(f"node {name!r}: fn must be callable")
        self._nodes[name] = _ensure_async(fn)
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self._edges.append(Edge(src=src, dst=dst))
        return self

    def add_conditional_edges(
        self,
        src: str,
        router: Router,
        mapping: Mapping[str, str] | None = None,
    ) -> "StateGraph":
        if not callable(router):
            raise TypeError("router must be callable")
        self._edges.append(
            ConditionalEdge(
                src=src,
                router=router,
                mapping=dict(mapping) if mapping is not None else None,
            )
        )
        return self

    def add_parallel_edges(
        self,
        src: str,
        dsts: list[str],
        join: str,
    ) -> "StateGraph":
        self._edges.append(
            ParallelEdge(src=src, dsts=tuple(dsts), join=join)
        )
        return self

    def set_entry_point(self, name: str) -> "StateGraph":
        if name not in self._nodes:
            raise ValueError(f"entry point {name!r} is not a registered node")
        self._entry = name
        return self

    def set_finish_point(self, name: str) -> "StateGraph":
        """Optional convenience: ``finish_point`` implies an edge to END.

        If set, :meth:`compile` will auto-wire ``name -> END`` if no
        outbound edge exists for ``name``.
        """
        if name not in self._nodes:
            raise ValueError(f"finish point {name!r} is not a registered node")
        self._finish = name
        return self

    def compile(
        self,
        *,
        recursion_limit: int = 25,
        checkpointer: "Checkpointer | None" = None,
        safety: "Guard | None" = None,
        tracer: "Tracer | None" = None,
    ) -> "CompiledGraph":
        """Validate the graph and return an immutable, executable form.

        ``checkpointer`` is optional: when supplied, the runtime persists
        a :class:`~eirel.checkpoint.base.CheckpointTuple` after each node
        executes (and on interrupt) so a paused graph can resume on a
        later turn. When omitted, the graph runs in-memory only and a
        :class:`~eirel.graph.runtime.Interrupt` raises a runtime error
        instead of being a clean pause.

        ``safety`` is optional: when supplied, the runtime calls
        :meth:`Guard.pre_input` once before the entry node and
        :meth:`Guard.post_output` once before reaching ``END``.
        ``allow=False`` short-circuits the run.

        ``tracer`` is optional: when supplied, the runtime emits
        ``span_start``/``span_end`` around every node execution and
        ``event`` calls for guard verdicts and interrupts.
        """
        plan = compile_graph(
            nodes=self._nodes,
            edges=list(self._edges),
            entry=self._entry,
            finish=self._finish,
            recursion_limit=recursion_limit,
        )
        return CompiledGraph(
            state_spec=self._spec,
            plan=plan,
            checkpointer=checkpointer,
            safety=safety,
            tracer=tracer,
        )


class CompiledGraph:
    """Frozen, executable graph produced by :meth:`StateGraph.compile`.

    Implements :meth:`ainvoke` and :meth:`astream` — the actual execution
    logic lives in :mod:`eirel.graph.runtime` to keep this module free of
    asyncio concerns. The runtime is imported lazily so that importing
    ``eirel.graph`` doesn't pull in the entire executor stack.
    """

    __slots__ = ("_spec", "_plan", "_checkpointer", "_safety", "_tracer")

    def __init__(
        self,
        state_spec: StateSpec,
        plan: ExecutionPlan,
        *,
        checkpointer: "Checkpointer | None" = None,
        safety: "Guard | None" = None,
        tracer: "Tracer | None" = None,
    ) -> None:
        self._spec = state_spec
        self._plan = plan
        self._checkpointer = checkpointer
        self._safety = safety
        self._tracer = tracer

    @property
    def state_spec(self) -> StateSpec:
        return self._spec

    @property
    def plan(self) -> ExecutionPlan:
        return self._plan

    @property
    def checkpointer(self) -> "Checkpointer | None":
        return self._checkpointer

    @property
    def safety(self) -> "Guard | None":
        return self._safety

    @property
    def tracer(self) -> "Tracer | None":
        return self._tracer

    async def ainvoke(self, state: dict[str, Any], *, config=None, context=None) -> dict[str, Any]:
        from eirel.graph.runtime import run_to_completion

        return await run_to_completion(self, state, config=config, context=context)

    async def astream(self, state: dict[str, Any], *, config=None, context=None):
        from eirel.graph.runtime import stream_events

        async for chunk in stream_events(self, state, config=config, context=context):
            yield chunk

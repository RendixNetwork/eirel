"""Compile-time validation of a :class:`StateGraph`.

Catches the easy mistakes (orphaned nodes, edges to undeclared targets,
parallel-edge typos, no path to ``END``) before the runtime starts so
miners don't ship a graph that deadlocks at the first user request.
Produces a frozen :class:`ExecutionPlan` the runtime walks.

Cycles are allowed if and only if the graph carries an explicit
``recursion_limit``; otherwise we reject so reflection/self-correction
loops don't accidentally run forever in production.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from eirel.graph.edge import (
    END,
    START,
    AnyEdge,
    ConditionalEdge,
    Edge,
    ParallelEdge,
)

if TYPE_CHECKING:
    from eirel.graph.node import NodeFn

__all__ = [
    "ExecutionPlan",
    "GraphValidationError",
    "compile_graph",
]


class GraphValidationError(ValueError):
    """Raised when a :class:`StateGraph` fails compile-time validation."""


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    """Frozen, validated, ready-to-execute graph layout.

    Built by :func:`compile_graph`. The runtime in ``graph.runtime`` walks
    these structures; it never inspects the original :class:`StateGraph`.
    """

    entry: str
    finish: str | None
    nodes: Mapping[str, "NodeFn"]
    static_edges: Mapping[str, str]
    """``src -> dst`` for plain unconditional edges."""
    conditional_edges: Mapping[str, ConditionalEdge]
    """``src -> ConditionalEdge`` for router-driven branching."""
    parallel_edges: Mapping[str, ParallelEdge]
    """``src -> ParallelEdge`` for declared fan-out/join."""
    recursion_limit: int
    has_cycles: bool


def compile_graph(
    *,
    nodes: Mapping[str, "NodeFn"],
    edges: list[AnyEdge],
    entry: str | None,
    finish: str | None,
    recursion_limit: int,
) -> ExecutionPlan:
    """Validate the graph and return a frozen execution plan.

    Validation rules:

      * Exactly one entry node is set.
      * Every edge references known nodes (or ``END`` as a destination).
      * No two outbound edges of incompatible types from one source.
      * Every reachable node either has an outbound edge or routes to ``END``.
      * Parallel-edge ``join`` exists and is reachable.
      * Cycles are forbidden unless ``recursion_limit > 0`` and the caller
        passed it explicitly. (The default of 25 is non-zero — cycles are
        allowed; we only reject when the caller passed ``recursion_limit=0``.)
    """
    if entry is None:
        raise GraphValidationError("StateGraph requires set_entry_point() before compile()")
    if entry not in nodes:
        raise GraphValidationError(f"entry point {entry!r} is not a declared node")
    if finish is not None and finish not in nodes and finish != END:
        raise GraphValidationError(f"finish point {finish!r} is not a declared node")
    if recursion_limit < 0:
        raise GraphValidationError("recursion_limit must be non-negative")

    static_map: dict[str, str] = {}
    cond_map: dict[str, ConditionalEdge] = {}
    par_map: dict[str, ParallelEdge] = {}

    # First pass: bucket by edge type, reject overlap on src.
    for edge in edges:
        src = edge.src
        if src == END:
            raise GraphValidationError(f"END cannot be the source of an edge")
        if src not in nodes and src != START:
            raise GraphValidationError(
                f"edge source {src!r} is not a declared node"
            )
        if src in static_map or src in cond_map or src in par_map:
            raise GraphValidationError(
                f"node {src!r} already has an outbound edge declared; "
                f"each source can have at most one edge type"
            )
        if isinstance(edge, Edge):
            if edge.dst != END and edge.dst not in nodes:
                raise GraphValidationError(
                    f"edge {src!r} -> {edge.dst!r}: destination is not a declared node"
                )
            static_map[src] = edge.dst
        elif isinstance(edge, ConditionalEdge):
            mapping = edge.mapping
            if mapping is not None:
                for label, dst in mapping.items():
                    if dst != END and dst not in nodes:
                        raise GraphValidationError(
                            f"conditional edge {src!r} mapping[{label!r}] = {dst!r}: "
                            f"destination is not a declared node"
                        )
            cond_map[src] = edge
        elif isinstance(edge, ParallelEdge):
            for dst in edge.dsts:
                if dst not in nodes:
                    raise GraphValidationError(
                        f"parallel edge {src!r} -> {dst!r}: destination is not a declared node"
                    )
            if edge.join != END and edge.join not in nodes:
                raise GraphValidationError(
                    f"parallel edge join {edge.join!r} is not a declared node"
                )
            par_map[src] = edge
        else:  # pragma: no cover — defensive
            raise GraphValidationError(f"unknown edge type: {type(edge).__name__}")

    # Translate START -> entry into the static map so the executor doesn't
    # need to special-case START.
    if START in static_map:
        # User added START -> X explicitly; align with set_entry_point.
        if static_map[START] != entry:
            raise GraphValidationError(
                f"explicit START edge points to {static_map[START]!r} but "
                f"set_entry_point({entry!r}) was also called"
            )
    else:
        static_map[START] = entry

    # Reachability pass.
    reachable = _walk_reachable(START, static_map, cond_map, par_map)
    declared = set(nodes)
    unreachable = declared - reachable
    if unreachable:
        raise GraphValidationError(
            f"unreachable nodes: {sorted(unreachable)} — every node must be on a path from START"
        )

    # Parallel-branch destinations don't need their own outbound edges:
    # the runtime implicitly returns control to the parallel edge's join
    # after every branch completes.
    parallel_branch_dsts: set[str] = set()
    for par in par_map.values():
        parallel_branch_dsts.update(par.dsts)

    # Wire implicit finish -> END before the END-reachability check so
    # set_finish_point() actually contributes a terminating edge.
    if (
        finish is not None
        and finish != END
        and finish not in static_map
        and finish not in cond_map
        and finish not in par_map
    ):
        static_map[finish] = END

    # Every reachable non-END, non-parallel-branch node must have an outbound edge.
    for node in reachable:
        if node in (START, END):
            continue
        if node in parallel_branch_dsts:
            continue
        if node not in static_map and node not in cond_map and node not in par_map:
            raise GraphValidationError(
                f"node {node!r} has no outbound edge and is not the finish point"
            )

    has_cycles = _has_cycles(static_map, cond_map, par_map)
    if has_cycles and recursion_limit == 0:
        raise GraphValidationError(
            "graph contains a cycle but recursion_limit=0 disables iteration"
        )

    # Re-walk reachability now that implicit-finish edges are wired in.
    reachable = _walk_reachable(START, static_map, cond_map, par_map)

    # END must be reachable (otherwise the executor will run until recursion_limit).
    if END not in reachable:
        raise GraphValidationError(
            "END is not reachable from START — the graph has no terminating path"
        )

    return ExecutionPlan(
        entry=entry,
        finish=finish,
        nodes=dict(nodes),
        static_edges=dict(static_map),
        conditional_edges=dict(cond_map),
        parallel_edges=dict(par_map),
        recursion_limit=recursion_limit,
        has_cycles=has_cycles,
    )


# -- internal helpers ---------------------------------------------------------


def _outbound(
    src: str,
    static_map: Mapping[str, str],
    cond_map: Mapping[str, ConditionalEdge],
    par_map: Mapping[str, ParallelEdge],
) -> list[str]:
    """All possible next-node names from ``src`` for graph-walk purposes.

    Used for reachability and cycle detection — for conditional edges we
    can't statically know which branch fires, so we treat every mapped
    destination as reachable.
    """
    out: list[str] = []
    if src in static_map:
        out.append(static_map[src])
    if src in cond_map:
        edge = cond_map[src]
        if edge.mapping is not None:
            out.extend(edge.mapping.values())
        else:
            # Unmapped router: dynamic destinations. Router output is
            # validated at runtime against declared nodes; we cannot
            # contribute to static reachability, so the validation here
            # will conservatively miss orphans on this branch.
            pass
    if src in par_map:
        edge = par_map[src]
        out.extend(edge.dsts)
        out.append(edge.join)
    return out


def _walk_reachable(
    start: str,
    static_map: Mapping[str, str],
    cond_map: Mapping[str, ConditionalEdge],
    par_map: Mapping[str, ParallelEdge],
) -> set[str]:
    visited: set[str] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for nxt in _outbound(node, static_map, cond_map, par_map):
            if nxt not in visited:
                stack.append(nxt)
    return visited


def _has_cycles(
    static_map: Mapping[str, str],
    cond_map: Mapping[str, ConditionalEdge],
    par_map: Mapping[str, ParallelEdge],
) -> bool:
    """DFS-based cycle detection on the static reachability graph.

    Conservatively reports cycles only if a static or mapped edge can
    return to an ancestor — unmapped conditional edges don't count
    because we can't prove they cycle until runtime.
    """
    WHITE, GRAY, BLACK = 0, 1, 2  # noqa: N806
    color: dict[str, int] = {}

    def visit(node: str) -> bool:
        c = color.get(node, WHITE)
        if c == GRAY:
            return True
        if c == BLACK:
            return False
        color[node] = GRAY
        for nxt in _outbound(node, static_map, cond_map, par_map):
            if visit(nxt):
                return True
        color[node] = BLACK
        return False

    seen: set[str] = set()
    seen.update(static_map.keys())
    seen.update(cond_map.keys())
    seen.update(par_map.keys())
    seen.update(static_map.values())
    for edge in cond_map.values():
        if edge.mapping is not None:
            seen.update(edge.mapping.values())
    for edge in par_map.values():
        seen.update(edge.dsts)
        seen.add(edge.join)
    for node in seen:
        if node in (END,):
            continue
        if visit(node):
            return True
    return False

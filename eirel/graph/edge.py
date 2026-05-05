"""Edge metadata for the graph builder.

An edge is just a label on the graph — the runtime executor in
``graph.runtime`` follows them. Three shapes cover what we need:

  * :class:`Edge` — unconditional ``src -> dst``.
  * :class:`ConditionalEdge` — at runtime a router function inspects the
    state and returns the name of the next node (or a list of node names
    for fan-out).
  * :class:`ParallelEdge` — declared fan-out from one source to N
    destinations, joining at a single ``join`` node. The runtime barriers
    on the join: every parallel branch must complete before the join
    runs. Useful for "do K tool calls in parallel, then summarize."

The ``END`` sentinel is just a string constant rather than a separate
type because edges store node names as strings — a node named ``"END"``
is the terminal sink. ``START`` is the symmetric source sentinel.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Union

__all__ = [
    "START",
    "END",
    "Edge",
    "ConditionalEdge",
    "ParallelEdge",
    "AnyEdge",
    "Router",
]


START = "__start__"
END = "__end__"


Router = Callable[[dict[str, Any]], Union[str, list[str]]]


@dataclass(frozen=True, slots=True)
class Edge:
    """Unconditional ``src -> dst``."""

    src: str
    dst: str


@dataclass(frozen=True, slots=True)
class ConditionalEdge:
    """Router-driven branching.

    ``router(state)`` returns either a single key or a list of keys. If
    ``mapping`` is ``None``, those keys are treated as node names directly.
    Otherwise they're looked up in ``mapping`` to get the actual node
    names, which lets the router emit semantic labels (``"tools"``,
    ``"respond"``) decoupled from the underlying node names.
    """

    src: str
    router: Router
    mapping: Mapping[str, str] | None = None

    def resolve(self, state: dict[str, Any]) -> list[str]:
        result = self.router(state)
        keys = result if isinstance(result, list) else [result]
        for key in keys:
            if not isinstance(key, str):
                raise TypeError(
                    f"router for {self.src!r} returned non-string {key!r}"
                )
        if self.mapping is None:
            return list(keys)
        resolved: list[str] = []
        for key in keys:
            if key not in self.mapping:
                raise KeyError(
                    f"router for {self.src!r} returned unmapped key {key!r}; "
                    f"mapping keys = {sorted(self.mapping)}"
                )
            resolved.append(self.mapping[key])
        return resolved


@dataclass(frozen=True, slots=True)
class ParallelEdge:
    """Declared fan-out from ``src`` to N ``dsts``, joining at ``join``.

    The runtime runs every ``dst`` concurrently and waits for all of
    them to finish before invoking ``join``. State updates from parallel
    branches are merged left-to-right in the order branches are listed,
    using the per-field reducers on the :class:`~eirel.graph.state.StateSpec`.
    """

    src: str
    dsts: tuple[str, ...]
    join: str

    def __post_init__(self) -> None:
        if len(self.dsts) < 2:
            # Single-dst parallel is just a regular edge — keep the type
            # honest.
            raise ValueError(
                f"ParallelEdge from {self.src!r} needs at least two destinations"
            )
        if self.join in self.dsts:
            raise ValueError(
                f"ParallelEdge join node {self.join!r} cannot also be a parallel dst"
            )


AnyEdge = Union[Edge, ConditionalEdge, ParallelEdge]

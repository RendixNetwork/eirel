"""Native graph primitives — eirel's LangGraph-equivalent.

Build a stateful agent by composing nodes and edges into a
:class:`StateGraph`, then :meth:`StateGraph.compile` to validate the
layout and produce an executable :class:`CompiledGraph`. Serve it as
a :class:`~eirel.agents.GraphAgent` through the standard :class:`MinerApp`.

See ``examples/graph_general_chat/app.py`` for a full reference miner.
"""
from __future__ import annotations

from eirel.graph.compile import ExecutionPlan, GraphValidationError
from eirel.graph.edge import (
    END,
    START,
    ConditionalEdge,
    Edge,
    ParallelEdge,
    Router,
)
from eirel.graph.graph import CompiledGraph, StateGraph
from eirel.graph.node import (
    BranchNode,
    LLMNode,
    NodeFn,
    PendingToolCall,
    ToolNode,
)
from eirel.graph.runtime import (
    GraphRecursionError,
    Interrupt,
    RunConfig,
    RunContext,
    current_run_context,
    run_to_completion,
    stream_events,
)
from eirel.graph.state import (
    Reducer,
    StateField,
    StateSpec,
    add_messages,
    merge_dict,
    replace,
)

__all__ = [
    # State
    "Reducer",
    "StateField",
    "StateSpec",
    "add_messages",
    "merge_dict",
    "replace",
    # Edges
    "START",
    "END",
    "Edge",
    "ConditionalEdge",
    "ParallelEdge",
    "Router",
    # Nodes
    "NodeFn",
    "PendingToolCall",
    "BranchNode",
    "LLMNode",
    "ToolNode",
    # Graph
    "StateGraph",
    "CompiledGraph",
    # Compile
    "ExecutionPlan",
    "GraphValidationError",
    # Runtime
    "RunConfig",
    "RunContext",
    "GraphRecursionError",
    "Interrupt",
    "current_run_context",
    "run_to_completion",
    "stream_events",
]

"""Reusable inference-time scaling patterns for general_chat agents.

These nodes are factories that build :class:`~eirel.graph.node.NodeFn`
callables, exactly like :class:`~eirel.graph.node.LLMNode` and
:class:`~eirel.graph.node.ToolNode`. They wrap one or more provider
calls into a higher-level pattern (self-consistency, reflection,
planner-executor, ReAct) so a miner can compose a competitive graph
without re-implementing the loop logic each submission.

Each pattern records sub-spans via the active
:class:`~eirel.families.general_chat.response.TraceRecorder` (read off
the run context) so the eval pipeline can attribute cost and latency
to the right level of nesting.
"""
from __future__ import annotations

from eirel.graph.patterns.planner_executor import PlannerExecutorNode, PlannerStep
from eirel.graph.patterns.react import ReActNode, ReActStep
from eirel.graph.patterns.reflection import ReflectionNode
from eirel.graph.patterns.self_consistency import (
    SelfConsistencyNode,
    majority_vote,
)

__all__ = [
    "PlannerExecutorNode",
    "PlannerStep",
    "ReActNode",
    "ReActStep",
    "ReflectionNode",
    "SelfConsistencyNode",
    "majority_vote",
]

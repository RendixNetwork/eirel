"""Concrete agent shapes that compose the SDK primitives.

Today this houses :class:`GraphAgent`, the bridge between a compiled
:class:`~eirel.graph.graph.CompiledGraph` and the
:class:`~eirel.base_agent.BaseAgent` contract that :class:`~eirel.app.MinerApp`
serves. Future agent shapes (multi-graph compositions, supervisor
agents, etc.) live alongside it.
"""
from __future__ import annotations

from eirel.agents.graph_agent import GraphAgent

__all__ = ["GraphAgent"]

"""Adapter: serve a compiled :class:`CompiledGraph` as a :class:`BaseAgent`.

The miner pod already speaks the ``BaseAgent`` shape — :class:`MinerApp`
expects ``infer`` and ``infer_stream`` methods. :class:`GraphAgent` lets a
miner build a graph and plug it in with two short callables that map
between the wire body (:class:`AgentInvocationRequest`) and the graph's
state dict, plus a third callable that lifts the final state back into
an :class:`AgentInvocationResponse`.

Why two mappers
---------------

Graph state is opinionated by the miner — names, shapes, what's tracked
where. The wire body is fixed by the subnet contract. Rather than force
the graph state to mirror the wire shape (or vice versa), we ask the
miner to specify the translation explicitly. This keeps graphs
reusable across families and the wire contract stable across SDK
changes.

Streaming
---------

:meth:`GraphAgent.infer_stream` delegates to the graph's
:func:`~eirel.graph.runtime.stream_events` and forwards every chunk
through. Today it emits a single ``done`` chunk; later milestones
will yield intermediate ``delta``/``tool_call``/``citation`` chunks
as nodes support them. The terminal-``done`` invariant is preserved.
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator, Callable
from typing import Any

from eirel.base_agent import BaseAgent
from eirel.graph.graph import CompiledGraph
from eirel.graph.runtime import (
    RunConfig,
    RunContext,
    _InterruptSignal,
    run_to_completion,
    stream_events,
)
from eirel.schemas import (
    AgentCapabilityMetadata,
    AgentInvocationRequest,
    AgentInvocationResponse,
    StreamChunk,
)

__all__ = ["GraphAgent"]


ToStateFn = Callable[[AgentInvocationRequest], dict[str, Any]]
FromStateFn = Callable[[dict[str, Any], AgentInvocationRequest], AgentInvocationResponse]
ContextFactory = Callable[[AgentInvocationRequest], RunContext | None]


class GraphAgent(BaseAgent):
    """:class:`BaseAgent` whose ``infer`` is driven by a compiled graph.

    Parameters
    ----------
    hotkey, endpoint, version, capabilities
        Standard :class:`BaseAgent` registration metadata.
    graph
        The compiled state graph.
    to_state
        Maps an incoming :class:`AgentInvocationRequest` to the graph's
        initial state dict. Should consume ``request.prompt``,
        ``request.history``, ``request.mode``, etc.
    from_state
        Maps the graph's final state dict (plus the original request)
        to an :class:`AgentInvocationResponse`. Owns surfacing the
        answer text into ``output``, citations into ``citations``, etc.
    run_context_factory
        Optional. Builds a :class:`RunContext` per request — wire up
        :class:`BudgetTracker`, :class:`TraceRecorder`, etc., here.
        Default: a fresh empty context.
    """

    def __init__(
        self,
        *,
        hotkey: str,
        endpoint: str,
        version: str,
        capabilities: AgentCapabilityMetadata,
        graph: CompiledGraph,
        to_state: ToStateFn,
        from_state: FromStateFn,
        run_context_factory: ContextFactory | None = None,
    ) -> None:
        super().__init__(
            hotkey=hotkey,
            endpoint=endpoint,
            version=version,
            capabilities=capabilities,
        )
        if not isinstance(graph, CompiledGraph):
            raise TypeError(
                f"GraphAgent expected a CompiledGraph, got {type(graph).__name__}"
            )
        if not callable(to_state) or not callable(from_state):
            raise TypeError("to_state and from_state must be callable")
        self._graph = graph
        self._to_state = to_state
        self._from_state = from_state
        self._run_context_factory = run_context_factory

    @property
    def graph(self) -> CompiledGraph:
        return self._graph

    def _build_run_config(self, request: AgentInvocationRequest) -> RunConfig:
        return RunConfig(
            thread_id=request.turn_id or request.task_id,
            resume_token=request.resume_token,
            checkpoint_id=None,
            metadata=dict(request.metadata or {}),
        )

    def _build_run_context(self, request: AgentInvocationRequest) -> RunContext:
        config = self._build_run_config(request)
        if self._run_context_factory is not None:
            ctx = self._run_context_factory(request)
            if ctx is None:
                return RunContext(config=config)
            # Caller-supplied context: preserve their handles, override
            # config so downstream code can rely on a populated thread_id
            # without the factory having to plumb it through manually.
            return RunContext(
                config=config,
                budget=ctx.budget,
                run_cost=ctx.run_cost,
                trace=ctx.trace,
                tracer=ctx.tracer,
                job_id=ctx.job_id or config.thread_id,
                extras=dict(ctx.extras),
            )
        return RunContext(config=config)

    async def infer(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        state = self._to_state(request)
        config = self._build_run_config(request)
        ctx = self._build_run_context(request)
        try:
            final = await run_to_completion(
                self._graph, state, config=config, context=ctx
            )
        except _InterruptSignal as exc:
            return _build_deferred_response(request, exc)
        return self._from_state(final, request)

    async def infer_stream(
        self, request: AgentInvocationRequest
    ) -> AsyncIterator[StreamChunk]:
        state = self._to_state(request)
        config = self._build_run_config(request)
        ctx = self._build_run_context(request)
        async for chunk in stream_events(
            self._graph, state, config=config, context=ctx
        ):
            yield chunk


def _build_deferred_response(
    request: AgentInvocationRequest, signal: _InterruptSignal
) -> AgentInvocationResponse:
    """Build a deferred AgentInvocationResponse from an interrupt signal.

    Issues a fresh resume token signed with ``EIREL_RESUME_TOKEN_SECRET``
    when the secret is configured; otherwise leaves ``resume_token`` empty
    and the validator must look up the checkpoint via ``thread_id`` directly.
    """
    secret = os.getenv("EIREL_RESUME_TOKEN_SECRET", "")
    resume_token: str | None = None
    if secret and signal.thread_id and signal.checkpoint_id:
        from eirel.checkpoint.resume import encode_thread_token

        resume_token = encode_thread_token(
            thread_id=signal.thread_id,
            checkpoint_id=signal.checkpoint_id,
            secret=secret,
        )
    return AgentInvocationResponse(
        task_id=request.turn_id or request.task_id,
        family_id=request.family_id,
        status="deferred",
        output={},
        resume_token=resume_token,
        metadata={
            "interrupt": {
                "node": signal.node,
                "payload": dict(signal.payload),
                "thread_id": signal.thread_id,
                "checkpoint_id": signal.checkpoint_id,
            }
        },
    )

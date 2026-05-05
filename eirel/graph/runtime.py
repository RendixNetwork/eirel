"""Async executor for compiled state graphs.

The executor walks the :class:`~eirel.graph.compile.ExecutionPlan` from
entry to ``END``, applying each node's partial-state update via the
graph's :class:`~eirel.graph.state.StateSpec` reducers.

Three edge shapes are handled:

  * **Static edges** (``src -> dst``): run ``src``, follow to ``dst``.
  * **Conditional edges**: run ``src``, then evaluate the router. The
    router accepts only single-target returns today — runtime fan-out
    is reserved for declared parallel edges.
  * **Parallel edges**: run ``src``, then concurrently run all ``dsts``,
    await all branches, merge their state updates in declared order,
    finally run ``join``.

A :class:`RunContext` is bound in a ``ContextVar`` for the duration of a
run so nodes can read shared per-run handles (budget tracker, run cost
tracker, trace recorder, tracer, job id) without threading them through
every call site.

Streaming
---------

:func:`stream_events` yields :class:`StreamChunk` events using only the
existing wire-stable event vocabulary (``delta``/``tool_call``/``citation``/
``done``). Today it emits a single ``done`` chunk carrying the final
state keys we know how to surface; per-token deltas inside an LLM node
depend on the provider proxy supporting SSE pass-through and are a
later enhancement. The shape stays forward-compatible: more granular
events can be added without changing the contract.
"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from eirel.graph.edge import END, START
from eirel.schemas import StreamChunk

if TYPE_CHECKING:
    from eirel.graph.compile import ExecutionPlan
    from eirel.graph.graph import CompiledGraph
    from eirel.graph.state import StateSpec

__all__ = [
    "RunConfig",
    "RunContext",
    "Interrupt",
    "GuardDeniedError",
    "current_run_context",
    "run_to_completion",
    "stream_events",
    "GraphRecursionError",
]


class GuardDeniedError(RuntimeError):
    """Raised internally when ``pre_input`` or ``post_output`` denies.

    The streaming helper surfaces this as a ``done`` chunk with
    ``status="failed"`` so the caller never sees graph state from a
    rejected request.
    """

    def __init__(self, *, stage: str, reason: str, metadata: dict[str, Any] | None = None) -> None:
        super().__init__(f"guard denied at {stage}: {reason}")
        self.stage = stage
        self.reason = reason
        self.metadata = dict(metadata or {})


class GraphRecursionError(RuntimeError):
    """Raised when a graph exceeds its declared ``recursion_limit``."""

    def __init__(self, limit: int, last_node: str) -> None:
        super().__init__(
            f"graph exceeded recursion_limit={limit} at node {last_node!r}"
        )
        self.limit = limit
        self.last_node = last_node


class Interrupt(Exception):
    """Raise inside a node to pause the graph and emit a resume token.

    The runtime catches this, persists the current state via the
    configured :class:`~eirel.checkpoint.base.Checkpointer`, and surfaces
    a clean ``status="deferred"`` outcome to the caller. The next turn
    can pick up by passing the matching resume token.

    ``payload`` is propagated into the checkpoint metadata so the next
    turn (and any human reviewer in the loop) can inspect what the
    graph was waiting on.
    """

    def __init__(self, *, node: str, payload: dict[str, Any] | None = None) -> None:
        super().__init__(f"graph interrupted at node {node!r}")
        self.node = node
        self.payload = dict(payload or {})


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Per-invocation knobs passed alongside the initial state.

    ``thread_id`` and ``resume_token`` are load-bearing. The runtime
    uses them to (a) namespace checkpoints, (b) hydrate state on resume.
    """

    thread_id: str | None = None
    resume_token: str | None = None
    checkpoint_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunContext:
    """Per-run handles shared across all nodes via a ``ContextVar``.

    Concrete types (``BudgetTracker``, ``RunCostTracker``,
    ``TraceRecorder``, tracer, ``Guard``) are layered in via the
    runtime's own wiring. Nodes read what they need; absent fields
    stay ``None``.
    """

    config: RunConfig
    budget: Any = None
    run_cost: Any = None
    trace: Any = None
    tracer: Any = None
    job_id: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


_RUN_CONTEXT: ContextVar[RunContext | None] = ContextVar(
    "eirel_graph_run_context", default=None
)


def current_run_context() -> RunContext | None:
    """Return the active :class:`RunContext`, if any."""
    return _RUN_CONTEXT.get()


# -- Public entry points ------------------------------------------------------


async def run_to_completion(
    graph: "CompiledGraph",
    state: dict[str, Any],
    *,
    config: RunConfig | None = None,
    context: RunContext | None = None,
) -> dict[str, Any]:
    """Run the graph to ``END`` and return the final state.

    When the configured checkpointer is non-None and ``config.thread_id``
    is set, the runtime persists a checkpoint after each node. If
    ``config.resume_token``/``checkpoint_id`` resolves to a stored
    checkpoint, the run starts from there instead of the supplied
    ``state`` (the supplied dict is treated as a fallback for fresh runs).
    """
    cfg = config or RunConfig()
    ctx = context or RunContext(config=cfg)
    token = _RUN_CONTEXT.set(ctx)
    try:
        state, start_node = await _hydrate_from_checkpoint(graph, state, cfg)
        # Pre-input guard runs only on a fresh start (start_node is None);
        # on resume the input has already been guarded once.
        if start_node is None:
            state = await _run_pre_input_guard(graph, state)
        final = await _walk(graph, state, start_node=start_node, config=cfg)
        final = await _run_post_output_guard(graph, final)
        return final
    finally:
        _RUN_CONTEXT.reset(token)


async def stream_events(
    graph: "CompiledGraph",
    state: dict[str, Any],
    *,
    config: RunConfig | None = None,
    context: RunContext | None = None,
) -> AsyncIterator[StreamChunk]:
    """Run the graph and yield :class:`StreamChunk` events.

    Still emits a single terminal ``done`` chunk; on
    :class:`Interrupt`, the chunk carries ``status="deferred"`` plus a
    resume token in metadata so the validator can pick up next turn.
    Token-level deltas are a later enhancement.
    """
    cfg = config or RunConfig()
    ctx = context or RunContext(config=cfg)
    token = _RUN_CONTEXT.set(ctx)
    try:
        try:
            state, start_node = await _hydrate_from_checkpoint(graph, state, cfg)
            if start_node is None:
                state = await _run_pre_input_guard(graph, state)
            final = await _walk(graph, state, start_node=start_node, config=cfg)
            final = await _run_post_output_guard(graph, final)
        except _InterruptSignal as exc:
            yield StreamChunk(
                event="done",
                status="deferred",
                output={},
                citations=[],
                metadata={
                    "interrupt": {
                        "node": exc.node,
                        "payload": dict(exc.payload),
                        "thread_id": exc.thread_id,
                        "checkpoint_id": exc.checkpoint_id,
                    },
                },
            )
            return
        except GuardDeniedError as exc:
            yield StreamChunk(
                event="done",
                status="failed",
                error=f"guard_denied: {exc.reason}",
                metadata={
                    "guard": {
                        "stage": exc.stage,
                        "reason": exc.reason,
                        **dict(exc.metadata or {}),
                    }
                },
            )
            return
        except Exception as exc:  # noqa: BLE001
            yield StreamChunk(event="done", status="failed", error=str(exc))
            return
        output, citations, metadata = _surface_final_state(final)
        yield StreamChunk(
            event="done",
            output=output,
            citations=citations,
            status="completed",
            metadata=metadata,
        )
    finally:
        _RUN_CONTEXT.reset(token)


# -- Walker -------------------------------------------------------------------


class _InterruptSignal(Exception):
    """Internal signal: ``Interrupt`` after the runtime has captured a checkpoint.

    Carries the persisted ``thread_id``/``checkpoint_id`` so the streaming
    helper can surface a resume token without a second checkpoint round-trip.
    """

    def __init__(
        self,
        *,
        node: str,
        payload: dict[str, Any],
        thread_id: str | None,
        checkpoint_id: str | None,
    ) -> None:
        super().__init__(f"graph interrupted at {node!r}")
        self.node = node
        self.payload = payload
        self.thread_id = thread_id
        self.checkpoint_id = checkpoint_id


async def _walk(
    graph: "CompiledGraph",
    state: dict[str, Any],
    *,
    start_node: str | None = None,
    config: RunConfig,
) -> dict[str, Any]:
    """Execute the plan from START (or ``start_node`` on resume) to END.

    Static edges: linear progression. Conditional edges: route after the
    node runs. Parallel edges: fan-out → join with state-update merging
    in declared destination order.

    After each node executes successfully, the runtime persists a
    :class:`~eirel.checkpoint.base.CheckpointTuple` if a checkpointer is
    attached and ``config.thread_id`` is set. ``Interrupt`` raised inside
    a node is captured *after* the pre-node checkpoint write so resume
    starts at the interrupted node.
    """
    plan = graph.plan
    spec = graph.state_spec
    tracer = graph.tracer
    current = start_node if start_node is not None else plan.static_edges[START]
    steps = 0
    limit = plan.recursion_limit if plan.recursion_limit > 0 else 1
    parent_checkpoint_id: str | None = config.checkpoint_id
    while current != END:
        steps += 1
        if steps > limit:
            raise GraphRecursionError(limit=plan.recursion_limit, last_node=current)

        # Persist a *pre-node* checkpoint so a resume after Interrupt
        # re-runs the interrupted node from the same input state.
        parent_checkpoint_id = await _maybe_checkpoint(
            graph, config, state, node=current, parent_id=parent_checkpoint_id
        )

        if current in plan.parallel_edges:
            par = plan.parallel_edges[current]
            try:
                state = await _run_node(plan, spec, current, state, tracer=tracer)
                state = await _run_parallel(plan, spec, par.dsts, state, tracer=tracer)
            except Interrupt as exc:
                if tracer is not None:
                    tracer.event(
                        "graph.interrupt",
                        attrs={"node": exc.node, "payload": dict(exc.payload)},
                    )
                raise _InterruptSignal(
                    node=exc.node,
                    payload=exc.payload,
                    thread_id=config.thread_id,
                    checkpoint_id=parent_checkpoint_id,
                ) from exc
            current = par.join
            continue

        try:
            state = await _run_node(plan, spec, current, state, tracer=tracer)
        except Interrupt as exc:
            if tracer is not None:
                tracer.event(
                    "graph.interrupt",
                    attrs={"node": exc.node, "payload": dict(exc.payload)},
                )
            raise _InterruptSignal(
                node=exc.node,
                payload=exc.payload,
                thread_id=config.thread_id,
                checkpoint_id=parent_checkpoint_id,
            ) from exc

        if current in plan.conditional_edges:
            cond = plan.conditional_edges[current]
            destinations = cond.resolve(state)
            if len(destinations) != 1:
                raise NotImplementedError(
                    f"conditional router for {current!r} returned {len(destinations)} destinations; "
                    "runtime fan-out is reserved for declared parallel edges"
                )
            current = destinations[0]
            continue

        if current in plan.static_edges:
            current = plan.static_edges[current]
            continue

        # Implicit finish: graph ended at a terminal node without an edge.
        if plan.finish is not None and current == plan.finish:
            current = END
            continue

        raise RuntimeError(
            f"node {current!r} has no outbound edge — should have been caught at compile time"
        )

    return state


async def _hydrate_from_checkpoint(
    graph: "CompiledGraph",
    state: dict[str, Any],
    config: RunConfig,
) -> tuple[dict[str, Any], str | None]:
    """If config carries a resume token / checkpoint_id, load that state.

    Returns ``(state, start_node)`` — when no resume info is provided,
    falls through with the caller's ``state`` and ``start_node=None``
    (executor enters at the graph's entry point).
    """
    checkpointer = graph.checkpointer
    if checkpointer is None or not config.thread_id:
        return state, None

    target_id: str | None = config.checkpoint_id
    if target_id is None and config.resume_token:
        from eirel.checkpoint.resume import (
            InvalidResumeToken,
            decode_thread_token,
        )
        import os

        secret = os.getenv("EIREL_RESUME_TOKEN_SECRET", "")
        if secret:
            try:
                payload = decode_thread_token(config.resume_token, secret)
                if payload.get("thread_id") == config.thread_id:
                    target_id = str(payload.get("checkpoint_id"))
            except InvalidResumeToken:
                # Bad token: fall through to a fresh run rather than
                # leaking the verification failure as a 500.
                target_id = None

    if target_id is None:
        return state, None

    tup = await checkpointer.aget(config.thread_id, target_id)
    if tup is None:
        return state, None

    return dict(tup.state), tup.node


async def _maybe_checkpoint(
    graph: "CompiledGraph",
    config: RunConfig,
    state: dict[str, Any],
    *,
    node: str,
    parent_id: str | None,
) -> str | None:
    """Persist a pre-node checkpoint when configured. Returns the new id."""
    checkpointer = graph.checkpointer
    if checkpointer is None or not config.thread_id:
        return parent_id

    from eirel.checkpoint.base import new_checkpoint_id

    new_id = new_checkpoint_id()
    await checkpointer.aput(
        thread_id=config.thread_id,
        checkpoint_id=new_id,
        parent_id=parent_id,
        node=node,
        state=state,
        metadata={"about_to_run": node},
    )
    return new_id


async def _run_node(
    plan: "ExecutionPlan",
    spec: "StateSpec",
    name: str,
    state: dict[str, Any],
    *,
    tracer: "Any" = None,
) -> dict[str, Any]:
    """Execute one node and apply its partial-state update via spec reducers.

    When a tracer is supplied, the call is bracketed by ``span_start`` /
    ``span_end`` so external observability can reconstruct node-level
    latencies and error rates.
    """
    node = plan.nodes[name]
    span = tracer.span_start(f"node:{name}", attrs={"node": name}) if tracer is not None else None
    try:
        update = await node(state)
    except Exception as exc:  # noqa: BLE001
        if tracer is not None and span is not None:
            tracer.span_end(span, status="error", attrs={"error": str(exc)})
        raise
    if update is None:
        if tracer is not None and span is not None:
            tracer.span_end(span, status="ok", attrs={"updated_keys": []})
        return state
    if not isinstance(update, dict):
        raise TypeError(
            f"node {name!r} returned {type(update).__name__}; expected dict | None"
        )
    if tracer is not None and span is not None:
        tracer.span_end(span, status="ok", attrs={"updated_keys": sorted(update.keys())})
    return spec.merge(state, update)


async def _run_parallel(
    plan: "ExecutionPlan",
    spec: "StateSpec",
    dsts: tuple[str, ...],
    state: dict[str, Any],
    *,
    tracer: "Any" = None,
) -> dict[str, Any]:
    """Run ``dsts`` concurrently, merging their updates in declared order.

    Each branch sees the same starting ``state`` (snapshot) and returns a
    partial update dict. We then fold those updates into ``state`` in
    declared order — left-to-right — so reducers see a deterministic
    sequence even though branches ran concurrently. If any branch raises,
    pending sibling branches are cancelled and the exception propagates.
    """
    snapshot = dict(state)

    async def branch(name: str) -> dict[str, Any] | None:
        node = plan.nodes[name]
        if tracer is None:
            return await node(snapshot)
        span = tracer.span_start(f"node:{name}", attrs={"node": name, "parallel": True})
        try:
            result = await node(snapshot)
        except Exception as exc:  # noqa: BLE001
            tracer.span_end(span, status="error", attrs={"error": str(exc)})
            raise
        tracer.span_end(
            span,
            status="ok",
            attrs={"updated_keys": sorted(result.keys()) if isinstance(result, dict) else []},
        )
        return result

    tasks = [asyncio.create_task(branch(name), name=f"graph-parallel:{name}") for name in dsts]
    try:
        results = await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        # Wait for cancellations to settle so we don't leak tasks.
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    merged = state
    for update in results:
        if update is None:
            continue
        if not isinstance(update, dict):
            raise TypeError(
                f"parallel node returned {type(update).__name__}; expected dict | None"
            )
        merged = spec.merge(merged, update)
    return merged


async def _run_pre_input_guard(
    graph: "CompiledGraph",
    state: dict[str, Any],
) -> dict[str, Any]:
    """Run the configured Guard's pre_input hook; merge redactions, deny."""
    guard = graph.safety
    if guard is None:
        return state
    verdict = await guard.pre_input(state)
    tracer = graph.tracer
    if tracer is not None:
        tracer.event(
            "guard.pre_input",
            attrs={
                "allow": verdict.allow,
                "reason": verdict.reason,
                **dict(verdict.metadata or {}),
            },
        )
    if not verdict.allow:
        raise GuardDeniedError(
            stage="pre_input",
            reason=verdict.reason or "denied",
            metadata=dict(verdict.metadata or {}),
        )
    if verdict.redactions:
        state = graph.state_spec.merge(state, dict(verdict.redactions))
    return state


async def _run_post_output_guard(
    graph: "CompiledGraph",
    state: dict[str, Any],
) -> dict[str, Any]:
    """Run the configured Guard's post_output hook; merge redactions, deny."""
    guard = graph.safety
    if guard is None:
        return state
    verdict = await guard.post_output(state)
    tracer = graph.tracer
    if tracer is not None:
        tracer.event(
            "guard.post_output",
            attrs={
                "allow": verdict.allow,
                "reason": verdict.reason,
                **dict(verdict.metadata or {}),
            },
        )
    if not verdict.allow:
        raise GuardDeniedError(
            stage="post_output",
            reason=verdict.reason or "denied",
            metadata=dict(verdict.metadata or {}),
        )
    if verdict.redactions:
        state = graph.state_spec.merge(state, dict(verdict.redactions))
    return state


# -- Final-state surfacing ----------------------------------------------------


def _surface_final_state(
    state: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    """Best-effort extraction of an output/citations/metadata triple.

    Mirrors :class:`~eirel.base_agent.BaseAgent`'s default unary→stream
    adapter so a graph-built agent emits the same wire shape as a
    classic ``BaseAgent`` subclass.
    """
    if not isinstance(state, dict):  # defensive
        return {}, [], {}
    output: dict[str, Any] = {}
    raw_output = state.get("output")
    if isinstance(raw_output, dict):
        output = dict(raw_output)
    else:
        # Fallback: if a graph exposes a final answer under "answer" or
        # the last assistant message, surface that as output["answer"].
        answer = state.get("answer")
        if isinstance(answer, str) and answer:
            output["answer"] = answer
        else:
            messages = state.get("messages")
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        content = msg.get("content")
                        if isinstance(content, str) and content:
                            output["answer"] = content
                            break

    citations: list[str] = []
    raw_citations = state.get("citations")
    if isinstance(raw_citations, list):
        for c in raw_citations:
            if isinstance(c, str):
                citations.append(c)
            elif isinstance(c, dict) and isinstance(c.get("url"), str):
                citations.append(c["url"])

    metadata: dict[str, Any] = {}
    raw_meta = state.get("metadata")
    if isinstance(raw_meta, dict):
        metadata = dict(raw_meta)

    return output, citations, metadata

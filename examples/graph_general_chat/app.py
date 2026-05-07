"""Reference miner: ``general_chat`` family agent built as a state graph.

Mirrors ``examples/general_chat_agent/app.py`` but composed via the new
``eirel.graph`` primitives. The graph layout is intentionally simple to
showcase the building blocks; real miners can grow it (planner +
self-critique loop, parallel tool dispatch, structured output, etc.)
without changing the wire contract.

Graph topology
--------------

  prepare ──► [web_search?] ──► llm ──► finalize ──► END

* ``prepare`` builds the message list from request prompt + history,
  initializes the per-turn ``BudgetTracker`` / ``RunCostTracker`` /
  ``TraceRecorder`` and stashes them on the ``RunContext`` so downstream
  nodes (and any tools they call) share the same accounting.
* ``web_search`` only runs when the request enables it; it executes the
  tool service via the family catalog and injects a system note with
  the results before the LLM call.
* ``llm`` calls the configured provider with the accumulated messages.
* ``finalize`` lifts the assistant reply into the response shape
  (``answer`` + ``citations`` + ``executed_tool_calls``).

Provider / model configuration is identical to the classic example
(see ``MinerProviderConfig.from_env()`` and the manifest's ``inference``
block); the only manifest delta is ``runtime.kind: graph`` so eirel-ai
knows the new envelope is in play.
"""
from __future__ import annotations

import os
from typing import Any

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
    merge_dict,
    replace,
)
from eirel.families.general_chat import (
    BudgetTracker,
    Citation,
    GeneralChatContext,
    GeneralChatToolCatalog,
    RunBudget,
    RunCostTracker,
    SandboxTool,
    TraceRecorder,
    WebSearchTool,
    context_from_request,
)
from eirel.families.general_chat.tools._service_client import (
    ToolServiceClient,
    ToolServiceConfig,
)
from eirel.graph import END
from eirel.provider import AgentProviderClient, MinerProviderConfig

_PROVIDER_CONFIG = MinerProviderConfig.from_env()
_PROVIDER_CONFIG.validate_for_runtime()
_PROVIDER_CLIENT = AgentProviderClient(_PROVIDER_CONFIG)


# -- Graph state shape --------------------------------------------------------


_STATE_SPEC = StateSpec({
    # OpenAI-style chat list. Reduced via add_messages so any node can
    # append (system notes from web_search, the LLM's final reply, etc.).
    "messages": StateField(reducer=add_messages, default_factory=list),
    # The latest user message — derived once in ``prepare`` so downstream
    # nodes don't have to scan the messages list.
    "user_prompt": StateField(reducer=replace, default=""),
    # Mode + flags from the request, surfaced into state for the LLM payload.
    "mode": StateField(reducer=replace, default="instant"),
    "web_search_enabled": StateField(reducer=replace, default=False),
    "max_tokens": StateField(reducer=replace, default=2048),
    # Web-search results (raw); the web_search node populates this and
    # the LLM node injects them into the prompt.
    "web_results": StateField(reducer=replace, default_factory=list),
    # Citations the agent gathered (URLs).
    "citations": StateField(reducer=add_messages, default_factory=list),
    # Executed tool calls (recorded via TraceRecorder, surfaced into metadata).
    "executed_tool_calls": StateField(reducer=add_messages, default_factory=list),
    # Final answer text.
    "answer": StateField(reducer=replace, default=""),
    # Routing decision after prepare: "web_search" or "llm".
    "next": StateField(reducer=replace, default=""),
    # Free-form scratchpad for nodes that want to share extra info.
    "scratchpad": StateField(reducer=merge_dict, default_factory=dict),
    # Wire-level metadata that the runtime's ``_surface_final_state``
    # lifts onto the streaming response. ``finalize`` populates
    # ``executed_tool_calls`` here so the validator's invocation parser
    # surfaces them as ``response.tool_calls`` with full {url, title,
    # snippet} per result — making titles render on the dashboard.
    "metadata": StateField(reducer=merge_dict, default_factory=dict),
})


# -- Tool catalog factory -----------------------------------------------------


def _service_client(env_url: str, *, job_id: str | None, env_token: str | None = None) -> ToolServiceClient:
    base_url = os.getenv(env_url, "")
    api_token = ""
    if env_token:
        api_token = os.getenv(env_token, "")
    if not api_token:
        api_token = os.getenv("EIREL_TOOL_SERVICE_TOKEN", "")
    return ToolServiceClient(
        ToolServiceConfig(base_url=base_url, api_token=api_token, job_id=job_id)
    )


def _build_catalog(job_id: str, budget: BudgetTracker, trace: TraceRecorder) -> GeneralChatToolCatalog:
    tools = [
        WebSearchTool(_service_client("EIREL_WEB_SEARCH_URL", job_id=job_id, env_token="EIREL_WEB_SEARCH_TOKEN")),
        SandboxTool(_service_client("EIREL_SANDBOX_URL", job_id=job_id, env_token="EIREL_SANDBOX_TOKEN")),
    ]
    return GeneralChatToolCatalog(tools, budget=budget, trace=trace)


# -- Graph nodes --------------------------------------------------------------


def _last_user_message(context: GeneralChatContext, request: AgentInvocationRequest) -> str:
    if request.prompt:
        return request.prompt
    for turn in reversed(context.conversation_history):
        if turn.role == "user" and turn.content:
            return turn.content
    return request.primary_goal or request.subtask or ""


async def prepare(state: dict[str, Any]) -> dict[str, Any]:
    """Seed messages + decide whether to run web_search."""
    context: GeneralChatContext = state["scratchpad"]["context"]
    request: AgentInvocationRequest = state["scratchpad"]["request"]
    user_prompt = _last_user_message(context, request)
    history = [
        {"role": turn.role, "content": turn.content}
        for turn in context.conversation_history
    ]
    if history and history[-1].get("role") == "user" and history[-1].get("content") == user_prompt:
        messages = list(history)
    else:
        messages = list(history) + [{"role": "user", "content": user_prompt}]

    max_tokens = max(2048, context.budget.output_tokens + context.budget.reasoning_tokens)
    return {
        "messages": messages,
        "user_prompt": user_prompt,
        "mode": context.mode,
        "web_search_enabled": context.web_search_enabled,
        "max_tokens": max_tokens,
        "next": "web_search" if context.web_search_enabled and user_prompt else "llm",
    }


def _extract_web_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    for key in ("documents", "results", "data", "items"):
        value = raw.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]
    return []


async def web_search(state: dict[str, Any]) -> dict[str, Any]:
    """Run web_search via the catalog and append a system note with results."""
    catalog: GeneralChatToolCatalog = state["scratchpad"]["catalog"]
    trace: TraceRecorder = state["scratchpad"]["trace"]
    user_prompt = state["user_prompt"]

    try:
        raw = await catalog.execute("web_search", query=user_prompt, max_results=5)
    except Exception as exc:  # noqa: BLE001 — tool failure shouldn't kill the graph
        trace.set_metadata("web_search_error", str(exc))
        return {"web_results": []}

    results = _extract_web_results(raw)
    citations: list[str] = []
    snippets: list[str] = []
    for idx, item in enumerate(results, start=1):
        url = str(item.get("url") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or item.get("content") or "").strip()
        if url:
            trace.record_citation(
                Citation(url=url, title=title or None, snippet=snippet or None, tool_name="web_search")
            )
            citations.append(url)
        snippets.append(f"[{idx}] {title} ({url})\n{snippet}" if url or title else snippet)

    update: dict[str, Any] = {"web_results": results, "citations": citations}
    if snippets:
        update["messages"] = {
            "role": "system",
            "content": (
                "You have access to the following fresh web search results. "
                "When relevant, incorporate them into your answer and cite the "
                "source URLs inline as [1], [2], etc.\n\n" + "\n\n".join(snippets)
            ),
        }
    return update


async def llm(state: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "messages": list(state["messages"]),
        "max_tokens": state["max_tokens"],
    }
    response = await _PROVIDER_CLIENT.chat_completions(payload)
    content = ""
    try:
        content = response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        content = ""
    return {
        "messages": {"role": "assistant", "content": content},
        "answer": content,
    }


async def finalize(state: dict[str, Any]) -> dict[str, Any]:
    """Snapshot trace state into the executed_tool_calls/citations fields.

    Two outputs are wire-relevant:
    * ``citations`` — bare URL list used by older consumers.
    * ``metadata.executed_tool_calls`` — richer per-tool record carrying
      ``{url, title, snippet}`` for each search hit, grouped by
      ``tool_name``. The validator's invocation parser lifts this from
      response metadata into the top-level ``tool_calls`` payload, where
      its citation extractor reads ``{url, title}`` per result. This is
      what makes citation titles render on the dashboard.
    """
    trace: TraceRecorder = state["scratchpad"]["trace"]
    frozen = trace.freeze()

    # Group citations by their originating tool so each tool_call entry
    # carries the right {tool_name, result.results} shape the validator
    # extractor (``_extract_miner_citations``) walks for {url, title}.
    by_tool: dict[str, list[dict[str, Any]]] = {}
    for c in frozen.citations:
        by_tool.setdefault(c.tool_name, []).append({
            "url": c.url,
            "title": c.title or "",
            "snippet": c.snippet or "",
        })
    executed_tool_calls: list[dict[str, Any]] = [
        {"tool_name": tool_name, "result": {"results": results}}
        for tool_name, results in by_tool.items()
    ]

    return {
        "executed_tool_calls": executed_tool_calls,
        "citations": [c.url for c in frozen.citations],
        # Bridge into state.metadata so the runtime's _surface_final_state
        # forwards it to response.metadata.executed_tool_calls.
        "metadata": {"executed_tool_calls": executed_tool_calls},
    }


# -- Graph assembly -----------------------------------------------------------


def _build_graph():
    g = StateGraph(_STATE_SPEC)
    g.add_node("prepare", prepare)
    g.add_node("web_search", web_search)
    g.add_node("llm", llm)
    g.add_node("finalize", finalize)
    g.add_conditional_edges(
        "prepare",
        router=lambda s: s["next"],
        mapping={"web_search": "web_search", "llm": "llm"},
    )
    g.add_edge("web_search", "llm")
    g.add_edge("llm", "finalize")
    g.add_edge("finalize", END)
    g.set_entry_point("prepare")
    return g.compile()


_COMPILED_GRAPH = _build_graph()


# -- Wire ↔ state mappers -----------------------------------------------------


def _to_state(request: AgentInvocationRequest) -> dict[str, Any]:
    """Build initial state and stash a per-request scratchpad bundle.

    The scratchpad carries the per-turn handles (context, catalog, trace,
    budget) that the graph nodes need but that don't belong on the wire.
    Storing them in state keeps the nodes pure async functions of state
    without leaking these objects into ``RunContext`` — Tracer/Guard
    wiring on the context layer is a later enhancement.
    """
    context = context_from_request(request.model_dump(mode="json"))
    budget = BudgetTracker(budget=context.budget)
    run_budget_usd = float(os.getenv("EIREL_RUN_BUDGET_USD", "30.0"))
    run_cost = RunCostTracker(budget=RunBudget(max_usd=run_budget_usd))
    trace = TraceRecorder()

    job_id = (
        os.getenv("EIREL_PROVIDER_PROXY_JOB_ID")
        or request.turn_id
        or request.task_id
        or "unknown-job"
    )
    catalog = _build_catalog(job_id, budget, trace)

    return _STATE_SPEC.init(scratchpad={
        "context": context,
        "request": request,
        "budget": budget,
        "run_cost": run_cost,
        "trace": trace,
        "catalog": catalog,
        "job_id": job_id,
    })


def _from_state(state: dict[str, Any], request: AgentInvocationRequest) -> AgentInvocationResponse:
    context: GeneralChatContext = state["scratchpad"]["context"]
    return AgentInvocationResponse(
        task_id=request.turn_id or request.task_id,
        family_id=request.family_id or "general_chat",
        status="completed",
        output={"answer": state["answer"]},
        citations=list(state.get("citations") or []),
        metadata={
            "mode": context.mode,
            "web_search": context.web_search_enabled,
            "executed_tool_calls": list(state.get("executed_tool_calls") or []),
        },
    )


# -- Agent + app --------------------------------------------------------------


agent = GraphAgent(
    hotkey=os.getenv("EIREL_HOTKEY", "5HotkeyPlaceholder"),
    endpoint=os.getenv("EIREL_ENDPOINT", "http://localhost:8000"),
    version="0.1.0",
    capabilities=AgentCapabilityMetadata(
        family_id="general_chat",
        description="Graph-based general chat agent (reference miner)",
        supports_streaming=True,
        latency_ms_p50=1500,
    ),
    graph=_COMPILED_GRAPH,
    to_state=_to_state,
    from_state=_from_state,
)


# ``build_agent_app`` knows how to serve any ``BaseAgent`` — including
# ``GraphAgent`` — via the standard ``/v1/agent/infer[/stream]``
# endpoints with Hotkey-signed inbound auth. No bespoke wiring needed.
app = build_agent_app(agent)

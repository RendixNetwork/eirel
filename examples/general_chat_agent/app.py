"""General chat agent example.

Demonstrates the ``general_chat`` family with the four user-selectable
modes (``instant`` / ``thinking`` × web_search off/on).  Tool calls are
routed through the owner-api tool services for web search, X, Semantic
Scholar, and a server-side Python sandbox for verifiable computation.

The handler in this example is intentionally minimal: it builds a
:class:`GeneralChatContext`, wires up a :class:`GeneralChatToolCatalog`
with the four available tools, and synthesizes a single-turn response
by calling the underlying LLM provider.  Real miners should replace
the body of ``_run_turn`` with their own orchestration logic.

Provider / model configuration
------------------------------

The LLM provider and model are NOT set in this file.  They are declared
in ``submission.yaml`` next to the agent code, e.g.::

    inference:
      model: gpt-4.1-mini
      providers:
        - openai

At deployment time the owner-api reads the manifest and injects env
vars into the runtime pod:

    MINER_PROVIDER         <- manifest.inference.providers[0]
    MINER_MODEL            <- manifest.inference.model
    PROVIDER_PROXY_URL     <- subnet-operated LLM proxy (forced)
    PROVIDER_PROXY_TOKEN   <- proxy auth token        (forced)

``MinerProviderConfig.from_env()`` picks these up automatically.  In
production, all LLM traffic is routed through the subnet provider-proxy
— miners never hold raw API keys, and per-run USD budgets are enforced
at the proxy layer.

For local testing (``eirel serve`` or ``python app.py``) you can bypass
the proxy by setting ``MINER_API_KEY`` directly; the SDK will fall back
to direct provider calls when no proxy URL is configured.
"""
from __future__ import annotations

import os
from typing import Any

from eirel.app import MinerApp
from eirel.families.general_chat import (
    BudgetTracker,
    Citation,
    ConversationTurn,
    GeneralChatContext,
    GeneralChatResponse,
    GeneralChatToolCatalog,
    RunBudget,
    RunCostTracker,
    SandboxTool,
    SemanticScholarTool,
    TraceRecorder,
    WebSearchTool,
    XApiTool,
    context_from_request,
)
from eirel.families.general_chat.tools._service_client import (
    ToolServiceClient,
    ToolServiceConfig,
)
from eirel.provider import AgentProviderClient, MinerProviderConfig
from eirel.schemas import AgentInvocationRequest, AgentInvocationResponse


# Resolve provider config once at import time.  ``validate_for_runtime()``
# fails fast if required env vars are missing — the pod will crash at boot
# rather than surface a cryptic 401 on the first validator request.
_PROVIDER_CONFIG = MinerProviderConfig.from_env()
_PROVIDER_CONFIG.validate_for_runtime()
_PROVIDER_CLIENT = AgentProviderClient(_PROVIDER_CONFIG)


def _service_client(
    env_url: str, *, job_id: str | None, env_token: str | None = None
) -> ToolServiceClient:
    base_url = os.getenv(env_url, "")
    # Prefer a tool-specific token env (e.g. EIREL_WEB_SEARCH_TOKEN) so
    # each tool service can enforce its own auth; fall back to the shared
    # EIREL_TOOL_SERVICE_TOKEN for older runtime secrets.
    api_token = ""
    if env_token:
        api_token = os.getenv(env_token, "")
    if not api_token:
        api_token = os.getenv("EIREL_TOOL_SERVICE_TOKEN", "")
    return ToolServiceClient(
        ToolServiceConfig(
            base_url=base_url,
            api_token=api_token,
            job_id=job_id,
        )
    )


def _build_catalog(
    job_id: str,
    budget: BudgetTracker,
    trace: TraceRecorder,
) -> GeneralChatToolCatalog:
    web_client = _service_client(
        "EIREL_WEB_SEARCH_URL", job_id=job_id, env_token="EIREL_WEB_SEARCH_TOKEN"
    )
    x_client = _service_client(
        "EIREL_X_API_URL", job_id=job_id, env_token="EIREL_X_API_TOKEN"
    )
    semantic_scholar_client = _service_client(
        "EIREL_SEMANTIC_SCHOLAR_URL",
        job_id=job_id,
        env_token="EIREL_SEMANTIC_SCHOLAR_TOKEN",
    )
    sandbox_client = _service_client(
        "EIREL_SANDBOX_URL", job_id=job_id, env_token="EIREL_SANDBOX_TOKEN"
    )

    tools = [
        WebSearchTool(web_client),
        XApiTool(x_client),
        SemanticScholarTool(semantic_scholar_client),
        SandboxTool(sandbox_client),
    ]
    return GeneralChatToolCatalog(tools, budget=budget, trace=trace)


def _last_user_message(
    context: GeneralChatContext, request: AgentInvocationRequest
) -> str:
    for turn in reversed(context.conversation_history):
        if turn.role == "user" and turn.content:
            return turn.content
    return request.primary_goal or request.subtask or ""


def _extract_web_results(raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalise the owner-api web_search response to a list of result dicts.

    The owner-api tool service is expected to return ``{"results": [...]}``
    where each entry carries ``url``/``title``/``snippet``.  We accept a
    couple of common alternate shapes defensively so the miner doesn't crash
    on a minor schema drift.
    """
    if not isinstance(raw, dict):
        return []
    for key in ("documents", "results", "data", "items"):
        value = raw.get(key)
        if isinstance(value, list):
            return [entry for entry in value if isinstance(entry, dict)]
    return []


async def _run_turn(
    context: GeneralChatContext,
    provider: AgentProviderClient,
    request: AgentInvocationRequest,
) -> GeneralChatResponse:
    budget = BudgetTracker(budget=context.budget)
    run_budget_usd = float(os.getenv("EIREL_RUN_BUDGET_USD", "30.0"))
    run_cost = RunCostTracker(budget=RunBudget(max_usd=run_budget_usd))
    trace = TraceRecorder()
    # Use the miner-level job id (same as the provider-proxy LLM job id)
    # so tool_cost_usd attributes to the miner's deployment record in the
    # provider-proxy alongside llm_cost_usd. Using request.task_id would
    # route charges to a non-existent per-task job and 404 in the proxy,
    # leaving DeploymentScoreRecord.tool_cost_usd at 0 forever.
    catalog_job_id = os.getenv("EIREL_PROVIDER_PROXY_JOB_ID") or request.task_id
    catalog = _build_catalog(catalog_job_id, budget, trace)
    _ = run_cost  # available for per-run USD enforcement at the proxy layer

    messages: list[dict[str, Any]] = [
        {"role": turn.role, "content": turn.content}
        for turn in context.conversation_history
    ]
    user_prompt = _last_user_message(context, request)
    if not messages:
        messages = [{"role": "user", "content": user_prompt}]

    # When web_search is enabled, retrieve fresh web context up-front and
    # inject it as a system note before the LLM call.  This deterministically
    # exercises the tool-service path and supplies the model with citations
    # to quote.  The catalog records the ToolCall in the trace automatically.
    if context.web_search_enabled and user_prompt:
        try:
            web_raw = await catalog.execute(
                "web_search", query=user_prompt, max_results=5
            )
        except Exception as exc:  # tool failure should not break the turn
            trace.set_metadata("web_search_error", str(exc))
            web_raw = {}
        results = _extract_web_results(web_raw)
        if results:
            snippets = []
            for idx, item in enumerate(results, start=1):
                url = str(item.get("url") or "").strip()
                title = str(item.get("title") or "").strip()
                snippet = str(item.get("snippet") or item.get("content") or "").strip()
                if url:
                    trace.record_citation(
                        Citation(url=url, snippet=snippet or None, tool_name="web_search")
                    )
                snippets.append(
                    f"[{idx}] {title} ({url})\n{snippet}" if url or title else snippet
                )
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": (
                        "You have access to the following fresh web search results. "
                        "When relevant, incorporate them into your answer and cite the "
                        "source URLs inline as [1], [2], etc.\n\n"
                        + "\n\n".join(snippets)
                    ),
                },
            )

    # For OpenAI-compatible APIs ``max_tokens`` bounds the *total* output,
    # including any hidden reasoning tokens a thinking model produces
    # before the final answer. Reasoning-only models (e.g. Kimi-K2.5-TEE)
    # can burn the entire budget on the reasoning phase, leaving
    # ``content`` empty and causing the miner to return a blank response.
    # Allocate reasoning_tokens on top of output_tokens so the final
    # answer always has room, and enforce a 2048 floor so instant mode
    # on a reasoning model still has enough headroom.
    max_tokens = max(
        2048, context.budget.output_tokens + context.budget.reasoning_tokens
    )
    payload: dict[str, Any] = {
        "messages": messages,
        "max_tokens": max_tokens,
    }
    chat_response = await provider.chat_completions(payload)
    content = ""
    try:
        content = chat_response["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        content = ""
    trace.add_content(content)
    trace.set_metadata("mode", context.mode)
    trace.set_metadata("web_search_enabled", context.web_search_enabled)
    return trace.freeze()


async def _handle_chat(payload: dict[str, Any]) -> dict[str, Any]:
    """Handle /v1/chat/completions — simple LLM passthrough."""
    return await _PROVIDER_CLIENT.chat_completions(payload)


async def _handle_agent(payload: dict[str, Any]) -> dict[str, Any]:
    """Handle /v1/agent/infer — general_chat single-turn handler."""
    request = AgentInvocationRequest.model_validate(payload)

    history_payload = dict(payload)
    if "messages" not in history_payload and not request.inputs.get("conversation_history"):
        history_payload.setdefault(
            "messages",
            [
                ConversationTurn(role="user", content=request.primary_goal).model_dump()
            ]
            if request.primary_goal
            else [],
        )

    context = context_from_request(history_payload)
    result = await _run_turn(context, _PROVIDER_CLIENT, request)

    response = AgentInvocationResponse(
        task_id=request.task_id,
        family_id=request.family_id,
        status="completed",
        output=result.model_dump(mode="json"),
        citations=[c.url for c in result.citations],
        tool_calls=[tc.model_dump(mode="json") for tc in result.tool_calls],
    )
    return response.model_dump(mode="json")


app = MinerApp(
    title="Eirel General Chat Agent",
    handler=_handle_chat,
    agent_handler=_handle_agent,
).fastapi_app()

"""Planner-Executor: planner emits a tool plan, executor runs it.

Pattern from the Plan-and-Solve / Toolformer family. Useful when a
turn needs multiple tool calls in a defined order ("search for X, then
compute Y from the result, then look up Z"). One LLM call drafts the
full plan, then the executor walks it without further planning unless
``replan_on_failure`` is true and a step trips.

The plan is a JSON array. Each element is one
:class:`PlannerStep` ``{tool, args, depends_on?}``. The executor groups
steps by depth (steps with no remaining dependencies run in parallel
via :meth:`GeneralChatToolCatalog.execute_many`); the next layer waits
on the prior one's results.

A final synthesis call is made to the executor LLM with the user's
original prompt + the collected tool results, producing the assistant
message that lands on ``messages_key``.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from eirel.graph.node import NodeFn, PendingToolCall
from eirel.graph.patterns._trace import record_substep

if TYPE_CHECKING:
    from eirel.families.general_chat.tools import GeneralChatToolCatalog
    from eirel.provider import AgentProviderClient

__all__ = ["PlannerExecutorNode", "PlannerStep"]

_logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PlannerStep:
    id: str
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()


_DEFAULT_PLANNER_INSTRUCTION = (
    "You are a planning assistant. Given the user's request and the "
    "available tools, emit a JSON array of steps to execute. Each step "
    'is {"id": str, "tool": str, "args": object, "depends_on": [str, ...]}. '
    "Steps with no entries in `depends_on` run first; later steps may "
    "reference earlier step ids in `args` as the literal placeholder "
    '"$id" (the executor substitutes the prior result before invoking '
    "the tool). Return ONLY the JSON array — no prose, no fences."
)

_DEFAULT_SYNTHESIS_INSTRUCTION = (
    "Use the tool results below to answer the user's question. "
    "Cite tool names inline if relevant. Do not invent results not "
    "present in the tool output."
)


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE.sub("", text or "").strip()


def _parse_plan(text: str) -> list[PlannerStep]:
    cleaned = _strip_fences(text)
    if not cleaned:
        return []
    # Tolerate prose preamble: find the first '[' and last ']'.
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start < 0 or end < start:
        raise ValueError(f"planner did not return a JSON array: {text[:200]!r}")
    raw = json.loads(cleaned[start : end + 1])
    if not isinstance(raw, list):
        raise ValueError("planner output must be a JSON array")
    out: list[PlannerStep] = []
    seen_ids: set[str] = set()
    for idx, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"planner step {idx} is not an object")
        sid = str(entry.get("id") or f"step_{idx}")
        if sid in seen_ids:
            raise ValueError(f"duplicate step id: {sid!r}")
        seen_ids.add(sid)
        tool = entry.get("tool")
        if not isinstance(tool, str) or not tool:
            raise ValueError(f"planner step {sid} missing 'tool'")
        args = entry.get("args") or {}
        if not isinstance(args, dict):
            raise ValueError(f"planner step {sid}: 'args' must be an object")
        deps_raw = entry.get("depends_on") or []
        if not isinstance(deps_raw, list):
            raise ValueError(f"planner step {sid}: 'depends_on' must be a list")
        deps = tuple(str(d) for d in deps_raw)
        out.append(PlannerStep(id=sid, tool=tool, args=dict(args), depends_on=deps))
    return out


def _substitute_refs(args: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    """Replace ``"$id"`` placeholders in ``args`` with prior results."""

    def _walk(node: Any) -> Any:
        if isinstance(node, str) and node.startswith("$") and node[1:] in results:
            return results[node[1:]]
        if isinstance(node, list):
            return [_walk(x) for x in node]
        if isinstance(node, dict):
            return {k: _walk(v) for k, v in node.items()}
        return node

    return {k: _walk(v) for k, v in args.items()}


def _layer_plan(plan: list[PlannerStep]) -> list[list[PlannerStep]]:
    """Group steps into execution layers respecting ``depends_on``.

    Each returned list is a layer that can run in parallel. Raises
    on cycles or unknown ids.
    """
    by_id = {s.id: s for s in plan}
    for step in plan:
        for dep in step.depends_on:
            if dep not in by_id:
                raise ValueError(
                    f"step {step.id!r} depends on unknown id {dep!r}"
                )
    pending = list(plan)
    done: set[str] = set()
    layers: list[list[PlannerStep]] = []
    while pending:
        ready = [s for s in pending if all(d in done for d in s.depends_on)]
        if not ready:
            raise ValueError("planner output contains a dependency cycle")
        layers.append(ready)
        done.update(s.id for s in ready)
        pending = [s for s in pending if s.id not in done]
    return layers


class PlannerExecutorNode:
    """Factory: builds a planner-then-executor node.

    Parameters
    ----------
    planner_provider
        LLM used to draft the plan.
    executor_provider
        LLM used for the final synthesis pass.
    tool_catalog
        :class:`GeneralChatToolCatalog` used to dispatch tool calls.
    messages_key
        State field carrying the chat history.
    max_tokens
        Token allowance for planner and synthesis calls.
    max_replans
        Times the planner can be asked to revise on a step failure.
        ``0`` disables replanning.
    planner_instruction
        System message text injected when calling the planner.
    synthesis_instruction
        System message text injected when calling the synthesizer.
    name
        Sub-span name prefix in the trace.
    """

    __slots__ = (
        "_planner",
        "_executor",
        "_catalog",
        "_messages_key",
        "_max_tokens",
        "_max_replans",
        "_planner_instruction",
        "_synthesis_instruction",
        "_name",
    )

    def __init__(
        self,
        planner_provider: "AgentProviderClient",
        executor_provider: "AgentProviderClient",
        tool_catalog: "GeneralChatToolCatalog",
        *,
        messages_key: str = "messages",
        max_tokens: int = 2048,
        max_replans: int = 1,
        planner_instruction: str = _DEFAULT_PLANNER_INSTRUCTION,
        synthesis_instruction: str = _DEFAULT_SYNTHESIS_INSTRUCTION,
        name: str = "planner_executor",
    ) -> None:
        if max_replans < 0:
            raise ValueError("max_replans must be non-negative")
        self._planner = planner_provider
        self._executor = executor_provider
        self._catalog = tool_catalog
        self._messages_key = messages_key
        self._max_tokens = max_tokens
        self._max_replans = max_replans
        self._planner_instruction = planner_instruction
        self._synthesis_instruction = synthesis_instruction
        self._name = name

    async def _llm_call(
        self,
        provider: "AgentProviderClient",
        messages: list[dict[str, Any]],
        sub_label: str,
    ) -> str:
        t0 = time.monotonic()
        response = await provider.chat_completions(
            {"messages": messages, "max_tokens": self._max_tokens},
        )
        latency_ms = int((time.monotonic() - t0) * 1000)
        content = ""
        try:
            content = response["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            content = ""
        record_substep(
            tool_name=f"{self._name}:{sub_label}",
            args=None,
            result_digest=content[:500] if content else None,
            latency_ms=latency_ms,
        )
        return content

    async def _request_plan(
        self,
        base_messages: list[dict[str, Any]],
        prior_failure: str | None,
        attempt: int,
    ) -> list[PlannerStep]:
        tools_listing = ", ".join(self._catalog.available) or "(none)"
        prompt_pieces = [
            self._planner_instruction,
            f"\nAvailable tools: {tools_listing}.",
        ]
        if prior_failure:
            prompt_pieces.append(
                f"\nThe previous plan failed: {prior_failure}\n"
                "Emit a corrected plan."
            )
        planner_messages: list[dict[str, Any]] = [
            {"role": "system", "content": "".join(prompt_pieces)},
            *base_messages,
        ]
        raw = await self._llm_call(
            self._planner, planner_messages, f"plan_{attempt}"
        )
        return _parse_plan(raw)

    def __call__(self) -> NodeFn:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            base_messages = list(state.get(self._messages_key) or [])
            failure: str | None = None
            results: dict[str, Any] = {}
            execution_log: list[dict[str, Any]] = []

            for attempt in range(self._max_replans + 1):
                try:
                    plan = await self._request_plan(base_messages, failure, attempt)
                    layers = _layer_plan(plan)
                except ValueError as exc:
                    failure = str(exc)
                    if attempt == self._max_replans:
                        # Out of replans — surface the planning failure.
                        return {
                            self._messages_key: {
                                "role": "assistant",
                                "content": (
                                    "I wasn't able to draft an executable plan "
                                    f"for this request: {failure}"
                                ),
                            },
                        }
                    continue

                step_failed = False
                for layer in layers:
                    calls: list[PendingToolCall] = [
                        PendingToolCall(
                            name=step.tool,
                            arguments=_substitute_refs(step.args, results),
                            call_id=step.id,
                        )
                        for step in layer
                    ]
                    layer_results = await self._catalog.execute_many(calls)
                    for entry in layer_results:
                        execution_log.append(entry)
                        sid = entry.get("call_id") or entry.get("name") or ""
                        if entry.get("status") == "ok":
                            results[sid] = entry.get("result")
                        else:
                            step_failed = True
                            failure = (
                                f"step {sid!r} ({entry.get('name')}) failed: "
                                f"{entry.get('error')}"
                            )
                            break
                    if step_failed:
                        break

                if not step_failed:
                    failure = None
                    break

            synthesis_messages: list[dict[str, Any]] = [
                {"role": "system", "content": self._synthesis_instruction},
                *base_messages,
                {
                    "role": "system",
                    "content": (
                        "Tool results:\n"
                        + json.dumps(execution_log, default=str)[:8000]
                    ),
                },
            ]
            answer = await self._llm_call(
                self._executor, synthesis_messages, "synthesize"
            )
            return {
                self._messages_key: {"role": "assistant", "content": answer},
            }

        run.__name__ = self._name
        return run

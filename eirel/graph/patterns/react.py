"""ReAct: Thought → Action → Observation loop.

Pattern from Yao et al. 2022 (arXiv:2210.03629). The LLM emits a
free-form Thought, then a structured Action (tool name + JSON args).
The runtime dispatches the tool, captures the result as an
Observation, appends it to the rolling context, and asks the LLM to
continue. Loop ends when the LLM emits ``Action: final_answer`` (or a
step cap fires).

Output format expected from the LLM (case-insensitive labels, blank
lines between turns)::

    Thought: I need to look up the population of Paris.
    Action: web_search
    Action Input: {"query": "Paris population 2024"}

Or a terminal step::

    Thought: I now have what I need.
    Action: final_answer
    Action Input: "Paris has about 2.1 million residents (2024)."

The tool name ``final_answer`` is reserved — its ``Action Input`` (a
JSON string OR a JSON object with key ``"answer"``) becomes the
assistant message and the loop terminates. If the cap fires before a
``final_answer`` is emitted, the last Thought is surfaced as the
fallback response.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from eirel.graph.node import NodeFn
from eirel.graph.patterns._trace import record_substep

if TYPE_CHECKING:
    from eirel.families.general_chat.tools import GeneralChatToolCatalog
    from eirel.provider import AgentProviderClient

__all__ = ["ReActNode", "ReActStep"]

_logger = logging.getLogger(__name__)


_FINAL_ANSWER_TOOL = "final_answer"

_THOUGHT_RE = re.compile(r"^\s*thought\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_ACTION_RE = re.compile(r"^\s*action\s*:\s*(.*?)\s*$", re.IGNORECASE | re.MULTILINE)
_ACTION_INPUT_RE = re.compile(
    r"^\s*action\s*input\s*:\s*(.*?)(?=^\s*(?:thought|action)\s*:|\Z)",
    re.IGNORECASE | re.MULTILINE | re.DOTALL,
)


_DEFAULT_INSTRUCTION = (
    "You are a step-by-step ReAct agent. After each user or observation "
    "message, emit exactly one Thought, one Action, and one Action Input.\n"
    "Format:\n"
    "  Thought: <your reasoning>\n"
    "  Action: <tool_name OR final_answer>\n"
    "  Action Input: <JSON for the tool args, or a string answer>\n"
    "Use the tool name `final_answer` when you have the answer; its "
    "Action Input becomes the message to the user. Do NOT invent "
    "tool results — call the tool and wait for the Observation."
)


@dataclass(frozen=True, slots=True)
class ReActStep:
    """One Thought-Action-Observation cycle in a ReAct loop."""

    thought: str
    action: str
    action_input: Any
    observation: Any | None = None
    error: str | None = None
    raw_completion: str = field(default="")


def _parse_step(text: str) -> ReActStep:
    """Pull the first Thought / Action / Action Input out of an LLM reply.

    Tolerates extra prose around the labels. Raises ``ValueError`` if
    Action is missing — the caller treats that as a malformed step and
    re-prompts.
    """
    thought_match = _THOUGHT_RE.search(text or "")
    action_match = _ACTION_RE.search(text or "")
    if action_match is None:
        raise ValueError("ReAct step missing 'Action:' label")
    input_match = _ACTION_INPUT_RE.search(text or "")
    raw_input = (input_match.group(1).strip() if input_match else "").strip()
    parsed_input: Any = raw_input
    if raw_input:
        try:
            parsed_input = json.loads(raw_input)
        except json.JSONDecodeError:
            # Strings without quotes are valid for `final_answer`; keep
            # the raw text. For tool calls the catalog will validate.
            parsed_input = raw_input
    return ReActStep(
        thought=(thought_match.group(1).strip() if thought_match else ""),
        action=action_match.group(1).strip(),
        action_input=parsed_input,
        raw_completion=text or "",
    )


class ReActNode:
    """Factory: builds a ReAct loop over ``provider`` + ``tool_catalog``.

    Parameters
    ----------
    provider
        LLM used for every Thought-Action turn.
    tool_catalog
        Tools the LLM can dispatch via ``Action: <tool>``.
    messages_key
        State field carrying the chat history.
    max_steps
        Hard cap on Thought-Action-Observation iterations. Default 8.
    max_tokens
        Output-token allowance per LLM call.
    instruction
        System message text injected before the rolling history.
    name
        Sub-span name prefix in the trace.
    """

    __slots__ = (
        "_provider",
        "_catalog",
        "_messages_key",
        "_max_steps",
        "_max_tokens",
        "_instruction",
        "_name",
    )

    def __init__(
        self,
        provider: "AgentProviderClient",
        tool_catalog: "GeneralChatToolCatalog",
        *,
        messages_key: str = "messages",
        max_steps: int = 8,
        max_tokens: int = 1024,
        instruction: str = _DEFAULT_INSTRUCTION,
        name: str = "react",
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        self._provider = provider
        self._catalog = tool_catalog
        self._messages_key = messages_key
        self._max_steps = max_steps
        self._max_tokens = max_tokens
        self._instruction = instruction
        self._name = name

    async def _llm_call(
        self, messages: list[dict[str, Any]], sub_label: str
    ) -> str:
        t0 = time.monotonic()
        response = await self._provider.chat_completions(
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

    def _final_answer_text(self, action_input: Any) -> str:
        if isinstance(action_input, str):
            return action_input
        if isinstance(action_input, dict):
            ans = action_input.get("answer")
            if isinstance(ans, str):
                return ans
            return json.dumps(action_input)
        return json.dumps(action_input, default=str)

    def __call__(self) -> NodeFn:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            base_messages = list(state.get(self._messages_key) or [])
            tools_listing = ", ".join(self._catalog.available) or "(none)"
            system_msg = {
                "role": "system",
                "content": (
                    f"{self._instruction}\n\nAvailable tools: {tools_listing}."
                ),
            }
            scratch: list[dict[str, Any]] = []
            last_thought: str = ""
            for step_idx in range(self._max_steps):
                messages = [system_msg, *base_messages, *scratch]
                completion = await self._llm_call(messages, f"step_{step_idx}")
                try:
                    step = _parse_step(completion)
                except ValueError as exc:
                    record_substep(
                        tool_name=f"{self._name}:parse_error_{step_idx}",
                        args=None,
                        result_digest=str(exc),
                        latency_ms=0,
                    )
                    # Nudge the model to fix the format and try again.
                    scratch.append(
                        {
                            "role": "assistant",
                            "content": completion,
                        }
                    )
                    scratch.append(
                        {
                            "role": "user",
                            "content": (
                                "Your last reply was missing 'Action:'. "
                                "Reply with Thought / Action / Action Input "
                                "labels exactly."
                            ),
                        }
                    )
                    continue
                last_thought = step.thought or last_thought
                if step.action.casefold() == _FINAL_ANSWER_TOOL.casefold():
                    answer = self._final_answer_text(step.action_input)
                    return {
                        self._messages_key: {"role": "assistant", "content": answer},
                    }
                # Dispatch a tool call.
                args = step.action_input if isinstance(step.action_input, dict) else {}
                t0 = time.monotonic()
                try:
                    result = await self._catalog.execute(step.action, **args)
                    error: str | None = None
                except Exception as exc:  # noqa: BLE001 — surface to LLM
                    result = None
                    error = str(exc)
                latency_ms = int((time.monotonic() - t0) * 1000)
                record_substep(
                    tool_name=f"{self._name}:observation_{step_idx}",
                    args={"action": step.action, "input": args},
                    result_digest=(
                        (json.dumps(result, default=str)[:500])
                        if result is not None
                        else (error or "")
                    ),
                    latency_ms=latency_ms,
                )
                # Append the assistant turn (verbatim model output) and the
                # observation as a user-role message so the next iteration
                # sees the rolling tape.
                scratch.append({"role": "assistant", "content": completion})
                obs_payload: dict[str, Any] = (
                    {"observation": result} if error is None else {"error": error}
                )
                scratch.append(
                    {
                        "role": "user",
                        "content": f"Observation: {json.dumps(obs_payload, default=str)}",
                    }
                )
            # Step cap hit — surface the last thought as a fallback so the
            # trace is auditable rather than silently empty.
            fallback = (
                last_thought
                or "Step budget exhausted before reaching a final answer."
            )
            record_substep(
                tool_name=f"{self._name}:step_cap",
                args={"max_steps": self._max_steps},
                result_digest=fallback[:500],
                latency_ms=0,
            )
            return {
                self._messages_key: {"role": "assistant", "content": fallback},
            }

        run.__name__ = self._name
        return run

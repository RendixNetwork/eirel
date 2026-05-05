"""Reflection: generate, critique, revise — bounded loop.

Pattern from Madaan et al. 2023 (arXiv:2303.17651, "Self-Refine"). One
pass through the LLM is often a draft; a second pass that *critiques*
the draft and a third that *revises* against the critique routinely
beats greedy single-shot on long-form tasks.

This node runs the loop:

    1. Generate a candidate answer with ``generator_provider``.
    2. Critique it with ``critic_provider`` (defaults to the generator).
       The critic is asked to either reply with the literal sentinel
       ``"NO_CHANGE"`` (case-insensitive) or list specific
       improvements.
    3. If the critic returned ``NO_CHANGE`` (or we hit ``max_iterations``),
       stop and emit the latest candidate.
    4. Otherwise, revise: prompt the generator with the candidate and
       the critique, asking for an improved version. Loop back to step 2.

Total LLM calls: at most ``1 + 2 * max_iterations`` (one initial generate,
plus per-iteration critique + revise).

Cost note: ``max_iterations`` is a hard cap, not a soft target. Keep it
small (default ``2``) — a poorly-scoped critic can otherwise loop on
nitpicks forever.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from eirel.graph.node import NodeFn
from eirel.graph.patterns._trace import record_substep

if TYPE_CHECKING:
    from eirel.provider import AgentProviderClient

__all__ = ["ReflectionNode"]

_logger = logging.getLogger(__name__)


_NO_CHANGE_SENTINEL = "NO_CHANGE"

_DEFAULT_CRITIC_INSTRUCTION = (
    "You are reviewing the assistant's response to the user's request. "
    "If the response is correct, complete, and well-supported, reply "
    f"with the single token {_NO_CHANGE_SENTINEL!r} and nothing else. "
    "Otherwise, list 1–3 concrete improvements to make. Be specific; "
    "don't paraphrase the original."
)

_DEFAULT_REVISE_INSTRUCTION = (
    "Revise your previous response based on the critique below. "
    "Return ONLY the improved response — no preamble, no explanation "
    "of changes."
)


def _is_no_change(critique_text: str) -> bool:
    return _NO_CHANGE_SENTINEL.casefold() in (critique_text or "").strip().casefold()


class ReflectionNode:
    """Factory: builds a generate → critique → revise loop.

    Parameters
    ----------
    generator_provider
        Provider used for the initial response and revisions.
    critic_provider
        Provider used to critique the candidate. Defaults to the
        generator. Pass a stronger / cheaper model here when you want
        the critic and the generator to be different LLMs.
    max_iterations
        Maximum critique-revise rounds (default 2). Total LLM calls
        ≤ ``1 + 2 * max_iterations``.
    messages_key
        State field carrying the chat history.
    max_tokens
        Output-token allowance per LLM call (used for all three roles).
    critic_instruction
        System message text injected when calling the critic.
    revise_instruction
        System message text injected when asking for a revision.
    name
        Sub-span name prefix in the trace.
    """

    __slots__ = (
        "_generator",
        "_critic",
        "_max_iterations",
        "_messages_key",
        "_max_tokens",
        "_critic_instruction",
        "_revise_instruction",
        "_name",
    )

    def __init__(
        self,
        generator_provider: "AgentProviderClient",
        *,
        critic_provider: "AgentProviderClient | None" = None,
        max_iterations: int = 2,
        messages_key: str = "messages",
        max_tokens: int = 2048,
        critic_instruction: str = _DEFAULT_CRITIC_INSTRUCTION,
        revise_instruction: str = _DEFAULT_REVISE_INSTRUCTION,
        name: str = "reflection",
    ) -> None:
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        self._generator = generator_provider
        self._critic = critic_provider or generator_provider
        self._max_iterations = max_iterations
        self._messages_key = messages_key
        self._max_tokens = max_tokens
        self._critic_instruction = critic_instruction
        self._revise_instruction = revise_instruction
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

    def __call__(self) -> NodeFn:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            base_messages = list(state.get(self._messages_key) or [])
            candidate = await self._llm_call(
                self._generator, base_messages, "generate"
            )
            for iteration in range(self._max_iterations):
                critic_messages: list[dict[str, Any]] = [
                    {"role": "system", "content": self._critic_instruction},
                    *base_messages,
                    {"role": "assistant", "content": candidate},
                ]
                critique = await self._llm_call(
                    self._critic, critic_messages, f"critique_{iteration}"
                )
                if _is_no_change(critique):
                    record_substep(
                        tool_name=f"{self._name}:converged",
                        args={"iteration": iteration},
                        result_digest="no_change",
                        latency_ms=0,
                    )
                    break
                revise_messages: list[dict[str, Any]] = [
                    *base_messages,
                    {"role": "assistant", "content": candidate},
                    {
                        "role": "user",
                        "content": (
                            f"{self._revise_instruction}\n\n"
                            f"Critique:\n{critique}"
                        ),
                    },
                ]
                candidate = await self._llm_call(
                    self._generator, revise_messages, f"revise_{iteration}"
                )
            return {
                self._messages_key: {"role": "assistant", "content": candidate},
            }

        run.__name__ = self._name
        return run

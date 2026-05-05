"""Structured output: bounded retry + JSON repair against a pydantic schema.

Many model providers don't honor ``response_format={"type": "json_object"}``
strictly — they emit fenced markdown, prose-prefixed JSON, or schema
drift. ``StructuredOutputNode`` wraps an LLM call with a tight retry
loop that:

  1. Asks the LLM with the schema appended as instructions.
  2. Strips common wrappers (markdown fences, prose preambles).
  3. Validates against the pydantic model.
  4. On failure, re-asks with the validation error inlined so the model
     can self-correct.

The interface mirrors :class:`~eirel.graph.node.LLMNode` so a graph can
swap one for the other. This is intentionally a small re-implementation
rather than a hard dep on Outlines/Guidance — Outlines pulls in
``transformers`` which is too heavy to require for every miner just to
get a JSON parser.
"""
from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, TypeVar

from pydantic import BaseModel, ValidationError

from eirel.graph.node import NodeFn

if TYPE_CHECKING:
    from eirel.provider import AgentProviderClient

__all__ = ["StructuredOutputNode", "StructuredOutputError"]

_logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when all retries fail to produce a schema-valid response."""

    def __init__(self, *, attempts: int, last_error: str, last_payload: str) -> None:
        super().__init__(
            f"structured output failed after {attempts} attempts: {last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error
        self.last_payload = last_payload


_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def _strip_fences(text: str) -> str:
    return _FENCE_PATTERN.sub("", text).strip()


def _extract_first_json_object(text: str) -> str:
    """Slice the first balanced ``{...}`` substring out of ``text``.

    Tolerates prose preambles ("Here is the JSON: { ... }"). Returns
    the input unchanged when no balanced block is found, so the caller
    sees the upstream's exact failure mode in the error path.
    """
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return text


def _try_parse(text: str, schema: type[T]) -> T:
    cleaned = _strip_fences(text)
    candidate = _extract_first_json_object(cleaned)
    raw = json.loads(candidate)
    return schema.model_validate(raw)


class StructuredOutputNode:
    """Factory: builds a node that parses an LLM response into ``schema``.

    Parameters
    ----------
    provider
        The :class:`AgentProviderClient` to call.
    schema
        A pydantic ``BaseModel`` subclass; the parsed instance lands at
        ``output_key``.
    output_key
        State field that receives the parsed instance. Reducer should
        be :func:`~eirel.graph.state.replace`.
    messages_key
        State field carrying the chat history. The factory appends a
        system instruction with the schema before calling.
    max_tokens
        Output-token allowance for the underlying LLM call.
    max_attempts
        Total tries (initial + retries). Default 3.
    """

    __slots__ = (
        "_provider",
        "_schema",
        "_output_key",
        "_messages_key",
        "_max_tokens",
        "_max_attempts",
        "_extra_payload",
        "_post_parse",
    )

    def __init__(
        self,
        provider: "AgentProviderClient",
        schema: type[BaseModel],
        *,
        output_key: str = "structured_output",
        messages_key: str = "messages",
        max_tokens: int = 2048,
        max_attempts: int = 3,
        extra_payload: dict[str, Any] | None = None,
        post_parse: Callable[[BaseModel], BaseModel] | None = None,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._provider = provider
        self._schema = schema
        self._output_key = output_key
        self._messages_key = messages_key
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts
        self._extra_payload = dict(extra_payload) if extra_payload else None
        self._post_parse = post_parse

    def _system_instruction(self, retry_error: str | None) -> dict[str, str]:
        body = (
            "Reply ONLY with a single JSON object matching this schema. "
            "Do not include prose, markdown fences, or commentary.\n\n"
            f"Schema: {json.dumps(self._schema.model_json_schema())}"
        )
        if retry_error:
            body += (
                "\n\nYour previous reply failed validation: "
                f"{retry_error}\nReturn a corrected JSON object."
            )
        return {"role": "system", "content": body}

    def __call__(self) -> NodeFn:
        async def run(state: dict[str, Any]) -> dict[str, Any]:
            messages: list[dict[str, Any]] = list(state.get(self._messages_key) or [])
            last_error: str = ""
            last_payload: str = ""
            for attempt in range(self._max_attempts):
                payload: dict[str, Any] = {
                    "messages": [
                        self._system_instruction(last_error if attempt else None),
                        *messages,
                    ],
                    "max_tokens": self._max_tokens,
                }
                if self._extra_payload:
                    for key, value in self._extra_payload.items():
                        payload.setdefault(key, value)
                response = await self._provider.chat_completions(payload)
                content = ""
                try:
                    content = response["choices"][0]["message"].get("content") or ""
                except (KeyError, IndexError, TypeError):
                    content = ""
                last_payload = content
                try:
                    parsed = _try_parse(content, self._schema)
                except (json.JSONDecodeError, ValidationError) as exc:
                    last_error = str(exc)
                    _logger.debug(
                        "StructuredOutputNode attempt %d/%d failed: %s",
                        attempt + 1,
                        self._max_attempts,
                        last_error,
                    )
                    continue
                if self._post_parse is not None:
                    parsed = self._post_parse(parsed)
                return {self._output_key: parsed.model_dump(mode="json")}
            raise StructuredOutputError(
                attempts=self._max_attempts,
                last_error=last_error,
                last_payload=last_payload,
            )

        run.__name__ = "structured_output_node"
        return run

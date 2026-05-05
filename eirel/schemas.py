from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from eirel import models as _eirel_models
from eirel.groups import FamilyId, ensure_active_family_id


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


MAX_METADATA_DEPTH = _env_int("EIREL_MAX_METADATA_DEPTH", 8)


def _check_depth(value: Any, limit: int, label: str, _level: int = 0) -> None:
    if _level > limit:
        raise ValueError(f"{label} nesting depth exceeds EIREL_MAX_METADATA_DEPTH ({limit})")
    if isinstance(value, dict):
        for child in value.values():
            _check_depth(child, limit, label, _level + 1)
    elif isinstance(value, list):
        for child in value:
            _check_depth(child, limit, label, _level + 1)


class AgentCapabilityMetadata(BaseModel):
    family_id: FamilyId
    description: str = Field(max_length=500)
    supports_streaming: bool = False
    latency_ms_p50: int = Field(default=1500, ge=1)
    estimated_cost_tao: float = Field(default=0.0, ge=0.0)
    max_context_tokens: int | None = Field(default=None, ge=1)

    @field_validator("family_id", mode="before")
    @classmethod
    def validate_family_id(cls, value: str) -> str:
        return ensure_active_family_id(value)


class ContextMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def enforce_content_size(cls, value: str) -> str:
        limit = _eirel_models.MAX_MESSAGE_CONTENT_BYTES
        if len(value.encode("utf-8")) > limit:
            raise ValueError(
                f"history content exceeds EIREL_MAX_MESSAGE_CONTENT_BYTES "
                f"({limit} bytes)"
            )
        return value


class Turn(BaseModel):
    """One scripted turn from a multi-turn task fixture.

    Used by ``AgentInvocationRequest.turns`` (eval-mode multi-turn
    dispatch). Each entry pairs a user message with the validator's
    pre-scripted assistant reply. The FINAL turn in a fixture has
    ``assistant=None`` — that's the open turn the agent must answer.
    """

    model_config = {"extra": "forbid"}
    user: str = Field(min_length=1, max_length=10_000)
    assistant: str | None = Field(default=None, max_length=10_000)


class InvocationConstraints(BaseModel):
    max_latency_ms: int | None = Field(default=None, ge=1)
    sync_mode: bool = True
    quality_tier: Literal["standard", "premium"] = "standard"
    modalities_allowed: list[str] = Field(default_factory=lambda: ["text"])


class AgentInvocationRequest(BaseModel):
    """Slim specialist contract.

    Family agents are stateless specialists invoked by the orchestrator
    (production) or the validator (eval). Both build the same body; the
    family agent never sees session/orchestration state directly. The
    contract is deliberately minimal:

      * ``prompt``       — the latest user message to answer.
      * ``history``      — prior conversation turns (user/assistant);
                           may be empty for the first turn or for tasks
                           with no preceding context.
      * ``turns``        — OPTIONAL structured conversation transcript
                           for tasks where the validator dispatches a
                           pre-scripted multi-turn fixture (eval mode).
                           Carries ``[{user, assistant?}]`` entries; the
                           final entry MAY have ``assistant=None`` to
                           signal the open turn the agent must answer.
                           When ``turns`` is set, ``history`` is also
                           populated (flattened) so agents that only
                           read ``history`` keep working — but agents
                           with session-memory infrastructure should
                           prefer ``turns`` for the structural signal.
      * ``mode``         — ``"instant"`` for fast turnaround, ``"thinking"``
                           for full reasoning.
      * ``web_search``   — explicit per-turn toggle; the family must
                           respect it.
      * ``turn_id``      — opaque correlation id for trace tagging.

    Agents should infer task shape from prompt/history/turns/inputs
    content. There is no ``category`` field — eval and product traffic
    are intentionally indistinguishable on the wire so behavior tuned
    for one transfers cleanly to the other.

    Tasks with attached content (long documents, CSVs, file uploads)
    surface the content under ``inputs.document_text`` (single doc) or
    ``inputs.attached_documents`` (list[str]). Agents that handle
    document tasks should inspect these keys.
    """

    # ── Slim contract (preferred) ───────────────────────────────────
    prompt: str | None = None
    history: list[ContextMessage] = Field(default_factory=list, max_length=100)
    turns: list[Turn] | None = None
    mode: Literal["instant", "thinking"] | None = None
    web_search: bool | None = None
    turn_id: str | None = None

    # ── Workflow-runtime fields (orchestrator-managed) ──────────────
    task_id: str | None = None
    session_id: str | None = None
    family_id: FamilyId | None = None
    episode_id: str | None = None
    workflow_spec_id: str | None = None
    workflow_version: str | None = None
    planner_node_id: str | None = None
    role_id: str | None = None
    upstream_node_outputs: dict[str, Any] = Field(default_factory=dict)
    context_bundle: dict[str, Any] = Field(default_factory=dict)
    checkpoint_state: dict[str, Any] = Field(default_factory=dict)
    resume_token: str | None = None
    artifact_requirements: dict[str, Any] = Field(default_factory=dict)
    trace_policy: dict[str, Any] = Field(default_factory=dict)
    inputs: dict[str, Any] = Field(default_factory=dict)
    constraints: InvocationConstraints = Field(default_factory=InvocationConstraints)
    max_execution_seconds: int | None = Field(default=None, ge=1)
    callback_url: str | None = None
    checkpoint_interval_seconds: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def validate_family_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_active_family_id(value)

    @model_validator(mode="after")
    def enforce_nesting_depth(self):
        _check_depth(self.metadata, MAX_METADATA_DEPTH, "metadata")
        _check_depth(self.inputs, MAX_METADATA_DEPTH, "inputs")
        _check_depth(self.context_bundle, MAX_METADATA_DEPTH, "context_bundle")
        return self

    @model_validator(mode="after")
    def require_prompt(self) -> AgentInvocationRequest:
        if not self.prompt:
            raise ValueError("AgentInvocationRequest requires `prompt`")
        return self


class ArtifactReference(BaseModel):
    artifact_id: str
    kind: Literal["image", "audio", "video", "document", "other"]
    uri: str
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class StreamChunk(BaseModel):
    """One unit of a streaming /v1/agent/infer/stream response.

    The stream is NDJSON: one JSON object per line, each parsing into a
    StreamChunk. Order is significant. The stream MUST end with a `done`
    chunk; any text accumulated by the receiver up to that point is the
    final answer. No `done` chunk = treat the stream as failed.

    Event types:
      - `delta`     : `text` carries the next slice of the answer body.
                      Receivers concatenate `text` across all `delta`
                      chunks in order to reconstruct the full response.
      - `tool_call` : `tool_call` carries one tool invocation record.
                      Surfaced for trace/citation extraction.
      - `citation`  : `citation` carries one URL/title pair the agent
                      wants to attribute. Equivalent to the citations
                      list on the non-streaming response.
      - `done`      : terminal chunk. `output`, `citations`, and
                      `metadata` mirror the non-streaming
                      AgentInvocationResponse so callers that only need
                      the final state can ignore the deltas. Executed
                      tool calls live in ``metadata.executed_tool_calls``
                      from 0.3.0 onward (top-level ``tool_calls``
                      removed).
    """

    event: Literal["delta", "tool_call", "citation", "done"]
    text: str | None = None
    tool_call: dict[str, Any] | None = None
    citation: dict[str, Any] | None = None
    output: dict[str, Any] | None = None
    citations: list[str] | None = None
    status: Literal["completed", "failed", "deferred"] | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentInvocationResponse(BaseModel):
    """Family agent response shape.

    Slim by design — the orchestrator and validator only need the
    answer text (``output``), the citations the agent gathered, and a
    status. Anything trace-related (executed tool calls, latency
    breakdown, web search ledger) lives under ``metadata`` so callers
    that only need the answer don't have to filter noise.

    Note: ``tool_calls`` was a top-level field through 0.2.x. It mixed
    "what the LLM emitted" with "what the Python code ran" and is
    confused naming with the OpenAI convention. From 0.3.0 it is
    surfaced as ``metadata["executed_tool_calls"]`` instead.
    """

    task_id: str | None = None
    family_id: FamilyId | None = None
    status: Literal["completed", "failed", "deferred"] = "completed"
    output: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    latency_ms: int = Field(default=0, ge=0)
    cost_tao: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    # Workflow-runtime fields used by the orchestrator's deferred /
    # checkpointing path. Family agents themselves don't populate these.
    checkpoint_events: list[dict[str, Any]] = Field(default_factory=list)
    runtime_state_patch: dict[str, Any] = Field(default_factory=dict)
    resume_token: str | None = None
    reliability_score: float | None = Field(default=None, ge=0.0, le=1.0)
    recovery_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def validate_response_family_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return ensure_active_family_id(value)


class AgentHealthStatus(BaseModel):
    status: Literal["ok", "degraded", "failed"] = "ok"
    family_id: FamilyId
    version: str
    details: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def validate_health_family_id(cls, value: str) -> str:
        return ensure_active_family_id(value)


class AgentRegistrationMetadata(BaseModel):
    hotkey: str
    endpoint: str
    family_id: FamilyId
    version: str
    capabilities: AgentCapabilityMetadata
    wallet_name: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def validate_registration_family_id(cls, value: str) -> str:
        return ensure_active_family_id(value)

from __future__ import annotations

import os
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from eirel import models as _eirel_models
from eirel.groups import FamilyId, ensure_active_family_id
from eirel.models import ToolDefinition


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
                f"context_history content exceeds EIREL_MAX_MESSAGE_CONTENT_BYTES "
                f"({limit} bytes)"
            )
        return value


class InvocationConstraints(BaseModel):
    max_latency_ms: int | None = Field(default=None, ge=1)
    sync_mode: bool = True
    quality_tier: Literal["standard", "premium"] = "standard"
    modalities_allowed: list[str] = Field(default_factory=lambda: ["text"])


class AgentInvocationRequest(BaseModel):
    task_id: str
    session_id: str | None = None
    primary_goal: str
    subtask: str
    family_id: FamilyId
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
    context_history: list[ContextMessage] = Field(default_factory=list, max_length=100)
    tools: list[ToolDefinition] | None = None
    tool_choice: str | dict[str, Any] | None = None
    constraints: InvocationConstraints = Field(default_factory=InvocationConstraints)
    execution_mode: str | None = None
    max_execution_seconds: int | None = Field(default=None, ge=1)
    callback_url: str | None = None
    checkpoint_interval_seconds: int | None = Field(default=None, ge=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def validate_family_id(cls, value: str) -> str:
        return ensure_active_family_id(value)

    @model_validator(mode="after")
    def enforce_nesting_depth(self):
        _check_depth(self.metadata, MAX_METADATA_DEPTH, "metadata")
        _check_depth(self.inputs, MAX_METADATA_DEPTH, "inputs")
        _check_depth(self.context_bundle, MAX_METADATA_DEPTH, "context_bundle")
        return self


class ArtifactReference(BaseModel):
    artifact_id: str
    kind: Literal["image", "audio", "video", "document", "other"]
    uri: str
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentInvocationResponse(BaseModel):
    task_id: str
    family_id: FamilyId
    status: Literal["completed", "failed", "deferred"] = "completed"
    output: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[ArtifactReference] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)
    latency_ms: int = Field(default=0, ge=0)
    cost_tao: float = Field(default=0.0, ge=0.0)
    error: str | None = None
    checkpoint_events: list[dict[str, Any]] = Field(default_factory=list)
    runtime_state_patch: dict[str, Any] = Field(default_factory=dict)
    resume_token: str | None = None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    reliability_score: float | None = Field(default=None, ge=0.0, le=1.0)
    recovery_score: float | None = Field(default=None, ge=0.0, le=1.0)
    progress: dict[str, Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("family_id", mode="before")
    @classmethod
    def validate_response_family_id(cls, value: str) -> str:
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

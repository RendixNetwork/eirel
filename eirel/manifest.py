from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from eirel.groups import FamilyId, ensure_active_family_id


TEXT_FAMILY_IDS = {"general_chat"}
MEDIA_FAMILY_IDS: set[str] = set()


DEFAULT_MODEL = "moonshotai/Kimi-K2.5-TEE"

# Per-family defaults for fields miners shouldn't worry about.
_FAMILY_DEFAULTS: dict[str, dict] = {
    "general_chat": {
        "cpu": 2,
        "memory_gb": 2,
        "timeout_seconds": 75,
        "protocol": "agent_invocation_v1",
        "invoke_path": "/v1/agent/infer",
    },
}

def _family_default(family_id: str, key: str, fallback=None):
    return _FAMILY_DEFAULTS.get(family_id, {}).get(key, fallback)


class AgentInfo(BaseModel):
    name: str
    version: str = "1.0.0"


class BuildInfo(BaseModel):
    context: str = "."
    dockerfile: str = "Dockerfile"


class RuntimeInfo(BaseModel):
    port: int = 8080
    health_path: str = "/healthz"
    invoke_path: str = "/v1/chat/completions"

    @field_validator("health_path", "invoke_path")
    @classmethod
    def ensure_path(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError("runtime paths must start with '/'")
        return value


class ResourceInfo(BaseModel):
    cpu: float = Field(default=1, gt=0)
    memory_gb: float = Field(default=1, gt=0)
    gpu: bool = False
    gpu_memory_gb: float | None = Field(default=None, gt=0)


class InferenceInfo(BaseModel):
    providers: list[str] = Field(default_factory=lambda: ["chutes"])
    fallback_order: list[str] = Field(default_factory=lambda: ["chutes"])
    protocol: Literal["openai_chat_completions_subset_v1", "agent_invocation_v1"] = (
        "openai_chat_completions_subset_v1"
    )
    requires_subnet_provider_proxy: bool = True
    model: str | None = DEFAULT_MODEL
    provider_mode: Literal["direct", "proxy", "auto"] = "proxy"

    @field_validator("providers", "fallback_order")
    @classmethod
    def ensure_unique_non_empty_entries(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            provider = str(item).strip()
            if not provider:
                raise ValueError("provider identifiers must be non-empty")
            if provider in seen:
                raise ValueError("provider identifiers must be unique")
            seen.add(provider)
            normalized.append(provider)
        return normalized


class SdkRuntimeInfo(BaseModel):
    package_mode: Literal["single_file", "package"] = "package"
    entry_module: str | None = None
    app_object: str = "app"
    dependency_group: str | None = None


class SubmissionManifest(BaseModel):
    schema_version: Literal[1] = 1
    agent: AgentInfo
    family_id: FamilyId
    build: BuildInfo = Field(default_factory=BuildInfo)
    runtime: RuntimeInfo = Field(default_factory=RuntimeInfo)
    capabilities: list[str] = Field(default_factory=list)
    resources: ResourceInfo = Field(default_factory=ResourceInfo)
    timeout_seconds: int = Field(default=30, gt=0)
    inference: InferenceInfo = Field(default_factory=InferenceInfo)
    sdk_runtime: SdkRuntimeInfo = Field(default_factory=SdkRuntimeInfo)

    @field_validator("family_id", mode="before")
    @classmethod
    def validate_family_id(cls, value: str) -> str:
        return ensure_active_family_id(value)

    @field_validator("capabilities")
    @classmethod
    def normalize_capabilities(cls, value: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for item in value:
            capability = str(item).strip()
            if not capability:
                continue
            if capability in seen:
                continue
            seen.add(capability)
            normalized.append(capability)
        return normalized

    @field_validator("inference")
    @classmethod
    def validate_inference(cls, value: InferenceInfo) -> InferenceInfo:
        if not value.requires_subnet_provider_proxy:
            raise ValueError("managed subnet submissions must require subnet provider proxy")
        if value.provider_mode == "direct":
            raise ValueError("managed subnet submissions may not use direct provider mode")
        return value

    @model_validator(mode="after")
    def validate_runtime_contract(self) -> SubmissionManifest:
        family_id = str(self.family_id)
        invoke_path = self.runtime.invoke_path
        protocol = self.inference.protocol
        if protocol == "agent_invocation_v1" and invoke_path != "/v1/agent/infer":
            raise ValueError("agent_invocation_v1 submissions must expose /v1/agent/infer")
        if family_id in MEDIA_FAMILY_IDS:
            if protocol != "agent_invocation_v1":
                raise ValueError(
                    f"{family_id} submissions must use the typed agent_invocation_v1 protocol"
                )
            if invoke_path != "/v1/agent/infer":
                raise ValueError(f"{family_id} submissions must expose /v1/agent/infer")
        if family_id in TEXT_FAMILY_IDS and protocol == "openai_chat_completions_subset_v1":
            return self
        return self


def _apply_family_defaults(payload: dict) -> dict:
    """Fill missing fields from family-specific defaults before validation."""
    family_id = str(payload.get("family_id", "")).strip()
    defaults = _FAMILY_DEFAULTS.get(family_id)
    if defaults is None:
        return payload
    # runtime
    if "runtime" not in payload:
        payload["runtime"] = {}
    rt = payload["runtime"]
    if "invoke_path" not in rt:
        rt["invoke_path"] = defaults["invoke_path"]
    if "port" not in rt:
        rt["port"] = 8080
    if "health_path" not in rt:
        rt["health_path"] = "/healthz"
    # resources
    if "resources" not in payload:
        payload["resources"] = {"cpu": defaults["cpu"], "memory_gb": defaults["memory_gb"]}
    else:
        res = payload["resources"]
        if "cpu" not in res:
            res["cpu"] = defaults["cpu"]
        if "memory_gb" not in res:
            res["memory_gb"] = defaults["memory_gb"]
    # timeout
    if "timeout_seconds" not in payload:
        payload["timeout_seconds"] = defaults["timeout_seconds"]
    # inference
    if "inference" not in payload:
        payload["inference"] = {}
    inf = payload["inference"]
    if "protocol" not in inf:
        inf["protocol"] = defaults["protocol"]
    if "model" not in inf:
        inf["model"] = DEFAULT_MODEL
    if "providers" not in inf:
        inf["providers"] = ["chutes"]
    if "fallback_order" not in inf:
        inf["fallback_order"] = ["chutes"]
    # capabilities
    if "capabilities" not in payload or not payload["capabilities"]:
        payload["capabilities"] = [family_id]
    return payload


def parse_manifest_bytes(raw: bytes) -> SubmissionManifest:
    try:
        payload = yaml.safe_load(raw)
        payload = _apply_family_defaults(payload)
        return SubmissionManifest.model_validate(payload)
    except (yaml.YAMLError, ValidationError) as exc:
        raise ValueError(f"invalid submission manifest: {exc}") from exc


def extract_manifest_from_archive(archive_bytes: bytes) -> SubmissionManifest:
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        member = tar.getmember("submission.yaml")
        handle = tar.extractfile(member)
        if handle is None:
            raise ValueError("submission.yaml missing from archive")
        return parse_manifest_bytes(handle.read())


def validate_submission_directory(path: Path) -> SubmissionManifest:
    manifest_path = path / "submission.yaml"
    if not manifest_path.exists():
        raise ValueError("submission.yaml missing from directory")
    return parse_manifest_bytes(manifest_path.read_bytes())

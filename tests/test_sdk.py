from __future__ import annotations

import io
import tarfile

from eirel.compliance import run_public_compliance_suite
from eirel.helpers import content_response
from eirel.manifest import SubmissionManifest, extract_manifest_from_archive
from eirel.provider import MinerProviderConfig


def test_manifest_extract_roundtrip():
    manifest = """
schema_version: 1
agent:
  name: demo
  version: 1.0.0
family_id: general_chat
runtime:
  port: 8080
  health_path: /healthz
  invoke_path: /v1/agent/infer
capabilities: [general_chat]
resources:
  cpu: 2
  memory_gb: 2
  gpu: false
timeout_seconds: 75
inference:
  providers: [openai]
  fallback_order: [openai]
  protocol: agent_invocation_v1
  provider_mode: proxy
  requires_subnet_provider_proxy: true
""".strip().encode()
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        info = tarfile.TarInfo("submission.yaml")
        info.size = len(manifest)
        archive.addfile(info, io.BytesIO(manifest))
    parsed = extract_manifest_from_archive(buffer.getvalue())
    assert isinstance(parsed, SubmissionManifest)
    assert parsed.agent.name == "demo"


def test_provider_config_supports_proxy_env_defaults(monkeypatch):
    monkeypatch.setenv("MINER_PROVIDER", "openai")
    monkeypatch.setenv("MINER_MODEL", "gpt-4.1-mini")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_URL", "http://proxy")
    monkeypatch.setenv("EIREL_PROVIDER_PROXY_TOKEN", "token")
    config = MinerProviderConfig.from_env()
    assert config.mode == "auto"
    assert config.subnet_proxy_url == "http://proxy"


def test_provider_config_accepts_generic_api_env_fallbacks(monkeypatch):
    monkeypatch.delenv("MINER_PROVIDER", raising=False)
    monkeypatch.delenv("MINER_MODEL", raising=False)
    monkeypatch.delenv("MINER_API_KEY", raising=False)
    monkeypatch.delenv("MINER_API_BASE_URL", raising=False)
    monkeypatch.setenv("PROVIDER", "chutes")
    monkeypatch.setenv("MODEL_NAME", "deepseek-ai/DeepSeek-V3-0324")
    monkeypatch.setenv("API_KEY", "secret")
    monkeypatch.setenv("API_ENDPOINT", "https://api.chutes.ai/v1/chat/completions")

    config = MinerProviderConfig.from_env()

    assert config.provider == "chutes"
    assert config.model == "deepseek-ai/DeepSeek-V3-0324"
    assert config.api_key == "secret"
    assert config.base_url == "https://api.chutes.ai/v1/chat/completions"


def test_public_compliance_suite_accepts_basic_responder():
    def sender(payload):
        return content_response("Hello, I can help with questions.").model_dump(mode="json")

    results = run_public_compliance_suite(sender)
    assert results[0]["passed"] is True


def test_general_chat_manifest_accepts_agent_invocation_protocol():
    payload = {
        "schema_version": 1,
        "agent": {"name": "demo", "version": "1.0.0"},
        "family_id": "general_chat",
        "runtime": {
            "port": 8080,
            "health_path": "/healthz",
            "invoke_path": "/v1/agent/infer",
        },
        "capabilities": ["general_chat"],
        "resources": {"cpu": 2, "memory_gb": 2, "gpu": False},
        "timeout_seconds": 75,
        "inference": {
            "providers": ["openai"],
            "fallback_order": ["openai"],
            "protocol": "agent_invocation_v1",
            "provider_mode": "proxy",
            "requires_subnet_provider_proxy": True,
        },
    }
    parsed = SubmissionManifest.model_validate(payload)
    assert parsed.family_id == "general_chat"
    assert parsed.runtime.invoke_path == "/v1/agent/infer"

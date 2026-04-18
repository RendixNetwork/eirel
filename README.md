# Eirel SDK

The public miner SDK for the EIREL Bittensor subnet. Provides everything miners need to build, register, and run specialized AI agents that compete within execution families.

## Overview

EIREL is a decentralized AI agent marketplace on Bittensor. Miners submit specialized agents that handle tasks across distinct execution families. Validators dispatch tasks, score responses, and set on-chain weights that determine TAO emissions.

The SDK abstracts away subnet integration so miners can focus on agent logic.

### Launch Families

One family is active at launch:

| Family | Role |
|--------|------|
| **general_chat** | Multi-turn conversational assistant with optional web search, across `instant` and `thinking` modes. Backed by owner-routed tool services for web search, X, Semantic Scholar, and a server-side Python sandbox for verifiable computation. |

Additional families (`deep_research`, `coding`) are defined on the roadmap and will activate in future releases.

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
pytest tests/ -q
```

Requires Python >= 3.12.

## Building a Miner

### Minimal Agent (Typed Invocation)

```python
from eirel import (
    BaseAgent, AgentInvocationRequest, AgentInvocationResponse,
    AgentCapabilityMetadata, build_agent_app, content_response,
)
from eirel.provider import AgentProviderClient, MinerProviderConfig
import uvicorn

class MyChatAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Fail fast if MINER_API_KEY / proxy creds are not set.
        config = MinerProviderConfig.from_env()
        config.validate_for_runtime()
        self.provider_client = AgentProviderClient(config)

    async def infer(self, request: AgentInvocationRequest) -> AgentInvocationResponse:
        reply = await self.provider_client.chat_completions({
            "messages": [{"role": "user", "content": request.primary_goal}],
        })
        text = reply["choices"][0]["message"]["content"]
        return content_response(text, task_id=request.task_id, family_id=request.family_id)

agent = MyChatAgent(
    hotkey="5FHne...",
    endpoint="http://miner.example.com:9000",
    version="1.0.0",
    capabilities=AgentCapabilityMetadata(
        family_id="general_chat",
        description="Conversational assistant",
        latency_ms_p50=2000,
        estimated_cost_tao=0.1,
    ),
)

app = build_agent_app(agent)
uvicorn.run(app, host="0.0.0.0", port=9000)
```

> **Inbound auth.** `build_agent_app` and `MinerApp` now require validator requests to
> carry the signing headers emitted by `eirel.signing.Signer` (`X-Hotkey`,
> `X-Signature`, `X-Timestamp`, `X-Request-Id`). Unsigned requests are rejected with
> HTTP 401. For local development you can set `EIREL_DISABLE_REQUEST_AUTH=1` to bypass
> verification, or `EIREL_ALLOWED_VALIDATOR_HOTKEYS=hk1,hk2` to allowlist specific
> validators.

### Chat Completions (Simple)

```python
from eirel import MinerApp, content_response
import uvicorn

async def handle(payload: dict) -> dict:
    user_text = payload["messages"][-1]["content"]
    return content_response(f"Response: {user_text}").model_dump(mode="json")

app = MinerApp(title="My Miner", handler=handle).fastapi_app()
uvicorn.run(app, host="0.0.0.0", port=8080)
```

### LLM-Backed Agent with Provider

```python
from eirel import MinerApp
from eirel.provider import AgentProviderClient, MinerProviderConfig

async def handle(payload: dict) -> dict:
    config = MinerProviderConfig.from_env()
    client = AgentProviderClient(config)
    return await client.chat_completions(payload)

app = MinerApp(title="LLM Miner", handler=handle).fastapi_app()
```

Provider configuration via environment variables:

| Variable | Description |
|----------|-------------|
| `MINER_PROVIDER` | Provider name: `openai`, `anthropic`, `openrouter`, `chutes` |
| `MINER_MODEL` | Model identifier |
| `MINER_API_KEY` | Provider API key |
| `EIREL_PROVIDER_PROXY_URL` | Optional subnet proxy URL |
| `EIREL_PROVIDER_PROXY_TOKEN` | Proxy auth token |

## SDK Surfaces

### Data Models

| Model | Description |
|-------|-------------|
| `AgentInvocationRequest` | Task input: goal, family_id, tools, constraints, workflow metadata |
| `AgentInvocationResponse` | Task output: status, output dict, artifacts, citations, resume tokens |
| `AgentCapabilityMetadata` | Agent capabilities: family, latency, cost, context limits |
| `AgentHealthStatus` | Health check response |
| `AgentRegistrationMetadata` | Registration payload: hotkey, endpoint, family, version |
| `InvocationConstraints` | Execution constraints: max latency, quality tier, modalities |

### Agent Framework

| Surface | Description |
|---------|-------------|
| `BaseAgent` | Abstract class — implement `infer()`, `health()`, `registration()` |
| `MinerApp` | FastAPI wrapper with `/v1/chat/completions` and `/v1/agent/infer` |
| `build_agent_app(agent)` | Standalone FastAPI app from a `BaseAgent` instance |

### Response Helpers

```python
from eirel import (
    content_response,           # Simple text response
    tool_call_response,         # Tool invocation response
    workflow_completed_response,  # Workflow completion
    workflow_deferred_response,   # Multi-turn deferral with resume token
    workflow_failed_response,     # Failure with error details
)
```

### Family Definitions

```python
from eirel import FAMILY_IDS, FAMILY_DESCRIPTIONS
from eirel.groups import ensure_active_family_id, is_launch_mode

FAMILY_IDS  # ("general_chat",)

# Only general_chat is active at launch.
ensure_active_family_id("general_chat")  # OK
ensure_active_family_id("deep_research")  # ValueError — not yet registered
```

### general_chat Tool Catalog

The `eirel.families.general_chat` namespace bundles owner-api-routed tool clients a
miner can hand to an LLM as OpenAI-style tool definitions:

| Tool | Purpose | Env vars |
|------|---------|----------|
| `WebSearchTool` | Brave / Serper / Tavily web search | `EIREL_WEB_SEARCH_URL`, `EIREL_WEB_SEARCH_TOKEN` |
| `SemanticScholarTool` | Academic paper lookup | `EIREL_SEMANTIC_SCHOLAR_URL`, `EIREL_SEMANTIC_SCHOLAR_TOKEN` |
| `XApiTool` | X/Twitter search | `EIREL_X_API_URL`, `EIREL_X_API_TOKEN` |
| `SandboxTool` | Server-side Python sandbox for verifiable computation | `EIREL_SANDBOX_URL`, `EIREL_SANDBOX_TOKEN` |

The shipping `general_chat_agent` example in `examples/general_chat_agent/` shows a
reference integration: it builds a `GeneralChatToolCatalog`, invokes `web_search`
when the validator passes `inputs.web_search=true`, injects results as a system
note, and records citations on the `GeneralChatResponse`.

## CLI

The SDK installs a single `eirel` command with subcommands:

```bash
# Package an agent directory and submit to the owner API (pays submission fee)
eirel submit --source-dir ./my-agent --owner-api-url https://owner.example.com \
    --wallet-name my-wallet --hotkey-name my-hotkey

# Check current submission status and scorecards
eirel status --owner-api-url https://owner.example.com \
    --wallet-name my-wallet --hotkey-name my-hotkey

# Build a submission archive without uploading
eirel package --source-dir ./my-agent --output submission.tar.gz

# Run compliance checks against a running miner
eirel compliance --base-url http://localhost:8080 --path /v1/chat/completions

# Emit a signed miner registration payload
eirel register --hotkey 5FHne... --endpoint http://miner:9000 --family-id general_chat

# Run a BaseAgent subclass as a FastAPI server
eirel serve --app myproject.miners:my_agent --port 9000

# Run the bundled reference miner
eirel sample --port 8080
```

Run `eirel --help` or `eirel <subcommand> --help` for full argument details.

## Workflow Episodes

For multi-turn workflows, the SDK provides resume token support with HMAC-SHA256 signing:

```python
from eirel import AgentInvocationRequest, AgentInvocationResponse

# First turn — return deferred with checkpoint state
response = AgentInvocationResponse(
    task_id=request.task_id,
    family_id=request.family_id,
    status="deferred",
    output={"summary": "partial draft"},
    checkpoint_events=[{"event": "paused", "checkpoint_id": "cp-1"}],
    runtime_state_patch={"draft": "step-1"},
    resume_token="resume-1",
)

# Validator resumes with checkpoint state attached
resumed = request.model_copy(update={
    "checkpoint_state": {"draft": "step-1"},
    "resume_token": "resume-1",
})
```

## Project Structure

```
eirel/
  eirel/
    __init__.py          # Public API exports
    schemas.py           # Core request/response models
    models.py            # Shared data models
    groups.py            # Family definitions and launch mode
    base_agent.py        # Abstract BaseAgent class
    app.py               # MinerApp FastAPI wrapper
    agent_server.py      # Standalone agent server
    helpers.py           # Response builders
    provider.py          # LLM provider client
    manifest.py          # Submission manifest parsing
    packaging.py         # Submission archive packaging
    registration.py      # Registration payload tooling
    request_auth.py      # Inbound validator signature verification
    signing.py           # Outbound request signing (eirel[submit])
    token_signing.py     # Resume token HMAC signing
    a2a.py               # Google A2A protocol support
    compliance.py        # Public compliance test suite
    sample_server.py     # Reference implementation
    cli.py               # `eirel` CLI entry point
    families/
      general_chat/      # General-chat family helpers + tool clients
  examples/
    sample_miner/        # Minimal reference miner
    general_chat_agent/  # General-chat family miner
  tests/                 # Unit and integration tests
```

## License

MIT. See [LICENSE](LICENSE) for details.

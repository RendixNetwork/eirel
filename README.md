# Eirel SDK

The public miner SDK for the EIREL Bittensor subnet. Provides everything miners need to build, register, and run specialized AI agents that compete within execution families.

## Overview

EIREL is a decentralized AI agent marketplace on Bittensor. Miners submit specialized agents that handle tasks across distinct execution families. Validators dispatch tasks, score responses, and set on-chain weights that determine TAO emissions.

The SDK abstracts away subnet integration so miners can focus on agent logic.

### Launch Families

One family is active at launch:

| Family | Role |
|--------|------|
| **general_chat** | Multi-turn conversational assistant with optional web search, across `instant` and `thinking` modes. Backed by owner-routed tool services for web search, URL fetching, and a server-side Python sandbox (with session persistence + file passthrough) for verifiable computation. |

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

The SDK supports two authoring shapes:

1. **Graph agent** (canonical, recommended) — compose a `StateGraph`
   and wrap with `GraphAgent`. Gets you parallel tool dispatch,
   conditional routing, checkpointing, memory, safety guards, and
   tracing as primitives.
2. **Minimal agent** — subclass `BaseAgent` directly, implement
   `async def infer`. Fewer primitives but minimal scaffolding.

Both ship through the same `MinerApp` and produce the same wire
contract — the difference is what's available inside your agent.

### Building a Graph Agent (canonical)

```python
from eirel import (
    AgentCapabilityMetadata, AgentInvocationRequest, AgentInvocationResponse,
    GraphAgent, StateGraph, END, build_agent_app, content_response,
)
from eirel.provider import AgentProviderClient, MinerProviderConfig
import uvicorn

# 1. Define your graph state.
graph_state = {
    "messages": [],     # add_messages reducer (append-only)
    "answer": None,     # replace reducer (overwrite)
}

# 2. Define nodes.
async def llm_node(state, ctx):
    config = MinerProviderConfig.from_env()
    client = AgentProviderClient(config)
    reply = await client.chat_completions({"messages": state["messages"]})
    text = reply["choices"][0]["message"]["content"]
    return {"answer": text}

# 3. Compose the graph.
graph = (
    StateGraph(state_schema=graph_state)
    .add_node("llm", llm_node)
    .set_entry_point("llm")
    .add_edge("llm", END)
    .compile()
)

# 4. Wire to_state / from_state mappers (slim contract → graph state).
def to_state(request: AgentInvocationRequest) -> dict:
    return {
        "messages": [
            *[{"role": m.role, "content": m.content} for m in request.history],
            {"role": "user", "content": request.prompt},
        ],
    }

def from_state(state: dict, request: AgentInvocationRequest) -> AgentInvocationResponse:
    return content_response(
        state["answer"],
        task_id=request.task_id,
        family_id=request.family_id,
    )

# 5. Wrap as GraphAgent.
agent = GraphAgent(
    hotkey="5FHne...",
    endpoint="http://miner.example.com:9000",
    version="1.0.0",
    capabilities=AgentCapabilityMetadata(
        family_id="general_chat",
        description="Conversational assistant (graph)",
        latency_ms_p50=2000,
        estimated_cost_tao=0.1,
    ),
    graph=graph,
    to_state=to_state,
    from_state=from_state,
)

app = build_agent_app(agent)
uvicorn.run(app, host="0.0.0.0", port=9000)
```

The full reference implementation lives at
`examples/graph_general_chat/app.py` — includes parallel tool
dispatch, conditional routing, and the proper `manifest.runtime.kind = "graph"`
wiring.

### Minimal Agent (BaseAgent subclass)

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
        # Slim contract — read prompt + history, NOT primary_goal/subtask
        # (those legacy fields were removed in 0.4.0).
        messages = [
            *[{"role": m.role, "content": m.content} for m in request.history],
            {"role": "user", "content": request.prompt},
        ]
        reply = await self.provider_client.chat_completions({"messages": messages})
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

### Multi-turn fixtures

When the validator dispatches a multi-turn fixture (eval-mode), the
request carries `request.turns: list[Turn]` with the full scripted
transcript in addition to `request.history` (flattened). Agents with
session-memory infrastructure (retrieval, rolling summarization,
structured fact extraction) should prefer `turns` for the structural
signal. Naive miners reading only `history` keep working unchanged.

Tasks with attached content surface the document under
`request.inputs.document_text` (single doc) or
`request.inputs.attached_documents` (list[str]). Agents that handle
document tasks should inspect these keys.

There is intentionally no `category` field on the wire — eval and
product traffic are indistinguishable, so the agent should infer
task shape from `prompt` / `history` / `turns` / `inputs` content.

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
| `AgentInvocationRequest` | Task input: prompt, history, mode, web_search, optional turns + inputs (slim contract — legacy fields removed in 0.4.0) |
| `AgentInvocationResponse` | Task output: status, output dict, artifacts, citations, resume tokens |
| `AgentCapabilityMetadata` | Agent capabilities: family, latency, cost, context limits |
| `AgentHealthStatus` | Health check response |
| `AgentRegistrationMetadata` | Registration payload: hotkey, endpoint, family, version |
| `InvocationConstraints` | Execution constraints: max latency, quality tier, modalities |
| `ContextMessage` | Single conversation message (`role`, `content`, `metadata`) |
| `Turn` | Multi-turn fixture pair (`user`, `assistant`) — eval-mode dispatch |

### Agent Framework

| Surface | Description |
|---------|-------------|
| `GraphAgent` | Canonical authoring class — wraps a compiled `StateGraph` + state mappers (recommended) |
| `BaseAgent` | Abstract class — implement `infer()`, `health()`, `registration()` (minimal alternative) |
| `MinerApp` | FastAPI wrapper with `/v1/chat/completions` and `/v1/agent/infer` |
| `build_agent_app(agent)` | Standalone FastAPI app from a `BaseAgent` (or `GraphAgent`) instance |

### Graph SDK (canonical authoring)

```python
from eirel import StateGraph, END, GraphAgent
from eirel.graph.patterns import (
    SelfConsistencyNode, ReflectionNode, PlannerExecutorNode, ReActNode,
)
```

| Surface | Purpose |
|---------|---------|
| `StateGraph` | Builder: `add_node`, `add_edge`, `add_conditional_edges`, `add_parallel_edges`, `set_entry_point`, `compile` |
| `END` | Terminal node sentinel |
| `Node` / `LLMNode` / `ToolNode` / `BranchNode` | Node primitives |
| `Edge` / `ConditionalEdge` / `ParallelEdge` | Edge primitives |
| `SelfConsistencyNode` | N-sample majority-vote with pluggable aggregator |
| `ReflectionNode` | Generate → critique → revise loop |
| `PlannerExecutorNode` | Decompose into ordered steps, execute, replan on failure |
| `ReActNode` | Thought → Action → Observation loop |

### Stateful primitives

```python
from eirel.checkpoint import (
    Checkpointer, MemoryCheckpointer, SqliteCheckpointer, PostgresCheckpointer,
)
from eirel.memory import RollingSummary, VectorStore
from eirel.safety import Guard, ChainedGuard, NoopGuard, GuardVerdict
from eirel.tracing import Tracer, NoopTracer, StdoutTracer
from eirel.structured import StructuredOutputNode
```

Optional adapters under extras:
- `eirel[sqlite]` — `SqliteCheckpointer`
- `eirel[postgres]` — `PostgresCheckpointer`
- `eirel[chroma]` / `eirel[qdrant]` — vector store adapters
- `eirel[safety-llamaguard]` / `eirel[safety-nemo]` — safety guards
- `eirel[tracing-langfuse]` — Langfuse tracer

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
| `UrlFetchTool` | Specific-URL extraction (HTML→text) | `EIREL_URL_FETCH_URL`, `EIREL_URL_FETCH_TOKEN` |
| `SandboxTool` | Server-side Python sandbox for verifiable computation; supports `session_id` for kernel reuse and `attachments` / `files` for FS passthrough | `EIREL_SANDBOX_URL`, `EIREL_SANDBOX_TOKEN` |

`RetryPolicy` and `FallbackChain` are available on the catalog —
`execute_with_policy(name, args, policy=)` and `execute_many` accept
per-call retry/fallback so transient tool errors recover without
bespoke graph wiring.

The reference integration lives at `examples/graph_general_chat/app.py` —
builds a `GeneralChatToolCatalog`, invokes `web_search` / `url_fetch` /
`sandbox` from graph nodes, injects tool results into state, and records
citations on the `GeneralChatResponse`.

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

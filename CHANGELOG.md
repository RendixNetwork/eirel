# Changelog

All notable changes to the Eirel SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.3.1] - 2026-04-28

### Added

- **`JobIdContextMiddleware`** (in `eirel.runtime_context`) — captures
  the per-request `X-Eirel-Job-Id` header set by owner-api into a
  `ContextVar`. `AgentProviderClient` (proxy mode) consults the
  context on each call and forwards the captured job_id to the subnet
  provider-proxy as the LLM-call attribution key, overriding the
  deployment-sticky `EIREL_PROVIDER_PROXY_JOB_ID` env-var default.
- Mounted automatically on `MinerApp` and `build_agent_app`. No agent
  app changes required.

### Why

Owner-api stamps a unique `task-eval=<turn_id>;deployment=<id>` tag
per validator request and reads back the provider-proxy ledger after
the response stream closes, injecting `metadata.proxy_cost_usd` into
the final `done` chunk. Without this middleware the miner would still
charge under the deployment-sticky job_id and per-task cost
attribution would silently lose precision. With 0.3.1 installed on
the miner, owner-api's per-task cost lookup matches the actual LLM
spend for that single task evaluation.

The header is set by owner-api on the only path validator traffic
takes to the miner pod, so the miner cannot tamper with attribution
short of running a custom SDK that ignores the header (which is
visible to operators as `metadata.proxy_cost_absent=true`).

## [0.3.0] - 2026-04-28

### Architecture

The subnet is repositioning around a **multi-agent / multi-family**
shape. A future DAG orchestrator will route consumer requests across
specialist family agents. `general_chat` is the first family — purely
specialized for general chat, no orchestration responsibility. The
0.3.0 release reshapes the family-agent contract to match: stateless,
specialty-only, no session/orchestration leakage.

### Changed (BREAKING)

- **Slim agent invocation contract.** `AgentInvocationRequest` now
  exposes flat `prompt`, `mode`, `web_search`, `history`, `turn_id`
  fields. The legacy bag (`primary_goal`, `subtask`, `inputs`,
  `context_history`, `family_id`, plus the workflow-DAG fields) remains
  on the model **for one release only** and is auto-folded into the
  new fields by a model validator. Validators / orchestrators now
  populate **both** during the migration window. After 0.4.0 the
  legacy fields are removed.
- **`AgentInvocationResponse.tool_calls` removed.** The top-level field
  conflated "what the LLM emitted" with "what the agent's Python ran"
  and clashed with the OpenAI naming. Executed tool calls now live
  under `metadata["executed_tool_calls"]`. Same change on
  `StreamChunk` — the trailing `done` chunk no longer carries
  top-level `tool_calls`.
- **Answer-key leak fixed.** The validator's `_build_body` previously
  merged `task.expected_output` into the miner-visible payload via
  `inputs.expected_output`. Datasets that populated this field
  effectively shipped the grading key on every call. The slim body
  drops `expected_output` and `metadata` entirely from the wire.

### Added

- `ChatRequest` (consumer-chat-api) gains `mode` (`instant`/`thinking`)
  and `web_search` fields, propagated through to the family agent on
  every turn.
- `context_from_request()` reads the new flat fields first and falls
  back to the 0.2.x shape, so an agent built against 0.3.0 keeps
  parsing legacy traffic during the migration.

### Migration notes

- Miners using `MinerApp` / `BaseAgent` should re-submit on 0.3.0 to
  read the slim contract directly. 0.2.x miners continue to receive
  legacy fields populated by the validator and the orchestrator.
- Anything reading `response.tool_calls` must move to
  `response.metadata["executed_tool_calls"]`. The `done` chunk uses
  the same key.

## [0.2.4] - 2026-04-27

### Added

- **`MinerApp` exposes `POST /v1/agent/infer/stream`** — the streaming
  invocation route was missing from `MinerApp` in 0.2.3 (it was only on
  `build_agent_app`). Validators and consumer-chat-api were 404'ing on
  every miner using `MinerApp` and falling back to the unary path. This
  release adds the route. With no other change, miners get a default
  fallback that buffers the unary handler's response and emits one
  `delta` + `done` chunk — same wire contract as `build_agent_app`.
- **`MinerApp(agent_stream_handler=...)`** — optional constructor arg.
  Pass an async generator that yields `StreamChunk | dict` objects to
  enable real token-by-token streaming. The default fallback is used
  when this is omitted.
- **`AgentProviderClient.chat_completions_stream(payload)`** — async
  generator that yields content deltas as the LLM produces them. Direct
  mode passes `stream=True` to chutes / openai / openrouter and parses
  SSE. Proxy mode (subnet provider-proxy) currently falls back to one
  buffered chunk; real proxy-side streaming is a follow-up when the
  provider-proxy adds an SSE pass-through endpoint. Anthropic also
  falls back to non-streaming until per-provider SSE shapes are wired.

### Why

0.2.3 added the streaming route only to `build_agent_app`, but the
common pattern — including the `examples/general_chat_agent` — uses
`MinerApp`. So in practice every production miner returned 404 on the
new endpoint and exercised the unary fallback. 0.2.4 closes the gap so
`MinerApp`-based agents satisfy the streaming contract by default and
can opt into real token streaming via `agent_stream_handler`.

## [0.2.3] - 2026-04-25

### Added

- **`POST /v1/agent/infer/stream`** — new streaming invocation endpoint
  on the agent server. Returns NDJSON (`application/x-ndjson`); each line
  parses to a `StreamChunk` with `event` ∈ `delta` | `tool_call` |
  `citation` | `done`. The stream MUST end with a `done` chunk; whatever
  the receiver accumulates from `delta.text` chunks is the final answer.
- **`StreamChunk`** schema (in `eirel.schemas`).
- **`BaseAgent.infer_stream(request)`** async generator. Default
  implementation calls non-streaming `infer()` and yields the whole
  answer as a single `delta` followed by `done`, so existing agents
  that only override `infer()` keep working without code changes —
  but their first-token latency will equal their full completion
  latency. Real streaming agents should override `infer_stream` and
  yield `delta` chunks as soon as the underlying LLM emits tokens;
  that's the only way to satisfy the validator's 10s TTFB SLA gate.

### Why

The consumer chat UI streams tokens to end users, so miners need a
streaming endpoint. The validator exercises the same path 
so a streaming-only regression is caught before it reaches users.

## [0.2.1] - 2026-04-24

### Added

- **`Citation.title`** — optional `title: str | None` field on the
  `Citation` model in `eirel.families.general_chat`. It flows through
  `TraceRecorder.record_citation` → `GeneralChatResponse.citations[*]`
  so validators and dashboards can show the source page title next to
  each URL. Backwards-compatible: defaults to `None`, existing
  `Citation(url=..., snippet=..., tool_name=...)` calls keep working.

### Changed

- **`general_chat_agent` example forwards web_search titles** —
  `_run_turn` already extracted `title` from each tool result for the
  system-prompt snippet but dropped it when constructing the
  `Citation`, so downstream consumers only saw URLs. Now passes
  `title=title or None` into the `Citation(...)` call.

## [0.2.0] - 2026-04-18

### Fixed

- **`WebSearchTool` endpoint and payload** — the tool client now calls
  `POST /v1/search` with `{query, top_k}` to match the tool-service
  contract; prior versions hit a non-existent `/v1/tools/web_search`
  path and always 404'd. Miners on 0.1.x who relied on web_search
  silently got zero citations. **Breaking** for anyone mocking the tool
  HTTP layer directly.
- **Reasoning-model response length** — `_run_turn` in the
  `general_chat_agent` example now sends
  `max_tokens = output_tokens + reasoning_tokens` with a 2048 floor.
  At the old `output_tokens` cap, reasoning-only models (e.g.
  Kimi-K2.5-TEE) burned the entire budget on hidden reasoning and
  returned `content=""` → cascading 502s at the subnet proxy.

### Added

- **Per-tool authentication tokens** — the example now prefers
  `EIREL_WEB_SEARCH_TOKEN`, `EIREL_SEMANTIC_SCHOLAR_TOKEN`,
  `EIREL_X_API_TOKEN`, `EIREL_SANDBOX_TOKEN` over the shared
  `EIREL_TOOL_SERVICE_TOKEN` fallback, letting each tool service
  enforce its own auth secret.
- **Miner-level job id for tool cost attribution** — `_build_catalog`
  uses `EIREL_PROVIDER_PROXY_JOB_ID` (which the runtime sets to
  `miner-<deployment_id>`) so LLM and tool costs accumulate on the
  same provider-proxy record that the subnet reads when aggregating
  `DeploymentScoreRecord.tool_cost_usd`.

### Changed

- **`httpx` dependency range loosened** from `<0.29` to `<1` so fresh
  installs don't break when httpx ships its next minor release.

## [0.1.0] - 2026-04-17

Initial public release of the EIREL miner SDK.

### Added
- `BaseAgent` abstract class with `infer`, `health`, and `registration` hooks.
- `MinerApp` FastAPI wrapper with `/v1/chat/completions` and `/v1/agent/infer` endpoints.
- `build_agent_app` helper for standalone `BaseAgent` FastAPI apps.
- `AgentProviderClient` with `MinerProviderConfig.from_env()` — supports OpenAI, Anthropic, OpenRouter, and Chutes backends.
- Inbound request authentication — `Signer` / `load_signer` via `eirel[submit]` extra, validated with `X-Hotkey` / `X-Signature` / `X-Timestamp` / `X-Request-Id` headers.
- Resume-token HMAC-SHA256 signing for multi-turn workflows.
- `general_chat` family support — budget, context, response helpers, and tool clients (web search, Semantic Scholar, X API, sandbox).
- Single `eirel` CLI with subcommands: `submit`, `status`, `package`, `compliance`, `register`, `serve`, `sample`.
- `py.typed` marker — SDK ships typed per PEP 561.

[Unreleased]: https://github.com/RendixNetwork/eirel/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/RendixNetwork/eirel/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/RendixNetwork/eirel/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/RendixNetwork/eirel/releases/tag/v0.1.0

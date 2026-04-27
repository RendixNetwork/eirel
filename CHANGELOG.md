# Changelog

All notable changes to the Eirel SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

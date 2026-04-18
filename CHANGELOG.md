# Changelog

All notable changes to the Eirel SDK are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/RendixNetwork/eirel/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/RendixNetwork/eirel/releases/tag/v0.2.0
[0.1.0]: https://github.com/RendixNetwork/eirel/releases/tag/v0.1.0

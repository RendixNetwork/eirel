from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

_logger = logging.getLogger(__name__)


class ProviderRequestError(RuntimeError):
    """Raised when the upstream LLM provider returns a non-2xx response.

    The message is sanitized — the upstream request headers (which carry the
    API key) are intentionally dropped from the exception chain via ``from None``
    so they cannot leak into logs.
    """


class ProviderQuotaExceeded(ProviderRequestError):
    """Raised when the miner-side request or token quota has been exhausted."""


ProviderId = Literal["openai", "anthropic", "openrouter", "chutes"]


class MinerProviderConfig(BaseModel):
    provider: ProviderId
    model: str
    mode: Literal["direct", "proxy", "auto"] = "auto"
    api_key: str | None = None
    base_url: str | None = None
    subnet_proxy_url: str | None = None
    subnet_proxy_token: str | None = None
    subnet_proxy_job_id: str | None = None
    run_budget_usd: float | None = None
    allowed_providers: list[str] = Field(default_factory=list)
    max_requests: int = 24
    max_total_tokens: int = 60000
    max_wall_clock_seconds: int = 300
    per_request_timeout_seconds: int = 60

    def validate_for_runtime(self) -> None:
        """Raise if the config is not usable for the requested mode.

        Call this at miner startup to fail fast on missing credentials
        instead of surfacing a cryptic 401 on the first validator request.
        """
        if self.mode == "proxy":
            if not self.subnet_proxy_url or not self.subnet_proxy_token:
                raise RuntimeError(
                    "MinerProviderConfig: mode='proxy' requires EIREL_PROVIDER_PROXY_URL "
                    "and EIREL_PROVIDER_PROXY_TOKEN to be set"
                )
            return
        if self.mode == "direct":
            if not self.api_key:
                raise RuntimeError(
                    f"MinerProviderConfig: mode='direct' with provider={self.provider!r} "
                    "requires MINER_API_KEY (or API_KEY) to be set"
                )
            return
        # mode == "auto"
        has_proxy = bool(self.subnet_proxy_url and self.subnet_proxy_token)
        has_direct = bool(self.api_key)
        if not has_proxy and not has_direct:
            raise RuntimeError(
                "MinerProviderConfig: mode='auto' requires either (EIREL_PROVIDER_PROXY_URL + "
                "EIREL_PROVIDER_PROXY_TOKEN) or MINER_API_KEY to be set"
            )

    @classmethod
    def from_env(
        cls,
        *,
        provider: str | None = None,
        model: str | None = None,
    ) -> "MinerProviderConfig":
        selected_provider = provider or os.getenv("MINER_PROVIDER") or os.getenv("PROVIDER") or "openai"
        selected_model = model or os.getenv("MINER_MODEL") or os.getenv("MODEL_NAME") or "gpt-4o-mini"
        budget_raw = os.getenv("EIREL_RUN_BUDGET_USD")
        budget: float | None = None
        if budget_raw is not None and budget_raw.strip():
            try:
                budget = float(budget_raw)
            except ValueError:
                budget = None
        def _int_env(key: str, default: int) -> int:
            raw = os.getenv(key, "")
            if not raw.strip():
                return default
            try:
                return int(raw)
            except ValueError:
                return default

        max_requests = _int_env("EIREL_PROVIDER_MAX_REQUESTS", 24)
        max_total_tokens = _int_env("EIREL_PROVIDER_MAX_TOTAL_TOKENS", 60_000)
        max_wall_clock_seconds = _int_env("EIREL_PROVIDER_MAX_WALL_CLOCK_SECONDS", 300)
        per_request_timeout_seconds = _int_env("EIREL_PROVIDER_PER_REQUEST_TIMEOUT_SECONDS", 60)
        return cls(
            provider=selected_provider,  # type: ignore[arg-type]
            model=selected_model,
            mode=os.getenv("MINER_PROVIDER_MODE", "auto"),  # type: ignore[arg-type]
            api_key=os.getenv("MINER_API_KEY") or os.getenv("API_KEY"),
            base_url=os.getenv("MINER_API_BASE_URL") or os.getenv("API_ENDPOINT"),
            subnet_proxy_url=os.getenv("EIREL_PROVIDER_PROXY_URL"),
            subnet_proxy_token=os.getenv("EIREL_PROVIDER_PROXY_TOKEN"),
            subnet_proxy_job_id=os.getenv("EIREL_PROVIDER_PROXY_JOB_ID"),
            run_budget_usd=budget,
            max_requests=max_requests,
            max_total_tokens=max_total_tokens,
            max_wall_clock_seconds=max_wall_clock_seconds,
            per_request_timeout_seconds=per_request_timeout_seconds,
            allowed_providers=[
                item.strip()
                for item in os.getenv("EIREL_PROVIDER_ALLOWED_PROVIDERS", "").split(",")
                if item.strip()
            ],
        )


class AgentProviderClient:
    def __init__(self, config: MinerProviderConfig):
        self.config = config
        self._request_count = 0
        self._token_count = 0

    def reset_quota(self) -> None:
        self._request_count = 0
        self._token_count = 0

    @property
    def request_count(self) -> int:
        return self._request_count

    @property
    def token_count(self) -> int:
        return self._token_count

    def _check_quota(self) -> None:
        if self._request_count >= self.config.max_requests:
            raise ProviderQuotaExceeded(
                f"provider request quota exhausted "
                f"({self._request_count}/{self.config.max_requests})"
            )
        if self._token_count >= self.config.max_total_tokens:
            raise ProviderQuotaExceeded(
                f"provider token quota exhausted "
                f"({self._token_count}/{self.config.max_total_tokens})"
            )

    @staticmethod
    def _extract_usage_tokens(parsed: dict[str, Any]) -> int:
        usage = parsed.get("usage") if isinstance(parsed, dict) else None
        if not isinstance(usage, dict):
            return 0
        value = usage.get("total_tokens")
        if isinstance(value, int):
            return value
        # Anthropic-style: input_tokens + output_tokens
        total = 0
        for key in ("input_tokens", "output_tokens"):
            item = usage.get(key)
            if isinstance(item, int):
                total += item
        return total

    async def chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._check_quota()
        if self._use_proxy():
            result = await self._proxy_chat_completions(payload)
        else:
            result = await self._direct_chat_completions(payload)
        self._request_count += 1
        self._token_count += self._extract_usage_tokens(result)
        return result

    async def chat_completions_stream(
        self, payload: dict[str, Any],
    ) -> AsyncIterator[str]:
        """Stream content tokens from the LLM as they arrive.

        Yields successive ``str`` deltas (the ``content`` field of each
        OpenAI ``choices[0].delta``). The caller is responsible for
        accumulating them. Raises ``ProviderRequestError`` on upstream
        failure.

        - **Direct mode**: passes ``stream=True`` to the upstream provider
          (chutes / openai / openrouter / anthropic), iterates SSE chunks,
          yields content as it arrives.
        - **Proxy mode**: the subnet provider-proxy currently buffers
          upstream responses and returns a single JSON body, so we fall
          back to the unary path and yield the full content as one chunk.
          Real streaming through the proxy will arrive in a follow-up
          when the proxy adds an SSE pass-through endpoint.
        """
        self._check_quota()
        if self._use_proxy():
            # Graceful fallback — yield the full answer in one chunk.
            result = await self._proxy_chat_completions(payload)
            self._request_count += 1
            self._token_count += self._extract_usage_tokens(result)
            content = ""
            try:
                content = result["choices"][0]["message"].get("content") or ""
            except (KeyError, IndexError, TypeError):
                content = ""
            if content:
                yield content
            return

        async for chunk in self._direct_chat_completions_stream(payload):
            yield chunk

    async def _direct_chat_completions_stream(
        self, payload: dict[str, Any],
    ) -> AsyncIterator[str]:
        if self.config.provider == "anthropic":
            # Anthropic SSE is a different shape; for v0.2.4 we ship
            # OpenAI-compatible streaming only (chutes / openai /
            # openrouter). Fall back to non-streaming for anthropic.
            result = await self._direct_chat_completions(payload)
            self._request_count += 1
            self._token_count += self._extract_usage_tokens(result)
            content = ""
            try:
                content = result["choices"][0]["message"].get("content") or ""
            except (KeyError, IndexError, TypeError):
                content = ""
            if content:
                yield content
            return

        url, headers, body = self._direct_request_parts(payload)
        body["stream"] = True
        # Some providers also want stream_options.include_usage so the
        # final chunk carries token counts; harmless if ignored.
        body.setdefault("stream_options", {"include_usage": True})

        usage_total: int = 0
        async with httpx.AsyncClient(
            timeout=self.config.per_request_timeout_seconds,
        ) as client:
            async with client.stream(
                "POST", url, json=body, headers=headers,
            ) as response:
                self._raise_for_status(response)
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data_str = line[len("data:"):].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        _logger.warning(
                            "malformed SSE chunk from %s: %r",
                            self.config.provider, data_str[:120],
                        )
                        continue
                    # Final chunk may carry usage; book-keep it.
                    usage = chunk.get("usage")
                    if isinstance(usage, dict):
                        usage_total = self._extract_usage_tokens({"usage": usage})
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    delta = choices[0].get("delta") or {}
                    text = delta.get("content")
                    if isinstance(text, str) and text:
                        yield text
        self._request_count += 1
        self._token_count += usage_total

    def _use_proxy(self) -> bool:
        if self.config.mode == "proxy":
            return True
        if self.config.mode == "direct":
            return False
        return bool(self.config.subnet_proxy_url and self.config.subnet_proxy_token)

    async def _proxy_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.subnet_proxy_url or not self.config.subnet_proxy_token:
            raise RuntimeError("subnet proxy mode requires proxy url and token")
        # Per-request job_id from owner-api wins over the
        # deployment-sticky env-var default. See runtime_context.py.
        from eirel.runtime_context import get_active_job_id
        job_id = (
            get_active_job_id()
            or self.config.subnet_proxy_job_id
            or "miner-sdk-job"
        )
        headers = {
            "Authorization": f"Bearer {self.config.subnet_proxy_token}",
            "X-Eirel-Job-Id": job_id,
        }
        if self.config.run_budget_usd is not None:
            headers["X-Eirel-Run-Budget-Usd"] = f"{self.config.run_budget_usd:.6f}"
        async with httpx.AsyncClient(timeout=self.config.per_request_timeout_seconds) as client:
            response = await client.post(
                f"{self.config.subnet_proxy_url.rstrip('/')}/v1/chat/completions",
                json={
                    "provider": self.config.provider,
                    "model": self.config.model,
                    "payload": payload,
                    "max_requests": self.config.max_requests,
                    "max_total_tokens": self.config.max_total_tokens,
                    "max_wall_clock_seconds": self.config.max_wall_clock_seconds,
                    "per_request_timeout_seconds": self.config.per_request_timeout_seconds,
                },
                headers=headers,
            )
            self._raise_for_status(response)
            return dict(response.json()["upstream_response"])

    async def _direct_chat_completions(self, payload: dict[str, Any]) -> dict[str, Any]:
        url, headers, body = self._direct_request_parts(payload)
        async with httpx.AsyncClient(timeout=self.config.per_request_timeout_seconds) as client:
            response = await client.post(url, json=body, headers=headers)
            self._raise_for_status(response)
            parsed = response.json()
        if self.config.provider == "anthropic":
            return self._anthropic_to_openai(parsed)
        return dict(parsed)

    def _raise_for_status(self, response: httpx.Response) -> None:
        if response.is_success:
            return
        # `from None` drops the original exception chain so the outgoing request
        # headers (which carry the API key) cannot leak into logs.
        body_excerpt = (response.text or "")[:500]
        message = (
            f"provider {self.config.provider} returned {response.status_code}: {body_excerpt}"
        )
        _logger.warning("%s", message)
        raise ProviderRequestError(message) from None

    def _direct_request_parts(self, payload: dict[str, Any]) -> tuple[str, dict[str, str], dict[str, Any]]:
        if not self.config.api_key:
            raise RuntimeError(
                f"MINER_API_KEY is not set for direct provider {self.config.provider!r}; "
                "set MINER_API_KEY (or API_KEY), switch to proxy mode, or call "
                "MinerProviderConfig.validate_for_runtime() at startup"
            )
        defaults = {
            "openai": ("https://api.openai.com/v1/chat/completions", {"Authorization": f"Bearer {self.config.api_key}"}),
            "openrouter": ("https://openrouter.ai/api/v1/chat/completions", {"Authorization": f"Bearer {self.config.api_key}"}),
            "chutes": ("https://api.chutes.ai/v1/chat/completions", {"Authorization": f"Bearer {self.config.api_key}"}),
            "anthropic": ("https://api.anthropic.com/v1/messages", {"x-api-key": self.config.api_key, "anthropic-version": "2023-06-01"}),
        }
        default_url, headers = defaults[self.config.provider]
        url = self.config.base_url or default_url
        if self.config.provider == "anthropic":
            return url, headers, self._openai_to_anthropic(payload)
        headers["Content-Type"] = "application/json"
        body = dict(payload)
        body["model"] = self.config.model
        return url, headers, body

    def _openai_to_anthropic(self, payload: dict[str, Any]) -> dict[str, Any]:
        system_parts: list[str] = []
        messages: list[dict[str, Any]] = []
        for message in payload.get("messages", []):
            role = message.get("role")
            content = message.get("content")
            if role == "system" and isinstance(content, str):
                system_parts.append(content)
                continue
            if role in {"user", "assistant"}:
                messages.append({"role": role, "content": content or ""})
        return {
            "model": self.config.model,
            "system": "\n".join(system_parts) or None,
            "messages": messages,
            "max_tokens": payload.get("max_tokens") or 512,
            "temperature": payload.get("temperature", 0),
        }

    def _anthropic_to_openai(self, payload: dict[str, Any]) -> dict[str, Any]:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        for item in payload.get("content", []):
            if not isinstance(item, dict):
                continue
            block_type = item.get("type")
            if block_type == "text":
                text_parts.append(str(item.get("text", "")))
            elif block_type == "tool_use":
                tool_calls.append({
                    "id": item.get("id", ""),
                    "type": "function",
                    "function": {
                        "name": item.get("name", ""),
                        "arguments": json.dumps(item.get("input", {})),
                    },
                })
        message: dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        return {
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": payload.get("stop_reason") or "stop",
                }
            ]
        }

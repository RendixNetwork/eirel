from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from eirel.families.general_chat._retry import retry_with_jitter


class ToolServiceError(RuntimeError):
    """Raised when an owner-api tool service request fails."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(slots=True, frozen=True)
class ToolServiceConfig:
    base_url: str
    api_token: str = ""
    job_id: str | None = None
    max_requests: int = 12
    timeout_seconds: float = 15.0


class ToolServiceClient:
    def __init__(
        self,
        config: ToolServiceConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout_seconds,
                transport=self.transport,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def request(self, *, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.config.base_url:
            raise ToolServiceError(
                "tool_environment_missing",
                "tool service base URL is not configured",
            )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_token:
            headers["Authorization"] = f"Bearer {self.config.api_token}"
        if self.config.job_id:
            headers["X-Eirel-Job-Id"] = self.config.job_id
        headers["X-Eirel-Max-Requests"] = str(self.config.max_requests)
        client = self._get_client()

        async def _do_request() -> dict[str, Any]:
            response = await client.post(
                f"{self.config.base_url.rstrip('/')}{path}",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            return response.json()

        try:
            parsed = await retry_with_jitter(_do_request)
        except (httpx.HTTPError, httpx.InvalidURL, ValueError, TypeError) as exc:
            raise ToolServiceError(
                "tool_service_failure",
                f"tool service request failed for {path}: {exc}",
            ) from exc
        if not isinstance(parsed, dict):
            raise ToolServiceError(
                "tool_response_invalid",
                "tool service response must be a JSON object",
            )
        return parsed

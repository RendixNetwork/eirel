from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from pydantic import ValidationError

from eirel.helpers import (
    chat_payload_from_agent_request,
    infer_response_from_chat_payload,
    validate_agent_request,
    validate_request,
)
from eirel.request_auth import verify_request_dependency

if TYPE_CHECKING:
    from eirel.provider import MinerProviderConfig

_logger = logging.getLogger(__name__)


class MinerApp:
    def __init__(
        self,
        *,
        title: str,
        handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
        agent_handler: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]] | None = None,
        provider_config: "MinerProviderConfig | None" = None,
    ):
        self.handler = handler
        self.agent_handler = agent_handler
        self.provider_config = provider_config
        self.app = FastAPI(title=title)

        @self.app.get("/healthz")
        async def healthz(deep: int = Query(default=0)) -> dict[str, str]:
            if not deep:
                return {"status": "ok"}
            if self.provider_config is None:
                return {"status": "ok", "provider": "unknown"}
            try:
                self.provider_config.validate_for_runtime()
            except RuntimeError as exc:
                return {"status": "degraded", "provider": f"misconfigured: {exc}"}
            return {"status": "ok", "provider": "configured"}

        @self.app.post("/v1/chat/completions")
        async def chat_completions(
            payload: dict[str, Any] = Depends(verify_request_dependency),
        ) -> dict[str, Any]:
            try:
                validate_request(payload)
            except ValidationError as exc:
                _logger.warning("chat_completions validation failed: %s", exc.error_count())
                raise HTTPException(status_code=400, detail=str(exc)) from None
            return await self.handler(payload)

        @self.app.post("/v1/agent/infer")
        async def agent_infer(
            payload: dict[str, Any] = Depends(verify_request_dependency),
        ) -> dict[str, Any]:
            try:
                request = validate_agent_request(payload)
            except ValidationError as exc:
                _logger.warning("agent_infer validation failed: %s", exc.error_count())
                raise HTTPException(status_code=400, detail=str(exc)) from None
            if self.agent_handler is not None:
                return await self.agent_handler(payload)
            return infer_response_from_chat_payload(
                request=request,
                payload=await self.handler(chat_payload_from_agent_request(request)),
            )

    def fastapi_app(self) -> FastAPI:
        return self.app


__all__ = ["MinerApp"]

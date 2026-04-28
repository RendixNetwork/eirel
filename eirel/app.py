from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TYPE_CHECKING, Any

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

from eirel.helpers import (
    chat_payload_from_agent_request,
    infer_response_from_chat_payload,
    validate_agent_request,
    validate_request,
)
from eirel.request_auth import verify_request_dependency
from eirel.schemas import StreamChunk

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
        agent_stream_handler: (
            Callable[[dict[str, Any]], AsyncIterator[StreamChunk | dict[str, Any]]] | None
        ) = None,
        provider_config: "MinerProviderConfig | None" = None,
    ):
        self.handler = handler
        self.agent_handler = agent_handler
        self.agent_stream_handler = agent_stream_handler
        self.provider_config = provider_config
        self.app = FastAPI(title=title)
        # Capture per-request job_id from owner-api (X-Eirel-Job-Id) so
        # AgentProviderClient forwards it on outbound provider-proxy
        # calls. See eirel.runtime_context for the full rationale.
        from eirel.runtime_context import JobIdContextMiddleware
        self.app.add_middleware(JobIdContextMiddleware)

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

        @self.app.post("/v1/agent/infer/stream")
        async def agent_infer_stream(
            payload: dict[str, Any] = Depends(verify_request_dependency),
        ):
            """Streaming agent invocation, NDJSON.

            If the miner registered an `agent_stream_handler`, that's the
            source of chunks. Otherwise we fall back to the unary handler
            and yield its full response as a single `delta` followed by
            `done` — same shape as the validator + consumer chat code
            expect, but TTFB will equal full completion latency.
            """
            try:
                request = validate_agent_request(payload)
            except ValidationError as exc:
                _logger.warning(
                    "agent_infer_stream validation failed: %s", exc.error_count(),
                )
                raise HTTPException(status_code=400, detail=str(exc)) from None

            async def _ndjson():
                try:
                    if self.agent_stream_handler is not None:
                        async for chunk in self.agent_stream_handler(payload):
                            if not isinstance(chunk, StreamChunk):
                                chunk = StreamChunk.model_validate(chunk)
                            yield (
                                chunk.model_dump_json(exclude_none=True) + "\n"
                            ).encode("utf-8")
                        return

                    # Unary fallback: run the same code path as /v1/agent/infer.
                    if self.agent_handler is not None:
                        infer_payload = await self.agent_handler(payload)
                    else:
                        infer_payload = infer_response_from_chat_payload(
                            request=request,
                            payload=await self.handler(
                                chat_payload_from_agent_request(request),
                            ),
                        )
                    output = (
                        infer_payload.get("output") if isinstance(infer_payload, dict) else {}
                    ) or {}
                    text = ""
                    for key in ("answer", "response", "text", "content", "message"):
                        value = output.get(key)
                        if isinstance(value, str) and value:
                            text = value
                            break
                    if text:
                        delta = StreamChunk(event="delta", text=text)
                        yield (
                            delta.model_dump_json(exclude_none=True) + "\n"
                        ).encode("utf-8")
                    done = StreamChunk(
                        event="done",
                        output=output,
                        citations=list(infer_payload.get("citations") or []),
                        tool_calls=list(infer_payload.get("tool_calls") or []),
                        status=infer_payload.get("status") or "completed",
                        error=infer_payload.get("error"),
                        metadata=dict(infer_payload.get("metadata") or {}),
                    )
                    yield (
                        done.model_dump_json(exclude_none=True) + "\n"
                    ).encode("utf-8")
                except Exception as exc:  # noqa: BLE001
                    _logger.exception("agent_infer_stream raised: %s", exc)
                    err = StreamChunk(
                        event="done", status="failed", error=str(exc),
                    )
                    yield (
                        err.model_dump_json(exclude_none=True) + "\n"
                    ).encode("utf-8")

            return StreamingResponse(
                _ndjson(),
                media_type="application/x-ndjson",
                headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
            )

    def fastapi_app(self) -> FastAPI:
        return self.app


__all__ = ["MinerApp"]

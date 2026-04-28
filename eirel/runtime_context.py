"""Per-request context propagated from the FastAPI ASGI layer through
:class:`AgentProviderClient` calls to the subnet provider-proxy.

The owner-api proxy stamps an ``X-Eirel-Job-Id`` header on every
miner-bound request — the value uniquely identifies *this validator
task evaluation against this miner* (e.g. ``task-eval=<turn_id>;
deployment=<deployment_id>``). The miner-side SDK reads that header in
:class:`JobIdContextMiddleware` and stashes it in a ContextVar; when
the agent code subsequently calls ``AgentProviderClient.chat_completions``
in proxy mode, the client picks up the per-call job_id from the
context and forwards it to provider-proxy as ``X-Eirel-Job-Id``,
overriding the deployment-sticky default from
``EIREL_PROVIDER_PROXY_JOB_ID``.

Why this matters: the provider-proxy ledgers cost per ``job_id``, and
owner-api reads that ledger after the response stream closes to inject
``metadata.proxy_cost_usd`` into the final ``done`` chunk. A stable
per-task job_id is what makes that per-task cost attribution possible.
The miner cannot change or remove the header — it's set by owner-api,
which is the only path validator traffic takes to the pod.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# Stores the active job_id for the current request. Reset to None
# between requests by the middleware so cross-request leakage is
# impossible.
_JOB_ID: ContextVar[str | None] = ContextVar("eirel_job_id", default=None)


def get_active_job_id() -> str | None:
    """Return the per-request job_id stamped by owner-api, if any."""
    return _JOB_ID.get()


def _set_active_job_id(value: str | None) -> object:
    """Set the per-request job_id and return the reset token. Internal
    helper for the middleware — agent code should not call this
    directly."""
    return _JOB_ID.set(value)


def _reset_active_job_id(token: Any) -> None:
    _JOB_ID.reset(token)


class JobIdContextMiddleware:
    """ASGI middleware that captures ``X-Eirel-Job-Id`` per request.

    Mounted on :class:`MinerApp` and ``build_agent_app`` automatically
    so authors don't need to wire it. No-op for non-HTTP scopes
    (lifespan, websocket).
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        job_id: str | None = None
        for raw_name, raw_value in scope.get("headers", []):
            try:
                name = raw_name.decode("latin-1").lower()
            except (UnicodeDecodeError, AttributeError):
                continue
            if name == "x-eirel-job-id":
                try:
                    job_id = raw_value.decode("latin-1") or None
                except (UnicodeDecodeError, AttributeError):
                    job_id = None
                break
        token = _set_active_job_id(job_id)
        try:
            await self.app(scope, receive, send)
        finally:
            _reset_active_job_id(token)


__all__ = [
    "JobIdContextMiddleware",
    "get_active_job_id",
]

from __future__ import annotations

import argparse
import logging
from typing import Any

import uvicorn
from fastapi import Depends, FastAPI, HTTPException
from pydantic import ValidationError

from eirel.base_agent import BaseAgent
from eirel.request_auth import verify_request_dependency
from eirel.schemas import AgentInvocationRequest

_logger = logging.getLogger(__name__)


def build_agent_app(agent: BaseAgent) -> FastAPI:
    app = FastAPI(title=f"{agent.capabilities.family_id}-agent")

    @app.get("/healthz")
    async def healthz():
        return (await agent.health()).model_dump()

    @app.get("/v1/registration")
    async def registration(_: dict[str, Any] = Depends(verify_request_dependency)):
        return agent.registration().model_dump()

    @app.post("/v1/agent/infer")
    async def infer(
        payload: dict[str, Any] = Depends(verify_request_dependency),
    ):
        try:
            request = AgentInvocationRequest.model_validate(payload)
        except ValidationError as exc:
            _logger.warning("agent infer validation failed: %s", exc.error_count())
            raise HTTPException(status_code=400, detail=str(exc)) from None
        return (await agent.infer(request)).model_dump()

    return app


def configure_parser(sp: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = sp.add_parser("serve", help="Run a BaseAgent subclass as a FastAPI server")
    parser.add_argument("--app", required=True, help="Import path to an instantiated BaseAgent (e.g. myproject.miners:my_agent)")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=9000, help="Bind port")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> None:
    if ":" not in args.app:
        raise SystemExit(
            "--app must be of the form 'module.path:object_name' (e.g. eirel.sample_server:agent)"
        )
    module_name, object_name = args.app.split(":", maxsplit=1)
    if not module_name or not object_name:
        raise SystemExit(
            "--app must be of the form 'module.path:object_name' (e.g. eirel.sample_server:agent)"
        )
    try:
        module = __import__(module_name, fromlist=[object_name])
    except ImportError as exc:
        raise SystemExit(f"failed to import module '{module_name}': {exc}") from None
    try:
        agent = getattr(module, object_name)
    except AttributeError:
        raise SystemExit(
            f"module '{module_name}' has no attribute '{object_name}'"
        ) from None
    if not isinstance(agent, BaseAgent):
        raise SystemExit(
            f"{args.app} did not resolve to a BaseAgent instance (got {type(agent).__name__})"
        )
    app = build_agent_app(agent)
    uvicorn.run(app, host=args.host, port=args.port)


__all__ = ["build_agent_app", "configure_parser", "run"]

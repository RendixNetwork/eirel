from __future__ import annotations

import argparse
import json

from pydantic import BaseModel, Field, field_validator

from eirel.groups import FamilyId, ensure_family_id


class RegistrationRequest(BaseModel):
    hotkey: str
    endpoint: str
    family_id: FamilyId
    stake_bond_tao: float = Field(default=0.0, ge=0.0)
    cooldown_epochs: int = Field(default=3, ge=0)

    @field_validator("family_id", mode="before")
    @classmethod
    def validate_family_id(cls, value: str) -> str:
        return ensure_family_id(value)


def build_registration_request(*, hotkey: str, endpoint: str, family_id: str) -> RegistrationRequest:
    return RegistrationRequest(
        hotkey=hotkey,
        endpoint=endpoint.rstrip("/"),
        family_id=ensure_family_id(family_id),
    )


def configure_parser(sp: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = sp.add_parser("register", help="Emit a signed miner registration payload")
    parser.add_argument("--hotkey", required=True, help="Miner hotkey SS58 address")
    parser.add_argument("--endpoint", required=True, help="Miner endpoint URL")
    parser.add_argument("--family-id", dest="family_id", required=True, help="Execution family ID")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> None:
    payload = build_registration_request(hotkey=args.hotkey, endpoint=args.endpoint, family_id=args.family_id)
    print(json.dumps(payload.model_dump(), indent=2, sort_keys=True))


__all__ = ["RegistrationRequest", "build_registration_request", "configure_parser", "run"]

from __future__ import annotations

"""Bittensor wallet signing for authenticated API requests.

Miners authenticate with the owner API by signing requests with their
Bittensor hotkey.  This module wraps the keypair operations so the CLI
commands and user code never touch raw cryptography directly.
"""

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import uuid4


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_signing_string(method: str, path: str, body_hash: str, timestamp: str) -> str:
    """Canonical string that both signer and verifier hash.

    Shared between outbound :class:`Signer` and the inbound request verifier so
    the two sides can never drift apart.
    """
    return f"{method.upper()}{path}{body_hash}{timestamp}"


# Backwards-compatible private aliases (kept in case anything internal imports them).
_sha256_hex = sha256_hex
_signature_message = build_signing_string


@dataclass(slots=True)
class Signer:
    """Wraps a Bittensor keypair for request signing."""

    keypair: object  # bt.Keypair — typed as object to keep import lazy

    @property
    def hotkey(self) -> str:
        return self.keypair.ss58_address  # type: ignore[attr-defined]

    def sign(self, method: str, path: str, body_hash: str, timestamp: str) -> str:
        message = build_signing_string(method, path, body_hash, timestamp)
        return f"0x{self.keypair.sign(message).hex()}"  # type: ignore[attr-defined]

    def signed_headers(self, method: str, path: str, body: bytes) -> dict[str, str]:
        timestamp = datetime.now(UTC).isoformat()
        body_hash = sha256_hex(body)
        return {
            "X-Hotkey": self.hotkey,
            "X-Signature": self.sign(method, path, body_hash, timestamp),
            "X-Timestamp": timestamp,
            "X-Request-Id": str(uuid4()),
        }


def load_signer(
    *,
    wallet_name: str,
    hotkey_name: str,
    wallet_path: str | None = None,
) -> Signer:
    """Load a Bittensor keypair and return a :class:`Signer`.

    Requires ``pip install eirel[submit]`` (bittensor).
    """
    try:
        import bittensor as bt
    except ImportError as exc:
        raise ImportError(
            "bittensor is required for signing. Install with: pip install eirel[submit]"
        ) from exc

    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    return Signer(wallet.hotkey)

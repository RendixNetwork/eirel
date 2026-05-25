from __future__ import annotations

"""Unified miner CLI for the Eirel subnet.

Commands:
    eirel submit       — package and submit an agent to the owner API
    eirel status       — check current submission status and scores
    eirel package      — build a submission archive without uploading
    eirel compliance   — run pre-flight compliance checks against a running miner
    eirel register     — emit a signed miner registration payload
    eirel serve        — run a BaseAgent subclass as a FastAPI server
    eirel sample       — run the bundled reference miner sample server
"""

import argparse
import asyncio
import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

import httpx

from eirel import agent_server, compliance, packaging, registration, sample_server
from eirel.manifest import validate_submission_directory

# ── Constants ──────────────────────────────────────────────────────────────
TREASURY_ADDRESS = "5Fq7mueVryNAxcLcP7ARDtiyE6LhBBm9nFn7PN9CsV4KMumM"
SUBMISSION_FEE_TAO = 0.1


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _build_archive(source_dir: Path) -> bytes:
    validate_submission_directory(source_dir)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(source_dir))
    return buffer.getvalue()


def _load_signer(args: argparse.Namespace):
    from eirel.signing import load_signer

    return load_signer(
        wallet_name=args.wallet_name,
        hotkey_name=args.hotkey_name,
        wallet_path=getattr(args, "wallet_path", None),
    )


def _add_wallet_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("wallet authentication")
    group.add_argument("--wallet-name", required=True, help="Bittensor wallet name")
    group.add_argument("--hotkey-name", required=True, help="Bittensor hotkey name")
    group.add_argument("--wallet-path", help="Custom wallet directory path")


# ---------------------------------------------------------------------------
# Fee payment
# ---------------------------------------------------------------------------

def _pay_submission_fee(
    *,
    wallet_name: str,
    hotkey_name: str,
    wallet_path: str | None,
    network: str,
) -> tuple[str, str]:
    import bittensor as bt

    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name, path=wallet_path)
    subtensor = bt.Subtensor(network=network)
    balance = subtensor.get_balance(wallet.coldkeypub.ss58_address)
    fee_balance = bt.Balance.from_tao(SUBMISSION_FEE_TAO)

    print(f"\nSubmission fee: {SUBMISSION_FEE_TAO} TAO")
    print(f"Treasury:       {TREASURY_ADDRESS}")
    print(f"Network:        {network}")
    print(f"Wallet balance: {balance}")

    if balance < fee_balance:
        print(f"\nInsufficient balance. Need at least {fee_balance}, have {balance}.")
        sys.exit(1)

    confirm = input(f"\nSend {SUBMISSION_FEE_TAO} TAO to treasury? [y/N] ").strip().lower()
    if confirm != "y":
        print("Submission cancelled.")
        sys.exit(0)

    print("Transferring fee...")

    call = subtensor.substrate.compose_call(
        call_module="Balances",
        call_function="transfer_keep_alive",
        call_params={
            "dest": TREASURY_ADDRESS,
            "value": fee_balance.rao,
        },
    )
    extrinsic = subtensor.substrate.create_signed_extrinsic(
        call=call,
        keypair=wallet.coldkey,
    )
    response = subtensor.substrate.submit_extrinsic(
        extrinsic,
        wait_for_inclusion=True,
        wait_for_finalization=False,
    )

    if not response.is_success:
        print(f"Transfer failed: {response.error_message}")
        sys.exit(1)

    extrinsic_hash = response.extrinsic_hash
    if isinstance(extrinsic_hash, bytes):
        extrinsic_hash = f"0x{extrinsic_hash.hex()}"
    elif not extrinsic_hash.startswith("0x"):
        extrinsic_hash = f"0x{extrinsic_hash}"

    block_hash = response.block_hash
    if isinstance(block_hash, bytes):
        block_hash = f"0x{block_hash.hex()}"
    elif block_hash and not block_hash.startswith("0x"):
        block_hash = f"0x{block_hash}"

    print(f"Fee paid. Extrinsic hash: {extrinsic_hash}")
    print(f"Block hash: {block_hash}")
    return extrinsic_hash, block_hash


# ---------------------------------------------------------------------------
# submit
# ---------------------------------------------------------------------------

async def _submit(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    print(f"Packaging {source_dir} ...")
    archive = _build_archive(source_dir)
    print(f"Archive built: {len(archive):,} bytes")

    signer = _load_signer(args)
    print(f"Signing as {signer.hotkey}")

    extrinsic_hash: str | None = args.extrinsic_hash
    block_hash: str | None = args.block_hash
    if not extrinsic_hash and not args.skip_fee:
        extrinsic_hash, block_hash = _pay_submission_fee(
            wallet_name=args.wallet_name,
            hotkey_name=args.hotkey_name,
            wallet_path=getattr(args, "wallet_path", None),
            network=args.network,
        )

    query_parts: list[str] = []
    if extrinsic_hash:
        query_parts.append(f"extrinsic_hash={extrinsic_hash}")
    if block_hash:
        query_parts.append(f"block_hash={block_hash}")
    params = "&".join(query_parts)
    path = f"/v1/submissions?{params}" if params else "/v1/submissions"
    boundary = "eirel-sdk-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="archive"; filename="archive.tar.gz"\r\n'
        "Content-Type: application/gzip\r\n\r\n"
    ).encode() + archive + f"\r\n--{boundary}--\r\n".encode()

    headers = signer.signed_headers("POST", "/v1/submissions", body)
    headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"

    base_url = args.owner_api_url.rstrip("/")
    print(f"Submitting to {base_url} ...")

    async with httpx.AsyncClient(base_url=base_url, timeout=args.timeout) as client:
        response = await client.post(path, content=body, headers=headers)
        response.raise_for_status()
        result = response.json()

    print(json.dumps(result, indent=2))
    submission_id = result.get("id") or result.get("submission_id", "")
    if submission_id:
        print(f"\nSubmission ID: {submission_id}")
        print(f"Check status:  eirel status --owner-api-url {base_url} --wallet-name {args.wallet_name} --hotkey-name {args.hotkey_name}")


def _run_submit(args: argparse.Namespace) -> None:
    asyncio.run(_submit(args))


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

_BUILD_VERDICTS = {
    "built": "✓ built — agent compiled, booted, and passed the handshake",
    "build_failed": "✗ build failed",
    "verifying": "… verifying — build + boot in progress",
    "pending": "… pending — build not started yet",
}


def _print_submission_summary(submission: dict, *, index: int, is_current: bool) -> None:
    sub_id = submission.get("id") or submission.get("submission_id") or "?"
    seq = submission.get("submission_seq")
    family = submission.get("family_id", "?")
    marker = "  (current)" if is_current else ""
    header = f"[{index}] submission {sub_id}"
    if seq is not None:
        header += f"  seq={seq}"
    header += f"  family={family}{marker}"
    print(header)

    build_status = submission.get("build_status") or "pending"
    verdict = _BUILD_VERDICTS.get(build_status, build_status)
    checked = submission.get("build_checked_at")
    suffix = f"   (checked {checked})" if checked else ""
    print(f"    BUILD: {verdict}{suffix}")
    # build_error is set on failures and also when a verify deferred at
    # peak (build_status stays "pending" with a "deferred: ..." reason),
    # so show it whenever it's present.
    error = submission.get("build_error")
    if error:
        print(f"      reason: {error}")

    print(f"    submission status: {submission.get('status')}")
    summary = submission.get("latest_scorecard_summary")
    if isinstance(summary, dict) and summary.get("overall_score") is not None:
        print(f"    latest score: {summary.get('overall_score')}")


async def _status(args: argparse.Namespace) -> None:
    signer = _load_signer(args)
    base_url = args.owner_api_url.rstrip("/")

    async with httpx.AsyncClient(base_url=base_url, timeout=args.timeout) as client:
        headers = signer.signed_headers("GET", "/v1/submissions/mine", b"")
        response = await client.get("/v1/submissions/mine", headers=headers)
        response.raise_for_status()
        submissions = response.json()

        if not submissions:
            print("No submissions found for this hotkey.")
            return

        print(f"=== Your Submissions ({len(submissions)}) ===\n")
        for index, submission in enumerate(submissions, start=1):
            _print_submission_summary(submission, index=index, is_current=index == 1)
            print()

        # Newest submission first (server orders by submission_seq desc).
        submission = submissions[0]
        submission_id = submission.get("id") or submission.get("submission_id")

        if submission_id:
            try:
                progress_path = f"/v1/submissions/{submission_id}/progress"
                progress_headers = signer.signed_headers("GET", progress_path, b"")
                progress_response = await client.get(progress_path, headers=progress_headers)
            except httpx.HTTPError as exc:
                print(f"\nProgress: request failed ({exc})")
            else:
                if progress_response.status_code == 404:
                    print("\nProgress: not available yet")
                elif progress_response.status_code == 200:
                    print("\n=== Progress ===")
                    print(json.dumps(progress_response.json(), indent=2))
                else:
                    print(
                        f"\nProgress: {progress_response.status_code} "
                        f"{progress_response.reason_phrase}"
                    )

            try:
                headers = signer.signed_headers("GET", f"/v1/submissions/{submission_id}/scorecards", b"")
                score_response = await client.get(
                    f"/v1/submissions/{submission_id}/scorecards",
                    headers=headers,
                    params={"limit": args.score_limit},
                )
            except httpx.HTTPError as exc:
                print(f"\nScorecards: request failed ({exc})")
            else:
                if score_response.status_code == 404:
                    print("\nScorecards: not available yet")
                elif score_response.status_code == 200:
                    scorecards = score_response.json()
                    if scorecards:
                        print(f"\n=== Scorecards (last {args.score_limit}) ===")
                        print(json.dumps(scorecards, indent=2))
                    else:
                        print("\nScorecards: none yet")
                else:
                    print(
                        f"\nScorecards: {score_response.status_code} "
                        f"{score_response.reason_phrase}"
                    )


def _run_status(args: argparse.Namespace) -> None:
    asyncio.run(_status(args))


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="eirel",
        description="Eirel subnet miner CLI — build, submit, serve, and monitor agents.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    # --- submit ---
    submit_parser = subparsers.add_parser("submit", help="Package and submit an agent to the owner API")
    submit_parser.add_argument("--source-dir", required=True, help="Agent source directory with submission.yaml")
    submit_parser.add_argument("--owner-api-url", required=True, help="Owner API base URL")
    submit_parser.add_argument("--network", default="finney", help="Bittensor network: finney, test, local (default: finney)")
    submit_parser.add_argument("--extrinsic-hash", default=None, help="Use an existing extrinsic hash instead of paying a new fee")
    submit_parser.add_argument("--block-hash", default=None, help="Block hash containing the extrinsic (required with --extrinsic-hash)")
    submit_parser.add_argument("--skip-fee", action="store_true", help="Skip the on-chain fee payment (for testing)")
    submit_parser.add_argument("--timeout", type=float, default=180.0, help="Request timeout in seconds")
    _add_wallet_args(submit_parser)
    submit_parser.set_defaults(func=_run_submit)

    # --- status ---
    status_parser = subparsers.add_parser("status", help="Check submission status and scores")
    status_parser.add_argument("--owner-api-url", required=True, help="Owner API base URL")
    status_parser.add_argument("--score-limit", type=int, default=5, help="Number of recent scorecards to show")
    status_parser.add_argument("--timeout", type=float, default=30.0, help="Request timeout in seconds")
    _add_wallet_args(status_parser)
    status_parser.set_defaults(func=_run_status)

    # --- composed subcommands from other modules ---
    packaging.configure_parser(subparsers)
    compliance.configure_parser(subparsers)
    registration.configure_parser(subparsers)
    agent_server.configure_parser(subparsers)
    sample_server.configure_parser(subparsers)

    args = parser.parse_args()
    args.func(args)

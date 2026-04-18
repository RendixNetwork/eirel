from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

import yaml

from eirel.manifest import SubmissionManifest, validate_submission_directory


def write_submission_directory(path: Path, manifest: SubmissionManifest) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "submission.yaml").write_text(
        yaml.safe_dump(manifest.model_dump(mode="json"), sort_keys=False)
    )


_ARCHIVE_EXCLUDE_DIRS = {"__pycache__", ".git", ".venv", "venv", ".pytest_cache", ".mypy_cache", ".ruff_cache", "node_modules", ".DS_Store"}
_ARCHIVE_EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".swp", ".swo"}


def _should_skip_archive_path(rel_path: Path) -> bool:
    parts = rel_path.parts
    if any(part in _ARCHIVE_EXCLUDE_DIRS for part in parts):
        return True
    if rel_path.suffix in _ARCHIVE_EXCLUDE_SUFFIXES:
        return True
    return False


def build_submission_archive(source_dir: Path, output_path: Path | None = None) -> bytes:
    validate_submission_directory(source_dir)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(source_dir)
            if _should_skip_archive_path(rel):
                continue
            archive.add(path, arcname=rel)
    payload = buffer.getvalue()
    if output_path is not None:
        output_path.write_bytes(payload)
    return payload


def configure_parser(sp: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = sp.add_parser("package", help="Build a submission archive without uploading")
    parser.add_argument("--source-dir", required=True, help="Agent source directory with submission.yaml")
    parser.add_argument("--output", default="submission.tar.gz", help="Output archive path")
    parser.set_defaults(func=run)
    return parser


def run(args: argparse.Namespace) -> None:
    payload = build_submission_archive(Path(args.source_dir), Path(args.output))
    print(f"built {len(payload)} bytes to {args.output}")

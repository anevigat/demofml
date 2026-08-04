"""Canonical, strict, no-replace records for Campaign 2 engineering."""

import hashlib
import json
import os
import re
import uuid
from pathlib import Path
from typing import Any

SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CONTENT_ID_PATTERN = re.compile(r"sha256-[0-9a-f]{64}")
IMAGE_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")


def canonical_json(value: object) -> bytes:
    """Serialize one hashable record with the frozen Campaign 2 encoding."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("ascii")


def content_id(value: object) -> str:
    """Return a prefixed SHA-256 identity for canonical record content."""
    return f"sha256-{hashlib.sha256(canonical_json(value)).hexdigest()}"


def sha256_file(path: Path) -> str:
    """Hash file bytes without interpreting their content."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field: {key}")
        value[key] = item
    return value


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_strict_json(path: Path, name: str) -> dict[str, Any]:
    """Read one object while rejecting duplicates, constants, and non-objects."""
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    try:
        value = json.loads(
            path.read_text(encoding="ascii"),
            object_pairs_hook=_strict_pairs,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"{name} is invalid: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def write_immutable_json(path: Path, value: object) -> None:
    """Durably create one canonical JSON record without replacement."""
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("immutable record parent must be an existing directory")
    if path.is_symlink():
        raise ValueError(f"refusing symlink destination: {path}")
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("xb") as target:
            target.write(canonical_json(value))
            target.flush()
            os.fsync(target.fileno())
        os.link(partial, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise RuntimeError(f"immutable record already exists: {path}") from error
    finally:
        partial.unlink(missing_ok=True)

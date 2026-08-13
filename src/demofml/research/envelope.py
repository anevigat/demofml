"""Sealed-envelope pre-registration for a research variant's four documents.

Campaign 2 failed on ordering, not on arithmetic: design decisions were taken
after historical results had already been observed, so no later run could serve
as independent confirmation of those decisions. A sealed envelope makes the
ordering auditable instead of merely asserted. Before a variant's first fold is
run, its hypothesis document, validation plan, model contract (including the
full hyperparameter search space) and acceptance contract are committed, and
their SHA-256 digests are recorded in one small TOML file that is committed in
the same change. Every later run verifies those digests, so silently relaxing a
threshold, widening a search space or rewriting a hypothesis after seeing
results becomes a detectable event rather than an invisible one.

The envelope deliberately seals exactly four roles. The append-only data-use
ledger is not sealed — it is meant to grow — and nothing else may be added,
because an envelope that can absorb arbitrary documents stops being evidence
about what was decided in advance.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

ENVELOPE_SCHEMA_ID = "sealed-envelope-v1"
REQUIRED_ROLES: tuple[str, ...] = (
    "hypothesis",
    "validation",
    "model",
    "acceptance",
)
_TOP_LEVEL_KEYS = frozenset(
    {"format_version", "schema", "id", "campaign", "sealed_at", "root", "documents"}
)
_DOCUMENT_KEYS = frozenset({"path", "sha256"})
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_HASH_BLOCK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class SealedDocument:
    """One pre-registered document and the digest it was sealed with."""

    role: str
    path: PurePosixPath
    sha256: str


@dataclass(frozen=True)
class SealedEnvelope:
    """The immutable set of documents frozen before a variant's first fold."""

    id: str
    campaign: str
    sealed_at: datetime
    root: Path
    documents: tuple[SealedDocument, ...]

    def document(self, role: str) -> SealedDocument:
        """Return the sealed document registered under one role."""
        for document in self.documents:
            if document.role == role:
                return document
        raise KeyError(f"sealed envelope has no {role} document")

    def resolved_path(self, role: str) -> Path:
        """Return the absolute path of one sealed document."""
        return self.root / self.document(role).path


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file's bytes."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _relative_document_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"sealed document path must stay inside the root: {value}")
    return path


def _sealed_at(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError("sealed_at must be a TOML datetime")
    offset = value.utcoffset()
    if offset is None or offset.total_seconds() != 0.0:
        raise ValueError("sealed_at must be an explicit UTC timestamp")
    return value


def load_sealed_envelope(path: Path) -> SealedEnvelope:
    """Load and strictly validate one sealed-envelope declaration."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Sealed envelope is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    unknown = set(values).difference(_TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"unknown sealed envelope fields: {sorted(unknown)}")
    try:
        if int(values["format_version"]) != 1:
            raise ValueError("sealed envelope format_version must be 1")
        if str(values["schema"]) != ENVELOPE_SCHEMA_ID:
            raise ValueError("sealed envelope schema is not supported")
        root = (path.parent / str(values["root"])).resolve()
        documents = values["documents"]
        if not isinstance(documents, dict):
            raise ValueError("sealed envelope documents must be a table")
        if tuple(sorted(documents)) != tuple(sorted(REQUIRED_ROLES)):
            raise ValueError(
                f"sealed envelope must declare exactly {sorted(REQUIRED_ROLES)}"
            )
        sealed: list[SealedDocument] = []
        for role in REQUIRED_ROLES:
            entry = documents[role]
            if not isinstance(entry, dict):
                raise ValueError(f"sealed document {role} must be a table")
            unknown_keys = set(entry).difference(_DOCUMENT_KEYS)
            if unknown_keys:
                raise ValueError(
                    f"unknown sealed document fields: {sorted(unknown_keys)}"
                )
            digest = str(entry["sha256"])
            if _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"sealed document {role} has an invalid digest")
            sealed.append(
                SealedDocument(
                    role=role,
                    path=_relative_document_path(str(entry["path"])),
                    sha256=digest,
                )
            )
        envelope = SealedEnvelope(
            id=str(values["id"]),
            campaign=str(values["campaign"]),
            sealed_at=_sealed_at(values["sealed_at"]),
            root=root,
            documents=tuple(sealed),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid sealed envelope field: {error}") from error
    if not envelope.root.is_dir():
        raise RuntimeError(f"Sealed envelope root is not a directory: {envelope.root}")
    return envelope


def verify_sealed_envelope(envelope: SealedEnvelope) -> dict[str, str]:
    """Return each role's current digest, raising if any seal is broken.

    Every sealed document is checked, and every failure is reported together:
    knowing that both the model contract and the acceptance thresholds moved is
    a different situation from knowing only that one of them did.
    """
    observed: dict[str, str] = {}
    broken: list[str] = []
    for document in envelope.documents:
        target = envelope.root / document.path
        if not target.is_file():
            broken.append(f"{document.role}: missing {document.path}")
            continue
        digest = file_sha256(target)
        observed[document.role] = digest
        if digest != document.sha256:
            broken.append(
                f"{document.role}: {document.path} is {digest}, "
                f"sealed as {document.sha256}"
            )
    if broken:
        raise RuntimeError(
            f"Sealed envelope {envelope.id} is broken: " + "; ".join(sorted(broken))
        )
    return observed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify that a research variant's sealed documents are intact."
    )
    parser.add_argument("--envelope", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the sealed-envelope verification command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        envelope = load_sealed_envelope(arguments.envelope)
        digests = verify_sealed_envelope(envelope)
        print(
            f"sealed envelope {envelope.id} verified: "
            f"{len(digests)} documents unchanged since "
            f"{envelope.sealed_at.isoformat()}"
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

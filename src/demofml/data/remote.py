"""Verified reads of immutable development data from S3."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tomllib
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

import boto3  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]

from demofml.data.ticks import validate_tick_schema

_HASH_BLOCK_SIZE = 8 * 1024 * 1024
_MAXIMUM_MANIFEST_SIZE = 16 * 1024 * 1024
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_VERSION_PATTERN = re.compile(r"^sha256-[0-9a-f]{64}$")
LOCKED_DATASET_SET_ID = "cleaned-ticks-locked-test-v1"


@dataclass(frozen=True)
class DevelopmentFile:
    """One explicitly authorized object in the development dataset."""

    symbol: str
    path: str
    sha256: str
    size_bytes: int
    rows: int


@dataclass(frozen=True)
class DevelopmentDataset:
    """Immutable allowlist for data that predates the locked test."""

    format_version: int
    id: str
    dataset_name: str
    dataset_version: str
    s3_prefix: str
    start: datetime
    end_exclusive: datetime
    files: tuple[DevelopmentFile, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return symbols in deterministic order."""
        return tuple(sorted({entry.symbol for entry in self.files}))

    def files_for_symbol(self, symbol: str) -> tuple[DevelopmentFile, ...]:
        """Return the ordered allowlist for one symbol."""
        return tuple(entry for entry in self.files if entry.symbol == symbol)


@dataclass(frozen=True)
class LockedTestDataset:
    """Immutable allowlist for the locked test dataset."""

    format_version: int
    id: str
    dataset_name: str
    dataset_version: str
    s3_prefix: str
    start: datetime
    end_exclusive: datetime
    files: tuple[DevelopmentFile, ...]

    @property
    def symbols(self) -> tuple[str, ...]:
        """Return symbols in deterministic order."""
        return tuple(sorted({entry.symbol for entry in self.files}))

    def files_for_symbol(self, symbol: str) -> tuple[DevelopmentFile, ...]:
        """Return the ordered allowlist for one symbol."""
        return tuple(entry for entry in self.files if entry.symbol == symbol)


def _parse_utc(value: object, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO-8601 UTC string") from error
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise ValueError(f"{field} must use UTC")
    return parsed


def _safe_relative_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} must be a safe relative path")
    return path.as_posix()


def load_development_dataset(path: Path) -> DevelopmentDataset:
    """Load and strictly validate a development-only object allowlist."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Development dataset config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    try:
        files = tuple(
            DevelopmentFile(
                symbol=str(entry["symbol"]),
                path=_safe_relative_path(entry["path"], "files.path"),
                sha256=str(entry["sha256"]),
                size_bytes=int(entry["size_bytes"]),
                rows=int(entry["rows"]),
            )
            for entry in values["files"]
        )
        dataset = DevelopmentDataset(
            format_version=int(values["format_version"]),
            id=str(values["id"]),
            dataset_name=str(values["dataset_name"]),
            dataset_version=str(values["dataset_version"]),
            s3_prefix=_safe_relative_path(values["s3_prefix"], "s3_prefix"),
            start=_parse_utc(values["start"], "start"),
            end_exclusive=_parse_utc(values["end_exclusive"], "end_exclusive"),
            files=files,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid development dataset field: {error}") from error

    if dataset.format_version != 1:
        raise ValueError("development dataset format_version must be 1")
    if not dataset.id or not dataset.dataset_name:
        raise ValueError("development dataset identifiers cannot be empty")
    if not _VERSION_PATTERN.fullmatch(dataset.dataset_version):
        raise ValueError("development dataset version must be content-addressed")
    if not dataset.start < dataset.end_exclusive:
        raise ValueError("development dataset interval must be non-empty")
    if not dataset.files:
        raise ValueError("development dataset must authorize at least one file")
    paths: set[str] = set()
    previous_by_symbol: dict[str, str] = {}
    for entry in dataset.files:
        if not entry.symbol or PurePosixPath(entry.path).parts[0] != entry.symbol:
            raise ValueError("development file path must start with its symbol")
        if not entry.path.endswith(".parquet"):
            raise ValueError("development files must be Parquet objects")
        if entry.path in paths:
            raise ValueError(f"duplicate development file path: {entry.path}")
        if not _SHA256_PATTERN.fullmatch(entry.sha256):
            raise ValueError(f"invalid development file SHA-256: {entry.path}")
        if entry.size_bytes <= 0 or entry.rows <= 0:
            raise ValueError("development file sizes and row counts must be positive")
        previous = previous_by_symbol.get(entry.symbol)
        if previous is not None and entry.path <= previous:
            raise ValueError(f"development files are not ordered for {entry.symbol}")
        paths.add(entry.path)
        previous_by_symbol[entry.symbol] = entry.path
    return dataset


def load_locked_test_dataset(path: Path) -> LockedTestDataset:
    """Load and strictly validate the locked test object allowlist."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Locked test dataset config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)

    expected_fields = {
        "format_version",
        "id",
        "dataset_name",
        "dataset_version",
        "s3_prefix",
        "start",
        "end_exclusive",
        "files",
    }
    if set(values) != expected_fields:
        raise ValueError("locked test dataset fields are invalid")
    raw_files = values["files"]
    if not isinstance(raw_files, list):
        raise ValueError("locked test dataset files must be an array")
    file_fields = {"symbol", "path", "sha256", "size_bytes", "rows"}
    if any(
        not isinstance(entry, dict) or set(entry) != file_fields for entry in raw_files
    ):
        raise ValueError("locked test dataset file fields are invalid")

    try:
        if (
            not isinstance(values["format_version"], int)
            or isinstance(values["format_version"], bool)
            or not isinstance(values["id"], str)
            or not isinstance(values["dataset_name"], str)
            or not isinstance(values["dataset_version"], str)
        ):
            raise ValueError("field types are invalid")
        files = tuple(
            DevelopmentFile(
                symbol=entry["symbol"],
                path=_safe_relative_path(entry["path"], "files.path"),
                sha256=entry["sha256"],
                size_bytes=entry["size_bytes"],
                rows=entry["rows"],
            )
            for entry in raw_files
        )
        dataset = LockedTestDataset(
            format_version=values["format_version"],
            id=values["id"],
            dataset_name=values["dataset_name"],
            dataset_version=values["dataset_version"],
            s3_prefix=_safe_relative_path(values["s3_prefix"], "s3_prefix"),
            start=_parse_utc(values["start"], "start"),
            end_exclusive=_parse_utc(values["end_exclusive"], "end_exclusive"),
            files=files,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid locked test dataset field: {error}") from error

    if dataset.format_version != 1:
        raise ValueError("locked test dataset format_version must be 1")
    if dataset.id != LOCKED_DATASET_SET_ID:
        raise ValueError(f"locked test dataset id must be {LOCKED_DATASET_SET_ID}")
    if not dataset.dataset_name:
        raise ValueError("locked test dataset name cannot be empty")
    if not _VERSION_PATTERN.fullmatch(dataset.dataset_version):
        raise ValueError("locked test dataset version must be content-addressed")
    if not dataset.start < dataset.end_exclusive:
        raise ValueError("locked test dataset interval must be non-empty")
    if not dataset.files:
        raise ValueError("locked test dataset must authorize at least one file")

    previous_path: str | None = None
    for entry in dataset.files:
        if not isinstance(entry.symbol, str) or not entry.symbol:
            raise ValueError("locked test file symbol cannot be empty")
        if PurePosixPath(entry.path).parts[0] != entry.symbol:
            raise ValueError("locked test file path must start with its symbol")
        if not entry.path.endswith(".parquet"):
            raise ValueError("locked test files must be Parquet objects")
        if previous_path is not None and entry.path <= previous_path:
            raise ValueError("locked test files must have unique ordered paths")
        if not isinstance(entry.sha256, str) or not _SHA256_PATTERN.fullmatch(
            entry.sha256
        ):
            raise ValueError(f"invalid locked test file SHA-256: {entry.path}")
        if (
            not isinstance(entry.size_bytes, int)
            or isinstance(entry.size_bytes, bool)
            or not isinstance(entry.rows, int)
            or isinstance(entry.rows, bool)
            or entry.size_bytes <= 0
            or entry.rows <= 0
        ):
            raise ValueError("locked test file sizes and row counts must be positive")
        previous_path = entry.path
    return dataset


def s3_client(endpoint_url: str, region_name: str) -> Any:
    """Create the path-style S3 client used by MinIO-backed research jobs."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region_name,
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 10, "mode": "standard"},
            s3={"addressing_style": "path"},
        ),
    )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def load_published_manifest(
    client: Any, bucket: str, dataset: DevelopmentDataset | LockedTestDataset
) -> dict[str, Any]:
    """Fetch and verify the completed publication manifest for an allowlist."""
    key = f"{dataset.s3_prefix}/{dataset.dataset_version}/manifest.json"
    head = client.head_object(Bucket=bucket, Key=key)
    size = int(head["ContentLength"])
    if size <= 0 or size > _MAXIMUM_MANIFEST_SIZE:
        raise RuntimeError("Published dataset manifest size is invalid")
    request = {"Bucket": bucket, "Key": key}
    if head.get("ETag"):
        request["IfMatch"] = str(head["ETag"])
    body = client.get_object(**request)["Body"]
    payload = body.read(size + 1)
    if len(payload) != size:
        raise RuntimeError("Published dataset manifest changed during download")
    metadata = {
        str(name).lower(): str(value)
        for name, value in head.get("Metadata", {}).items()
    }
    expected_payload_hash = metadata.get("sha256")
    if (
        expected_payload_hash is not None
        and hashlib.sha256(payload).hexdigest() != expected_payload_hash
    ):
        raise RuntimeError("Published dataset manifest failed SHA-256")
    try:
        manifest = json.loads(payload)
    except (TypeError, json.JSONDecodeError) as error:
        raise RuntimeError("Published dataset manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise RuntimeError("Published dataset manifest must be an object")
    try:
        version = str(manifest["dataset_version"])
        files = manifest["files"]
        core = {
            "format_version": int(manifest["format_version"]),
            "dataset_name": str(manifest["dataset_name"]),
            "file_count": int(manifest["file_count"]),
            "total_size_bytes": int(manifest["total_size_bytes"]),
            "total_rows": int(manifest["total_rows"]),
            "files": files,
        }
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"Published dataset manifest is invalid: {error}") from error
    expected_version = f"sha256-{hashlib.sha256(_canonical_json(core)).hexdigest()}"
    if version != expected_version or version != dataset.dataset_version:
        raise RuntimeError(
            "Published dataset manifest version does not match its content"
        )
    if core["format_version"] != 1 or core["dataset_name"] != dataset.dataset_name:
        raise RuntimeError("Published dataset identity is incompatible")
    if not isinstance(files, list) or core["file_count"] != len(files):
        raise RuntimeError("Published dataset file count is invalid")
    try:
        if core["total_size_bytes"] != sum(int(entry["size_bytes"]) for entry in files):
            raise RuntimeError("Published dataset byte total is invalid")
        if core["total_rows"] != sum(int(entry["rows"]) for entry in files):
            raise RuntimeError("Published dataset row total is invalid")
        by_path = {str(entry["path"]): entry for entry in files}
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Published dataset file entry is invalid: {error}"
        ) from error
    if len(by_path) != len(files):
        raise RuntimeError("Published dataset paths must be unique")
    for selected in dataset.files:
        remote = by_path.get(selected.path)
        if remote is None:
            raise RuntimeError(
                f"Authorized object is absent from manifest: {selected.path}"
            )
        if (
            str(remote.get("sha256")) != selected.sha256
            or int(remote.get("size_bytes", -1)) != selected.size_bytes
            or int(remote.get("rows", -1)) != selected.rows
        ):
            raise RuntimeError(
                f"Authorized object differs from manifest: {selected.path}"
            )
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def validate_development_parquet(
    path: Path,
    entry: DevelopmentFile,
    start: datetime,
    end_exclusive: datetime,
) -> None:
    """Scan actual timestamps and prove that an authorized file predates the lock."""
    parquet = pq.ParquetFile(path)
    validate_tick_schema(parquet.schema_arrow)
    if parquet.metadata.num_rows != entry.rows:
        raise RuntimeError(f"Parquet row count differs from allowlist: {entry.path}")
    timestamp_index = parquet.schema_arrow.get_field_index("timestamp")
    previous_max: datetime | None = None
    for index in range(parquet.metadata.num_row_groups):
        statistics = (
            parquet.metadata.row_group(index).column(timestamp_index).statistics
        )
        if statistics is None or not statistics.has_min_max:
            raise RuntimeError(f"Timestamp statistics are required: {entry.path}")
        minimum = statistics.min
        maximum = statistics.max
        if not isinstance(minimum, datetime) or not isinstance(maximum, datetime):
            raise RuntimeError(f"Timestamp statistics are invalid: {entry.path}")
        if minimum < start or maximum >= end_exclusive:
            raise RuntimeError(
                f"Timestamp statistics are outside development: {entry.path}"
            )
        if minimum > maximum or (previous_max is not None and minimum < previous_max):
            raise RuntimeError(
                f"Parquet row groups are not temporally ordered: {entry.path}"
            )
        previous_max = maximum

    previous_value: int | None = None
    scanned_rows = 0
    for batch in parquet.iter_batches(
        batch_size=1_000_000, columns=["timestamp"], use_threads=False
    ):
        timestamps = batch.column(0)
        if timestamps.null_count:
            raise RuntimeError(f"Timestamp values cannot be null: {entry.path}")
        minimum = pc.min(timestamps).as_py()
        maximum = pc.max(timestamps).as_py()
        if not isinstance(minimum, datetime) or not isinstance(maximum, datetime):
            raise RuntimeError(f"Timestamp values are invalid: {entry.path}")
        if minimum < start or maximum >= end_exclusive:
            raise RuntimeError(f"Timestamp is outside development: {entry.path}")
        values = pc.cast(timestamps, pa.int64())
        first = values[0].as_py()
        last = values[-1].as_py()
        if not isinstance(first, int) or not isinstance(last, int):
            raise RuntimeError(f"Timestamp values are invalid: {entry.path}")
        if previous_value is not None and first < previous_value:
            raise RuntimeError(f"Timestamp values are not ordered: {entry.path}")
        if len(values) > 1:
            out_of_order = pc.any(
                pc.less(values.slice(1), values.slice(0, len(values) - 1))
            ).as_py()
            if bool(out_of_order):
                raise RuntimeError(f"Timestamp values are not ordered: {entry.path}")
        previous_value = last
        scanned_rows += len(values)
    if scanned_rows != entry.rows:
        raise RuntimeError(f"Scanned row count differs from allowlist: {entry.path}")


def validate_locked_test_parquet(
    path: Path,
    entry: DevelopmentFile,
    start: datetime,
    end_exclusive: datetime,
) -> None:
    """Scan footer and actual timestamps for the locked interval."""
    parquet = pq.ParquetFile(path)
    validate_tick_schema(parquet.schema_arrow)
    if parquet.metadata.num_rows != entry.rows:
        raise RuntimeError(f"Parquet row count differs from allowlist: {entry.path}")
    timestamp_index = parquet.schema_arrow.get_field_index("timestamp")
    previous_max: datetime | None = None
    for index in range(parquet.metadata.num_row_groups):
        statistics = (
            parquet.metadata.row_group(index).column(timestamp_index).statistics
        )
        if statistics is None or not statistics.has_min_max:
            raise RuntimeError(f"Timestamp statistics are required: {entry.path}")
        minimum = statistics.min
        maximum = statistics.max
        if not isinstance(minimum, datetime) or not isinstance(maximum, datetime):
            raise RuntimeError(f"Timestamp statistics are invalid: {entry.path}")
        if minimum < start or maximum >= end_exclusive:
            raise RuntimeError(
                f"Timestamp statistics are outside locked test: {entry.path}"
            )
        if minimum > maximum or (previous_max is not None and minimum < previous_max):
            raise RuntimeError(
                f"Parquet row groups are not temporally ordered: {entry.path}"
            )
        previous_max = maximum

    previous_value: int | None = None
    scanned_rows = 0
    for batch in parquet.iter_batches(
        batch_size=1_000_000, columns=["timestamp"], use_threads=False
    ):
        timestamps = batch.column(0)
        if timestamps.null_count:
            raise RuntimeError(f"Timestamp values cannot be null: {entry.path}")
        minimum = pc.min(timestamps).as_py()
        maximum = pc.max(timestamps).as_py()
        if not isinstance(minimum, datetime) or not isinstance(maximum, datetime):
            raise RuntimeError(f"Timestamp values are invalid: {entry.path}")
        if minimum < start or maximum >= end_exclusive:
            raise RuntimeError(f"Timestamp is outside locked test: {entry.path}")
        values = pc.cast(timestamps, pa.int64())
        first = values[0].as_py()
        last = values[-1].as_py()
        if not isinstance(first, int) or not isinstance(last, int):
            raise RuntimeError(f"Timestamp values are invalid: {entry.path}")
        if previous_value is not None and first < previous_value:
            raise RuntimeError(f"Timestamp values are not ordered: {entry.path}")
        if len(values) > 1:
            out_of_order = pc.any(
                pc.less(values.slice(1), values.slice(0, len(values) - 1))
            ).as_py()
            if bool(out_of_order):
                raise RuntimeError(f"Timestamp values are not ordered: {entry.path}")
        previous_value = last
        scanned_rows += len(values)
    if scanned_rows != entry.rows:
        raise RuntimeError(f"Scanned row count differs from allowlist: {entry.path}")


def _safe_output_path(
    destination: Path, relative: str, cache_label: str = "Development"
) -> Path:
    current = destination
    if current.is_symlink() or (current.exists() and not current.is_dir()):
        raise RuntimeError(f"{cache_label} cache root is unsafe: {destination}")
    current.mkdir(parents=True, exist_ok=True)
    parts = PurePosixPath(relative).parts
    for part in parts[:-1]:
        current = current / part
        if current.is_symlink() or (current.exists() and not current.is_dir()):
            raise RuntimeError(f"{cache_label} cache path is unsafe: {current}")
        current.mkdir(exist_ok=True)
    return current / parts[-1]


def materialize_development_file(
    client: Any,
    bucket: str,
    dataset: DevelopmentDataset,
    entry: DevelopmentFile,
    destination: Path,
) -> Path:
    """Download, hash, time-check, and atomically publish one authorized object."""
    expanded_destination = destination.expanduser()
    if expanded_destination.is_symlink():
        raise RuntimeError(f"Development cache root is unsafe: {destination}")
    destination = expanded_destination.resolve()
    output = _safe_output_path(destination, entry.path)
    if output.is_symlink():
        raise RuntimeError(f"Development object path is unsafe: {output}")
    if output.is_file():
        if (
            output.stat().st_size != entry.size_bytes
            or _file_sha256(output) != entry.sha256
        ):
            raise RuntimeError(f"Cached development object is not immutable: {output}")
        validate_development_parquet(
            output, entry, dataset.start, dataset.end_exclusive
        )
        return output
    if output.exists():
        raise RuntimeError(f"Development object path is not a file: {output}")

    key = f"{dataset.s3_prefix}/{dataset.dataset_version}/data/{entry.path}"
    head = client.head_object(Bucket=bucket, Key=key)
    metadata = {
        str(name).lower(): str(value)
        for name, value in head.get("Metadata", {}).items()
    }
    if (
        int(head["ContentLength"]) != entry.size_bytes
        or metadata.get("sha256") != entry.sha256
        or metadata.get("dataset-version") != dataset.dataset_version
    ):
        raise RuntimeError(f"Remote development object metadata differs: {entry.path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    digest = hashlib.sha256()
    size = 0
    try:
        request = {"Bucket": bucket, "Key": key}
        if head.get("ETag"):
            request["IfMatch"] = str(head["ETag"])
        body = client.get_object(**request)["Body"]
        with partial.open("xb") as target:
            while block := body.read(_HASH_BLOCK_SIZE):
                target.write(block)
                digest.update(block)
                size += len(block)
                if size > entry.size_bytes:
                    raise RuntimeError(
                        f"Remote development object exceeds its size: {entry.path}"
                    )
            target.flush()
            os.fsync(target.fileno())
        if size != entry.size_bytes or digest.hexdigest() != entry.sha256:
            raise RuntimeError(
                f"Downloaded development object failed SHA-256: {entry.path}"
            )
        validate_development_parquet(
            partial, entry, dataset.start, dataset.end_exclusive
        )
        try:
            os.link(partial, output)
        except FileExistsError as error:
            if (
                output.stat().st_size != entry.size_bytes
                or _file_sha256(output) != entry.sha256
            ):
                raise RuntimeError(
                    f"Development object appeared with different data: {output}"
                ) from error
    finally:
        partial.unlink(missing_ok=True)
    return output


def materialize_locked_test_file(
    client: Any,
    bucket: str,
    dataset: LockedTestDataset,
    entry: DevelopmentFile,
    destination: Path,
) -> Path:
    """Download, verify, and atomically publish one locked test object."""
    expanded_destination = destination.expanduser()
    if expanded_destination.is_symlink():
        raise RuntimeError(f"Locked test cache root is unsafe: {destination}")
    destination = expanded_destination.resolve()
    output = _safe_output_path(destination, entry.path, "Locked test")
    if output.is_symlink():
        raise RuntimeError(f"Locked test object path is unsafe: {output}")
    if output.is_file():
        if (
            output.stat().st_size != entry.size_bytes
            or _file_sha256(output) != entry.sha256
        ):
            raise RuntimeError(f"Cached locked test object is not immutable: {output}")
        validate_locked_test_parquet(
            output, entry, dataset.start, dataset.end_exclusive
        )
        return output
    if output.exists():
        raise RuntimeError(f"Locked test object path is not a file: {output}")

    key = f"{dataset.s3_prefix}/{dataset.dataset_version}/data/{entry.path}"
    head = client.head_object(Bucket=bucket, Key=key)
    metadata = {
        str(name).lower(): str(value)
        for name, value in head.get("Metadata", {}).items()
    }
    if (
        int(head["ContentLength"]) != entry.size_bytes
        or metadata.get("sha256") != entry.sha256
        or metadata.get("dataset-version") != dataset.dataset_version
    ):
        raise RuntimeError(f"Remote locked test object metadata differs: {entry.path}")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    digest = hashlib.sha256()
    size = 0
    try:
        request = {"Bucket": bucket, "Key": key}
        if head.get("ETag"):
            request["IfMatch"] = str(head["ETag"])
        body = client.get_object(**request)["Body"]
        with partial.open("xb") as target:
            while block := body.read(_HASH_BLOCK_SIZE):
                target.write(block)
                digest.update(block)
                size += len(block)
                if size > entry.size_bytes:
                    raise RuntimeError(
                        f"Remote locked test object exceeds its size: {entry.path}"
                    )
            target.flush()
            os.fsync(target.fileno())
        if size != entry.size_bytes or digest.hexdigest() != entry.sha256:
            raise RuntimeError(
                f"Downloaded locked test object failed SHA-256: {entry.path}"
            )
        validate_locked_test_parquet(
            partial, entry, dataset.start, dataset.end_exclusive
        )
        try:
            os.link(partial, output)
        except FileExistsError as error:
            if (
                output.stat().st_size != entry.size_bytes
                or _file_sha256(output) != entry.sha256
            ):
                raise RuntimeError(
                    f"Locked test object appeared with different data: {output}"
                ) from error
    finally:
        partial.unlink(missing_ok=True)
    return output


def verify_materialized_inventory(
    destination: Path,
    entries: Iterable[DevelopmentFile],
    *,
    path_prefix: str | None = None,
) -> tuple[Path, ...]:
    """Verify an exact, symlink-free Parquet inventory in entry order."""
    expanded_destination = destination.expanduser()
    if expanded_destination.is_symlink():
        raise RuntimeError(f"Materialized inventory root is unsafe: {destination}")
    destination = expanded_destination.resolve()
    if not destination.is_dir():
        raise RuntimeError(
            f"Materialized inventory root is not a directory: {destination}"
        )

    ordered_entries = tuple(entries)
    ordered_files: list[Path] = []
    expected: set[Path] = set()
    for entry in ordered_entries:
        relative = _safe_relative_path(entry.path, "entries.path")
        parts = PurePosixPath(relative).parts
        if path_prefix is not None:
            if not parts or parts[0] != path_prefix:
                raise RuntimeError(
                    "Materialized inventory path is outside "
                    f"{path_prefix}: {entry.path}"
                )
            parts = parts[1:]
        output = destination.joinpath(*parts)
        if output in expected:
            raise RuntimeError(f"Duplicate materialized inventory path: {entry.path}")
        expected.add(output)
        ordered_files.append(output)

    actual: set[Path] = set()
    for root, directories, files in os.walk(destination, followlinks=False):
        root_path = Path(root)
        for name in directories:
            path = root_path / name
            if path.is_symlink():
                raise RuntimeError(f"Materialized inventory contains a symlink: {path}")
        for name in files:
            path = root_path / name
            if path.is_symlink():
                raise RuntimeError(f"Materialized inventory contains a symlink: {path}")
            if path.suffix == ".parquet":
                actual.add(path)

    missing = expected - actual
    if missing:
        raise RuntimeError(f"Materialized inventory is missing: {min(missing)}")
    extra = actual - expected
    if extra:
        raise RuntimeError(
            f"Materialized inventory has extra Parquet file: {min(extra)}"
        )
    return tuple(ordered_files)

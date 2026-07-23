"""Content-addressed publication of curated Parquet datasets to S3."""

import argparse
import hashlib
import json
import os
import re
from collections.abc import Sequence
from pathlib import Path, PurePosixPath
from typing import Any, TypedDict

import boto3  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from botocore.config import Config  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from demofml.data.progress import ProgressBar

_HASH_BLOCK_SIZE = 8 * 1024 * 1024
_MINIMUM_PART_SIZE = 5 * 1024 * 1024
_MAXIMUM_PARTS = 10_000
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


class SchemaField(TypedDict):
    """Stable representation of one Arrow field."""

    name: str
    type: str
    nullable: bool


class FileEntry(TypedDict):
    """Manifest details for one Parquet object."""

    path: str
    size_bytes: int
    sha256: str
    rows: int
    row_groups: int
    schema: list[SchemaField]


class DatasetManifest(TypedDict):
    """Canonical, content-addressed dataset manifest."""

    format_version: int
    dataset_name: str
    dataset_version: str
    file_count: int
    total_size_bytes: int
    total_rows: int
    files: list[FileEntry]


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    completed = 0
    progress = ProgressBar("hash", size)
    with path.open("rb") as source:
        while block := source.read(_HASH_BLOCK_SIZE):
            digest.update(block)
            completed += len(block)
            progress.update(completed)
    return digest.hexdigest()


def _inspect_file(path: Path, source: Path) -> FileEntry:
    before = path.stat()
    digest = _file_sha256(path)
    metadata = pq.read_metadata(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"File changed while building manifest: {path}")

    arrow_schema = metadata.schema.to_arrow_schema()
    schema: list[SchemaField] = [
        {
            "name": field.name,
            "type": str(field.type),
            "nullable": field.nullable,
        }
        for field in arrow_schema
    ]
    return {
        "path": path.relative_to(source).as_posix(),
        "size_bytes": before.st_size,
        "sha256": digest,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "schema": schema,
    }


def build_manifest(source: Path, dataset_name: str) -> DatasetManifest:
    """Hash and inspect every Parquet file below ``source``."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"Dataset source is not a directory: {source}")
    if not _NAME_PATTERN.fullmatch(dataset_name):
        raise RuntimeError(
            "Dataset name must contain only lowercase letters, numbers, _ or -"
        )

    paths = sorted(
        (path for path in source.rglob("*.parquet") if path.is_file()),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    if not paths:
        raise RuntimeError(f"No Parquet files found below: {source}")

    files: list[FileEntry] = []
    for index, path in enumerate(paths, start=1):
        relative_path = path.relative_to(source).as_posix()
        print(f"[{index}/{len(paths)}] hashing {relative_path}", flush=True)
        files.append(_inspect_file(path, source))

    core: dict[str, object] = {
        "format_version": 1,
        "dataset_name": dataset_name,
        "file_count": len(files),
        "total_size_bytes": sum(entry["size_bytes"] for entry in files),
        "total_rows": sum(entry["rows"] for entry in files),
        "files": files,
    }
    version = f"sha256-{hashlib.sha256(_canonical_json(core)).hexdigest()}"
    return {
        "format_version": 1,
        "dataset_name": dataset_name,
        "dataset_version": version,
        "file_count": len(files),
        "total_size_bytes": sum(entry["size_bytes"] for entry in files),
        "total_rows": sum(entry["rows"] for entry in files),
        "files": files,
    }


def manifest_bytes(manifest: DatasetManifest) -> bytes:
    """Serialize a manifest in its canonical form."""
    return _canonical_json(manifest)


def _head_object(client: Any, bucket: str, key: str) -> dict[str, Any] | None:
    try:
        response: dict[str, Any] = client.head_object(Bucket=bucket, Key=key)
        return response
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def _object_is_current(
    client: Any,
    bucket: str,
    key: str,
    size_bytes: int,
    sha256: str,
) -> bool:
    remote = _head_object(client, bucket, key)
    if remote is None:
        return False
    metadata = {
        str(name).lower(): str(value)
        for name, value in remote.get("Metadata", {}).items()
    }
    if int(remote["ContentLength"]) == size_bytes and metadata.get("sha256") == sha256:
        return True
    raise RuntimeError(
        f"Refusing to replace immutable object with different data: {key}"
    )


def _find_multipart_upload(client: Any, bucket: str, key: str) -> str | None:
    paginator = client.get_paginator("list_multipart_uploads")
    candidates: list[dict[str, Any]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=key):
        candidates.extend(
            upload for upload in page.get("Uploads", []) if upload.get("Key") == key
        )
    if not candidates:
        return None
    candidates.sort(key=lambda upload: str(upload.get("Initiated", "")), reverse=True)
    return str(candidates[0]["UploadId"])


def _list_parts(
    client: Any, bucket: str, key: str, upload_id: str
) -> dict[int, tuple[str, int]]:
    paginator = client.get_paginator("list_parts")
    parts: dict[int, tuple[str, int]] = {}
    for page in paginator.paginate(Bucket=bucket, Key=key, UploadId=upload_id):
        for part in page.get("Parts", []):
            parts[int(part["PartNumber"])] = (str(part["ETag"]), int(part["Size"]))
    return parts


def _new_multipart_upload(
    client: Any,
    bucket: str,
    key: str,
    metadata: dict[str, str],
) -> str:
    response = client.create_multipart_upload(
        Bucket=bucket,
        Key=key,
        ContentType="application/octet-stream",
        Metadata=metadata,
    )
    return str(response["UploadId"])


def _expected_part_size(file_size: int, part_size: int, part_number: int) -> int:
    offset = (part_number - 1) * part_size
    return min(part_size, file_size - offset)


def _upload_multipart(
    client: Any,
    bucket: str,
    key: str,
    path: Path,
    metadata: dict[str, str],
    part_size: int,
) -> None:
    file_size = path.stat().st_size
    part_count = (file_size + part_size - 1) // part_size
    if part_count > _MAXIMUM_PARTS:
        raise RuntimeError(
            f"Object requires {part_count} parts; increase --part-size-mib: {key}"
        )

    upload_id = _find_multipart_upload(client, bucket, key)
    if upload_id is None:
        upload_id = _new_multipart_upload(client, bucket, key, metadata)
        existing_parts: dict[int, tuple[str, int]] = {}
    else:
        existing_parts = _list_parts(client, bucket, key, upload_id)
        invalid_parts = [
            number
            for number, (_, size) in existing_parts.items()
            if number > part_count
            or size != _expected_part_size(file_size, part_size, number)
        ]
        if invalid_parts:
            client.abort_multipart_upload(
                Bucket=bucket, Key=key, UploadId=upload_id
            )
            upload_id = _new_multipart_upload(client, bucket, key, metadata)
            existing_parts = {}

    if existing_parts:
        print(f"  resuming after {len(existing_parts)} uploaded parts", flush=True)

    completed_parts: dict[int, str] = {
        number: etag for number, (etag, _) in existing_parts.items()
    }
    uploaded_bytes = sum(size for _, size in existing_parts.values())
    progress = ProgressBar("upload", file_size)
    progress.update(uploaded_bytes)
    with path.open("rb") as source:
        for part_number in range(1, part_count + 1):
            if part_number in completed_parts:
                continue
            expected_size = _expected_part_size(file_size, part_size, part_number)
            source.seek((part_number - 1) * part_size)
            body = source.read(expected_size)
            if len(body) != expected_size:
                raise RuntimeError(f"Local file changed during upload: {path}")
            response = client.upload_part(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
                PartNumber=part_number,
                Body=body,
            )
            completed_parts[part_number] = str(response["ETag"])
            uploaded_bytes += expected_size
            progress.update(uploaded_bytes)

    client.complete_multipart_upload(
        Bucket=bucket,
        Key=key,
        UploadId=upload_id,
        MultipartUpload={
            "Parts": [
                {"ETag": completed_parts[number], "PartNumber": number}
                for number in range(1, part_count + 1)
            ]
        },
    )


def _upload_file(
    client: Any,
    bucket: str,
    key: str,
    path: Path,
    entry: FileEntry,
    dataset_version: str,
    part_size: int,
) -> None:
    if _object_is_current(
        client, bucket, key, entry["size_bytes"], entry["sha256"]
    ):
        print("  already verified; skipping", flush=True)
        return

    metadata = {
        "sha256": entry["sha256"],
        "dataset-version": dataset_version,
    }
    if entry["size_bytes"] <= part_size:
        progress = ProgressBar("upload", entry["size_bytes"])
        with path.open("rb") as body:
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentLength=entry["size_bytes"],
                ContentType="application/octet-stream",
                Metadata=metadata,
            )
        progress.update(entry["size_bytes"])
    else:
        _upload_multipart(client, bucket, key, path, metadata, part_size)

    if not _object_is_current(
        client, bucket, key, entry["size_bytes"], entry["sha256"]
    ):
        raise RuntimeError(f"Uploaded object failed remote verification: {key}")


def publish_dataset(
    client: Any,
    source: Path,
    bucket: str,
    prefix: str,
    manifest: DatasetManifest,
    part_size: int,
) -> str:
    """Upload missing dataset objects and return their versioned S3 prefix."""
    client.head_bucket(Bucket=bucket)
    source = source.expanduser().resolve()
    normalized_prefix = prefix.strip("/")
    if not normalized_prefix or ".." in PurePosixPath(normalized_prefix).parts:
        raise RuntimeError("S3 prefix must be a non-empty relative path")

    root = f"{normalized_prefix}/{manifest['dataset_version']}"
    for index, entry in enumerate(manifest["files"], start=1):
        key = f"{root}/data/{entry['path']}"
        path = source.joinpath(*PurePosixPath(entry["path"]).parts)
        print(f"[{index}/{manifest['file_count']}] publishing {entry['path']}")
        _upload_file(
            client,
            bucket,
            key,
            path,
            entry,
            manifest["dataset_version"],
            part_size,
        )

    payload = manifest_bytes(manifest)
    manifest_key = f"{root}/manifest.json"
    payload_sha256 = hashlib.sha256(payload).hexdigest()
    if not _object_is_current(
        client, bucket, manifest_key, len(payload), payload_sha256
    ):
        client.put_object(
            Bucket=bucket,
            Key=manifest_key,
            Body=payload,
            ContentLength=len(payload),
            ContentType="application/json",
            Metadata={
                "sha256": payload_sha256,
                "dataset-version": manifest["dataset_version"],
            },
        )
    if not _object_is_current(
        client, bucket, manifest_key, len(payload), payload_sha256
    ):
        raise RuntimeError("Uploaded manifest failed remote verification")
    return f"s3://{bucket}/{root}/"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Publish an immutable, content-addressed Parquet dataset to S3."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dataset-name")
    parser.add_argument(
        "--bucket", default=os.environ.get("DEMOFML_DATA_BUCKET", "demofml-data")
    )
    parser.add_argument("--prefix", default="curated")
    parser.add_argument(
        "--endpoint-url", default=os.environ.get("S3_ENDPOINT_URL")
    )
    parser.add_argument("--part-size-mib", type=int, default=16)
    parser.add_argument(
        "--manifest-directory", type=Path, default=Path("artifacts/manifests")
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the local manifest without connecting to S3.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the dataset publisher command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    source: Path = arguments.source
    dataset_name = arguments.dataset_name or source.expanduser().resolve().name.lower()
    part_size = arguments.part_size_mib * 1024 * 1024
    if part_size < _MINIMUM_PART_SIZE:
        parser.error("--part-size-mib must be at least 5")

    try:
        manifest = build_manifest(source, dataset_name)
        manifest_directory: Path = arguments.manifest_directory
        manifest_directory.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_directory / f"{manifest['dataset_version']}.json"
        manifest_path.write_bytes(manifest_bytes(manifest))
        print(f"manifest: {manifest_path}")
        print(f"dataset version: {manifest['dataset_version']}")
        print(
            f"files: {manifest['file_count']}; rows: {manifest['total_rows']}; "
            f"bytes: {manifest['total_size_bytes']}"
        )

        if arguments.dry_run:
            print("dry run complete; no objects uploaded")
            return
        if not arguments.endpoint_url:
            parser.error("set S3_ENDPOINT_URL or pass --endpoint-url")

        client = boto3.client(
            "s3",
            endpoint_url=arguments.endpoint_url,
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 10, "mode": "standard"},
                s3={"addressing_style": "path"},
            ),
        )
        location = publish_dataset(
            client,
            source,
            arguments.bucket,
            arguments.prefix,
            manifest,
            part_size,
        )
        print(f"published and verified: {location}")
    except (ClientError, OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")

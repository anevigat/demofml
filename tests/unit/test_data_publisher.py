import hashlib
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from demofml.data.publisher import (
    DatasetManifest,
    FileEntry,
    build_manifest,
    main,
    manifest_bytes,
    publish_dataset,
)


class _Paginator:
    def __init__(self, client: "_S3", operation: str) -> None:
        self.client = client
        self.operation = operation

    def paginate(self, **arguments: Any) -> list[dict[str, Any]]:
        if self.operation == "list_multipart_uploads":
            uploads = [
                {
                    "Key": upload["key"],
                    "UploadId": upload_id,
                    "Initiated": upload_id,
                }
                for upload_id, upload in self.client.uploads.items()
                if str(upload["key"]).startswith(arguments["Prefix"])
            ]
            return [{"Uploads": uploads}]
        upload = self.client.uploads[arguments["UploadId"]]
        return [
            {
                "Parts": [
                    {
                        "PartNumber": number,
                        "ETag": f'"etag-{number}"',
                        "Size": len(body),
                    }
                    for number, body in sorted(upload["parts"].items())
                ]
            }
        ]


class _S3:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, dict[str, str]]] = {}
        self.uploads: dict[str, dict[str, Any]] = {}
        self.put_calls = 0
        self.aborted = 0

    def head_bucket(self, **arguments: Any) -> None:
        pass

    def head_object(self, **arguments: Any) -> dict[str, Any]:
        key = arguments["Key"]
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404"}}, "HeadObject")
        body, metadata = self.objects[key]
        return {"ContentLength": len(body), "Metadata": metadata}

    def put_object(self, **arguments: Any) -> None:
        body = arguments["Body"]
        payload = body if isinstance(body, bytes) else body.read()
        self.objects[arguments["Key"]] = (payload, arguments["Metadata"])
        self.put_calls += 1

    def get_paginator(self, operation: str) -> _Paginator:
        return _Paginator(self, operation)

    def create_multipart_upload(self, **arguments: Any) -> dict[str, str]:
        upload_id = f"upload-{len(self.uploads) + 1}"
        self.uploads[upload_id] = {
            "key": arguments["Key"],
            "metadata": arguments["Metadata"],
            "parts": {},
        }
        return {"UploadId": upload_id}

    def upload_part(self, **arguments: Any) -> dict[str, str]:
        self.uploads[arguments["UploadId"]]["parts"][arguments["PartNumber"]] = (
            arguments["Body"]
        )
        return {"ETag": f'"etag-{arguments["PartNumber"]}"'}

    def complete_multipart_upload(self, **arguments: Any) -> None:
        upload = self.uploads.pop(arguments["UploadId"])
        body = b"".join(upload["parts"][number] for number in sorted(upload["parts"]))
        self.objects[upload["key"]] = (body, upload["metadata"])

    def abort_multipart_upload(self, **arguments: Any) -> None:
        self.uploads.pop(arguments["UploadId"])
        self.aborted += 1


def _write_parquet(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "timestamp": pa.array(values, type=pa.int64()),
            "bid": pa.array([1.1 + value / 100 for value in values]),
        }
    )
    pq.write_table(table, path, row_group_size=2)


def test_manifest_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    source = tmp_path / "cleaned_ticks"
    _write_parquet(source / "USDJPY" / "b.parquet", [3])
    _write_parquet(source / "EURUSD" / "a.parquet", [1, 2])

    first = build_manifest(source, "cleaned_ticks")
    second = build_manifest(source, "cleaned_ticks")

    assert manifest_bytes(first) == manifest_bytes(second)
    assert first["dataset_version"].startswith("sha256-")
    assert first["file_count"] == 2
    assert first["total_rows"] == 3
    assert [entry["path"] for entry in first["files"]] == [
        "EURUSD/a.parquet",
        "USDJPY/b.parquet",
    ]
    assert first["files"][0]["rows"] == 2
    assert first["files"][0]["row_groups"] == 1
    assert first["files"][0]["schema"][0] == {
        "name": "timestamp",
        "type": "int64",
        "nullable": True,
    }

    _write_parquet(source / "EURUSD" / "a.parquet", [1, 2, 3])
    changed = build_manifest(source, "cleaned_ticks")
    assert changed["dataset_version"] != first["dataset_version"]


def test_dry_run_writes_manifest_without_s3(tmp_path: Path) -> None:
    source = tmp_path / "ticks"
    manifests = tmp_path / "manifests"
    _write_parquet(source / "ticks.parquet", [1, 2])

    main(
        [
            "--source",
            str(source),
            "--manifest-directory",
            str(manifests),
            "--dry-run",
        ]
    )

    generated = list(manifests.glob("sha256-*.json"))
    assert len(generated) == 1
    assert b'"dataset_name":"ticks"' in generated[0].read_bytes()


def test_publish_dataset_uploads_and_skips_verified_objects(tmp_path: Path) -> None:
    source = tmp_path / "ticks"
    _write_parquet(source / "ticks.parquet", [1, 2])
    manifest = build_manifest(source, "ticks")
    client = _S3()

    location = publish_dataset(
        client, source, "data", "curated", manifest, 10 * 1024 * 1024
    )
    repeated = publish_dataset(
        client, source, "data", "curated", manifest, 10 * 1024 * 1024
    )

    assert location == repeated
    assert location == f"s3://data/curated/{manifest['dataset_version']}/"
    assert len(client.objects) == 2
    assert client.put_calls == 2


def _binary_manifest(path: Path, version: str = "sha256-test") -> DatasetManifest:
    payload = path.read_bytes()
    entry: FileEntry = {
        "path": path.name,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "rows": 1,
        "row_groups": 1,
        "schema": [],
    }
    return {
        "format_version": 1,
        "dataset_name": "ticks",
        "dataset_version": version,
        "file_count": 1,
        "total_size_bytes": len(payload),
        "total_rows": 1,
        "files": [entry],
    }


def test_publish_dataset_resumes_multipart_upload(tmp_path: Path) -> None:
    source = tmp_path / "ticks"
    source.mkdir()
    path = source / "large.parquet"
    path.write_bytes(b"hello-world")
    manifest = _binary_manifest(path)
    client = _S3()
    key = "curated/sha256-test/data/large.parquet"
    upload_id = client.create_multipart_upload(
        Key=key,
        Metadata={
            "sha256": manifest["files"][0]["sha256"],
            "dataset-version": "sha256-test",
        },
    )["UploadId"]
    client.upload_part(UploadId=upload_id, PartNumber=1, Body=b"hello")

    publish_dataset(client, source, "data", "curated", manifest, part_size=5)

    assert client.objects[key][0] == b"hello-world"
    assert not client.uploads


def test_publish_dataset_restarts_incompatible_multipart(tmp_path: Path) -> None:
    source = tmp_path / "ticks"
    source.mkdir()
    path = source / "large.parquet"
    path.write_bytes(b"hello-world")
    manifest = _binary_manifest(path)
    client = _S3()
    upload_id = client.create_multipart_upload(
        Key="curated/sha256-test/data/large.parquet",
        Metadata={},
    )["UploadId"]
    client.upload_part(UploadId=upload_id, PartNumber=1, Body=b"bad")

    publish_dataset(client, source, "data", "curated", manifest, part_size=5)

    assert client.aborted == 1


def test_publish_dataset_rejects_invalid_prefix_and_immutable_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "ticks"
    source.mkdir()
    path = source / "file.parquet"
    path.write_bytes(b"data")
    manifest = _binary_manifest(path)
    client = _S3()

    with pytest.raises(RuntimeError, match="relative path"):
        publish_dataset(client, source, "data", "../bad", manifest, 10)

    key = "curated/sha256-test/data/file.parquet"
    client.objects[key] = (b"wrong", {"sha256": "wrong"})
    with pytest.raises(RuntimeError, match="immutable object"):
        publish_dataset(client, source, "data", "curated", manifest, 10)


def test_publisher_main_uses_s3_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "ticks"
    manifests = tmp_path / "manifests"
    _write_parquet(source / "ticks.parquet", [1])
    client = _S3()
    monkeypatch.setattr(
        "demofml.data.publisher.boto3.client", lambda *args, **kwargs: client
    )

    main(
        [
            "--source",
            str(source),
            "--endpoint-url",
            "https://s3.invalid",
            "--manifest-directory",
            str(manifests),
        ]
    )

    assert client.objects


def test_publisher_main_rejects_small_parts(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--source", str(tmp_path), "--part-size-mib", "1"])
    assert error.value.code == 2

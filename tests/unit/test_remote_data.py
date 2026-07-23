import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from demofml.data.publisher import build_manifest, manifest_bytes
from demofml.data.remote import (
    load_development_dataset,
    load_published_manifest,
    materialize_development_file,
    validate_development_parquet,
)

PROJECT_ROOT = Path(__file__).parents[2]
DATASET_CONFIG = (
    PROJECT_ROOT / "configs/datasets/cleaned-ticks-development-v1.toml"
)


class _S3:
    def __init__(self, objects: dict[str, tuple[bytes, dict[str, str]]]) -> None:
        self.objects = objects
        self.get_requests: list[dict[str, Any]] = []

    def get_object(self, **arguments: Any) -> dict[str, io.BytesIO]:
        self.get_requests.append(arguments)
        body, _ = self.objects[arguments["Key"]]
        return {"Body": io.BytesIO(body)}

    def head_object(self, **arguments: Any) -> dict[str, object]:
        body, metadata = self.objects[arguments["Key"]]
        return {
            "ContentLength": len(body),
            "Metadata": metadata,
            "ETag": '"immutable-etag"',
        }


def _write_ticks(path: Path, timestamp: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "timestamp": pa.array([timestamp], type=pa.timestamp("ns", tz="UTC")),
                "bid": pa.array([1.0], type=pa.float64()),
                "ask": pa.array([1.1], type=pa.float64()),
                "mid": pa.array([1.05], type=pa.float64()),
                "spread": pa.array([0.1], type=pa.float64()),
            }
        ),
        path,
        write_statistics=True,
    )


def _fixture(tmp_path: Path, timestamp: datetime) -> tuple[Path, _S3]:
    source = tmp_path / "ticks"
    relative = Path("EURUSD/2018/eurusd.parquet")
    parquet = source / relative
    _write_ticks(parquet, timestamp)
    manifest = build_manifest(source, "cleaned_ticks")
    entry = manifest["files"][0]
    config = tmp_path / "development.toml"
    config.write_text(
        "\n".join(
            [
                "format_version = 1",
                'id = "test-development-v1"',
                'dataset_name = "cleaned_ticks"',
                f'dataset_version = "{manifest["dataset_version"]}"',
                's3_prefix = "curated"',
                'start = "2018-01-01T00:00:00Z"',
                'end_exclusive = "2025-01-01T00:00:00Z"',
                "[[files]]",
                'symbol = "EURUSD"',
                f'path = "{entry["path"]}"',
                f'sha256 = "{entry["sha256"]}"',
                f'size_bytes = {entry["size_bytes"]}',
                f'rows = {entry["rows"]}',
                "",
            ]
        ),
        encoding="utf-8",
    )
    root = f'curated/{manifest["dataset_version"]}'
    payload = parquet.read_bytes()
    objects = {
        f"{root}/manifest.json": (manifest_bytes(manifest), {}),
        f"{root}/data/{entry['path']}": (
            payload,
            {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "dataset-version": manifest["dataset_version"],
            },
        ),
    }
    return config, _S3(objects)


def test_committed_development_allowlist_excludes_locked_objects() -> None:
    dataset = load_development_dataset(DATASET_CONFIG)

    assert len(dataset.files) == 14
    assert dataset.symbols == (
        "AUDUSD",
        "EURCHF",
        "EURJPY",
        "EURUSD",
        "GBPJPY",
        "GBPUSD",
        "USDCAD",
        "USDJPY",
    )
    assert all("2025_now" not in entry.path for entry in dataset.files)
    assert all("recent" not in entry.path for entry in dataset.files)


def test_manifest_and_download_are_verified_and_reusable(tmp_path: Path) -> None:
    config, client = _fixture(tmp_path, datetime(2024, 12, 31, tzinfo=UTC))
    dataset = load_development_dataset(config)

    manifest = load_published_manifest(client, "data", dataset)
    output = materialize_development_file(
        client, "data", dataset, dataset.files[0], tmp_path / "cache"
    )
    repeated = materialize_development_file(
        client, "data", dataset, dataset.files[0], tmp_path / "cache"
    )

    assert manifest["dataset_version"] == dataset.dataset_version
    assert output == repeated
    assert pq.read_table(output).num_rows == 1
    assert all(
        request["IfMatch"] == '"immutable-etag"'
        for request in client.get_requests
    )

    output.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="not immutable"):
        materialize_development_file(
            client, "data", dataset, dataset.files[0], tmp_path / "cache"
        )


def test_download_rejects_locked_timestamp_before_publication(tmp_path: Path) -> None:
    config, client = _fixture(tmp_path, datetime(2025, 1, 1, tzinfo=UTC))
    dataset = load_development_dataset(config)

    with pytest.raises(RuntimeError, match="outside development"):
        materialize_development_file(
            client, "data", dataset, dataset.files[0], tmp_path / "cache"
        )

    assert not list((tmp_path / "cache").rglob("*.parquet"))


def test_download_rejects_timestamp_before_development(tmp_path: Path) -> None:
    config, client = _fixture(tmp_path, datetime(2017, 12, 31, tzinfo=UTC))
    dataset = load_development_dataset(config)

    with pytest.raises(RuntimeError, match="outside development"):
        materialize_development_file(
            client, "data", dataset, dataset.files[0], tmp_path / "cache"
        )


def test_manifest_rejects_content_version_mismatch(tmp_path: Path) -> None:
    config, client = _fixture(tmp_path, datetime(2024, 1, 1, tzinfo=UTC))
    dataset = load_development_dataset(config)
    key = f"curated/{dataset.dataset_version}/manifest.json"
    payload, metadata = client.objects[key]
    client.objects[key] = (
        payload.replace(b'"file_count":1', b'"file_count":2'),
        metadata,
    )

    with pytest.raises(RuntimeError, match="version does not match"):
        load_published_manifest(client, "data", dataset)


def test_actual_timestamp_scan_does_not_trust_footer_statistics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _fixture(tmp_path, datetime(2024, 1, 1, tzinfo=UTC))
    dataset = load_development_dataset(config)
    path = next((tmp_path / "ticks").rglob("*.parquet"))
    parquet = pq.ParquetFile(path)

    class _ForgedParquet:
        schema_arrow = parquet.schema_arrow
        metadata = parquet.metadata

        def iter_batches(self, **arguments: Any) -> list[pa.RecordBatch]:
            return [
                pa.record_batch(
                    [
                        pa.array(
                            [datetime(2025, 1, 1, tzinfo=UTC)],
                            type=pa.timestamp("ns", tz="UTC"),
                        )
                    ],
                    names=["timestamp"],
                )
            ]

    monkeypatch.setattr(
        "demofml.data.remote.pq.ParquetFile", lambda unused: _ForgedParquet()
    )

    with pytest.raises(RuntimeError, match="outside development"):
        validate_development_parquet(
            path,
            dataset.files[0],
            dataset.start,
            dataset.end_exclusive,
        )


def test_download_rejects_symlinked_cache_path(tmp_path: Path) -> None:
    config, client = _fixture(tmp_path, datetime(2024, 1, 1, tzinfo=UTC))
    dataset = load_development_dataset(config)
    cache = tmp_path / "cache"
    outside = tmp_path / "outside"
    cache.mkdir()
    outside.mkdir()
    (cache / "EURUSD").symlink_to(outside, target_is_directory=True)

    with pytest.raises(RuntimeError, match="cache path is unsafe"):
        materialize_development_file(
            client, "data", dataset, dataset.files[0], cache
        )

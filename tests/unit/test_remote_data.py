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
    LOCKED_DATASET_SET_ID,
    load_development_dataset,
    load_locked_test_dataset,
    load_published_manifest,
    materialize_development_file,
    materialize_locked_test_file,
    validate_development_parquet,
    validate_locked_test_parquet,
    verify_materialized_inventory,
)

PROJECT_ROOT = Path(__file__).parents[2]
DATASET_CONFIG = PROJECT_ROOT / "configs/datasets/cleaned-ticks-development-v1.toml"


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
                f"size_bytes = {entry['size_bytes']}",
                f"rows = {entry['rows']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    root = f"curated/{manifest['dataset_version']}"
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


def _locked_fixture(tmp_path: Path, timestamp: datetime) -> tuple[Path, _S3]:
    config, client = _fixture(tmp_path, timestamp)
    contents = config.read_text(encoding="utf-8")
    config.write_text(
        contents.replace(
            'id = "test-development-v1"', f'id = "{LOCKED_DATASET_SET_ID}"'
        )
        .replace(
            'start = "2018-01-01T00:00:00Z"',
            'start = "2025-01-01T00:00:00Z"',
        )
        .replace(
            'end_exclusive = "2025-01-01T00:00:00Z"',
            'end_exclusive = "2026-01-01T00:00:00Z"',
        ),
        encoding="utf-8",
    )
    return config, client


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
        request["IfMatch"] == '"immutable-etag"' for request in client.get_requests
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
        materialize_development_file(client, "data", dataset, dataset.files[0], cache)


def test_locked_dataset_helpers_and_materialization(tmp_path: Path) -> None:
    config, client = _locked_fixture(tmp_path, datetime(2025, 1, 1, tzinfo=UTC))
    dataset = load_locked_test_dataset(config)

    manifest = load_published_manifest(client, "data", dataset)
    output = materialize_locked_test_file(
        client, "data", dataset, dataset.files[0], tmp_path / "locked-cache"
    )
    repeated = materialize_locked_test_file(
        client, "data", dataset, dataset.files[0], tmp_path / "locked-cache"
    )

    assert dataset.id == LOCKED_DATASET_SET_ID
    assert dataset.symbols == ("EURUSD",)
    assert dataset.files_for_symbol("EURUSD") == dataset.files
    assert manifest["dataset_version"] == dataset.dataset_version
    assert output == repeated
    assert verify_materialized_inventory(tmp_path / "locked-cache", dataset.files) == (
        output,
    )
    assert all(
        request["IfMatch"] == '"immutable-etag"' for request in client.get_requests
    )

    output.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="not immutable"):
        materialize_locked_test_file(
            client, "data", dataset, dataset.files[0], tmp_path / "locked-cache"
        )


@pytest.mark.parametrize(
    ("timestamp", "accepted"),
    [
        (datetime(2025, 1, 1, tzinfo=UTC), True),
        (datetime(2025, 12, 31, 23, 59, 59, 999999, tzinfo=UTC), True),
        (datetime(2024, 12, 31, 23, 59, 59, tzinfo=UTC), False),
        (datetime(2026, 1, 1, tzinfo=UTC), False),
    ],
)
def test_locked_timestamp_interval_boundaries(
    tmp_path: Path, timestamp: datetime, accepted: bool
) -> None:
    config, _ = _locked_fixture(tmp_path, timestamp)
    dataset = load_locked_test_dataset(config)
    path = next((tmp_path / "ticks").rglob("*.parquet"))

    if accepted:
        validate_locked_test_parquet(
            path, dataset.files[0], dataset.start, dataset.end_exclusive
        )
    else:
        with pytest.raises(RuntimeError, match="outside locked test"):
            validate_locked_test_parquet(
                path, dataset.files[0], dataset.start, dataset.end_exclusive
            )


def test_locked_scan_rejects_forged_actual_values_despite_valid_footer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _locked_fixture(tmp_path, datetime(2025, 6, 1, tzinfo=UTC))
    dataset = load_locked_test_dataset(config)
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
                            [datetime(2026, 1, 1, tzinfo=UTC)],
                            type=pa.timestamp("ns", tz="UTC"),
                        )
                    ],
                    names=["timestamp"],
                )
            ]

    monkeypatch.setattr(
        "demofml.data.remote.pq.ParquetFile", lambda unused: _ForgedParquet()
    )

    with pytest.raises(RuntimeError, match="Timestamp is outside locked test"):
        validate_locked_test_parquet(
            path, dataset.files[0], dataset.start, dataset.end_exclusive
        )


def test_locked_footer_is_fully_validated_when_actual_values_are_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, _ = _locked_fixture(tmp_path, datetime(2025, 6, 1, tzinfo=UTC))
    dataset = load_locked_test_dataset(config)
    path = next((tmp_path / "ticks").rglob("*.parquet"))
    parquet = pq.ParquetFile(path)

    class _ValidStatistics:
        has_min_max = True
        min = datetime(2025, 6, 1, tzinfo=UTC)

        max = min

    class _InvalidStatistics:
        has_min_max = True
        min = datetime(2025, 6, 2, tzinfo=UTC)
        max = datetime(2026, 1, 1, tzinfo=UTC)

    class _Column:
        def __init__(self, index: int) -> None:
            self.statistics = _ValidStatistics() if index == 0 else _InvalidStatistics()

    class _RowGroup:
        def __init__(self, index: int) -> None:
            self.index = index

        def column(self, unused: int) -> _Column:
            return _Column(self.index)

    class _Metadata:
        num_rows = parquet.metadata.num_rows
        num_row_groups = 2

        def row_group(self, index: int) -> _RowGroup:
            return _RowGroup(index)

    class _ForgedParquet:
        schema_arrow = parquet.schema_arrow
        metadata = _Metadata()

        def iter_batches(self, **arguments: Any) -> Any:
            return parquet.iter_batches(**arguments)

    monkeypatch.setattr(
        "demofml.data.remote.pq.ParquetFile", lambda unused: _ForgedParquet()
    )

    with pytest.raises(RuntimeError, match="statistics are outside locked test"):
        validate_locked_test_parquet(
            path, dataset.files[0], dataset.start, dataset.end_exclusive
        )


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("format_version = 1", "format_version = 2", "format_version"),
        (
            f'id = "{LOCKED_DATASET_SET_ID}"',
            'id = "some-other-dataset"',
            "id must be",
        ),
        (
            'dataset_version = "sha256-',
            'dataset_version = "mutable-',
            "content-addressed",
        ),
        (
            'start = "2025-01-01T00:00:00Z"',
            'start = "2026-01-01T00:00:00Z"',
            "interval must be non-empty",
        ),
        ('path = "EURUSD/', 'path = "../', "safe relative path"),
        ("size_bytes = ", "size_bytes = -", "must be positive"),
    ],
)
def test_locked_dataset_rejects_invalid_config(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    config, _ = _locked_fixture(tmp_path, datetime(2025, 6, 1, tzinfo=UTC))
    contents = config.read_text(encoding="utf-8")
    config.write_text(contents.replace(old, new, 1), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_locked_test_dataset(config)


def test_locked_dataset_rejects_unordered_paths(tmp_path: Path) -> None:
    config, _ = _locked_fixture(tmp_path, datetime(2025, 6, 1, tzinfo=UTC))
    contents = config.read_text(encoding="utf-8")
    entry = contents[contents.index("[[files]]") :]
    second = entry.replace("2018/eurusd.parquet", "2017/eurusd.parquet")
    config.write_text(f"{contents}{second}", encoding="utf-8")

    with pytest.raises(ValueError, match="unique ordered paths"):
        load_locked_test_dataset(config)


def test_materialized_inventory_rejects_extra_missing_and_symlinked_files(
    tmp_path: Path,
) -> None:
    config, client = _locked_fixture(tmp_path, datetime(2025, 6, 1, tzinfo=UTC))
    dataset = load_locked_test_dataset(config)
    cache = tmp_path / "locked-cache"
    materialize_locked_test_file(client, "data", dataset, dataset.files[0], cache)
    extra = cache / "EURUSD/extra.parquet"
    extra.write_bytes(b"extra")

    with pytest.raises(RuntimeError, match="extra Parquet file"):
        verify_materialized_inventory(cache, dataset.files)

    extra.unlink()
    (cache / "linked").symlink_to(tmp_path / "ticks", target_is_directory=True)
    with pytest.raises(RuntimeError, match="contains a symlink"):
        verify_materialized_inventory(cache, dataset.files)

    (cache / "linked").unlink()
    (cache / dataset.files[0].path).unlink()
    with pytest.raises(RuntimeError, match="inventory is missing"):
        verify_materialized_inventory(cache, dataset.files)

from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.data.publisher import build_manifest, main, manifest_bytes


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

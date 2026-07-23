from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.data.splitter import split_dataset, split_file


def test_split_file_preserves_rows_schema_and_order(tmp_path: Path) -> None:
    source = tmp_path / "ticks.parquet"
    destination = tmp_path / "ticks"
    original = pa.table(
        {
            "timestamp": pa.array(range(12), type=pa.int64()),
            "symbol": pa.array(["EURUSD"] * 12),
            "bid": pa.array([1.1 + value / 100 for value in range(12)]),
        }
    )
    pq.write_table(original, source, row_group_size=3, compression="zstd")

    part_count = split_file(source, destination, target_size=1, compression="zstd")
    repeated_count = split_file(
        source, destination, target_size=1, compression="zstd"
    )

    parts = sorted(destination.glob("part-*.parquet"))
    reconstructed = pa.concat_tables([pq.read_table(path) for path in parts])
    assert part_count == 4
    assert repeated_count == part_count
    assert reconstructed.equals(original)
    assert (destination / "split-metadata.json").is_file()


def test_replace_source_is_safe_to_run_again(tmp_path: Path) -> None:
    source_root = tmp_path / "ticks"
    source_root.mkdir()
    source = source_root / "large.parquet"
    table = pa.table({"value": list(range(12))})
    pq.write_table(table, source, row_group_size=3, compression="zstd")

    first = split_dataset(source_root, None, True, 1, "zstd")
    second = split_dataset(source_root, None, True, 1, "zstd")

    assert first == (1, 4)
    assert second == (0, 4)
    assert not source.exists()
    assert len(list((source_root / "large").glob("part-*.parquet"))) == 4

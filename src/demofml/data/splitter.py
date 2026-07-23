"""Streaming conversion of large Parquet files into smaller parts."""

import argparse
import json
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.data.progress import ProgressBar


@dataclass(frozen=True)
class PartPlan:
    """Contiguous source row groups assigned to one output part."""

    first_row_group: int
    last_row_group: int
    rows: int
    compressed_bytes: int


def _row_group_size(metadata: Any, index: int) -> int:
    row_group = metadata.row_group(index)
    return sum(
        int(row_group.column(column).total_compressed_size)
        for column in range(row_group.num_columns)
    )


def plan_parts(metadata: Any, target_size: int) -> list[PartPlan]:
    """Group consecutive row groups by approximate compressed size."""
    plans: list[PartPlan] = []
    first = 0
    rows = 0
    size = 0
    for index in range(metadata.num_row_groups):
        row_group = metadata.row_group(index)
        row_group_size = _row_group_size(metadata, index)
        if index > first and size + row_group_size > target_size:
            plans.append(PartPlan(first, index, rows, size))
            first = index
            rows = 0
            size = 0
        rows += int(row_group.num_rows)
        size += row_group_size
    if metadata.num_row_groups:
        plans.append(PartPlan(first, metadata.num_row_groups, rows, size))
    return plans


def _part_name(index: int) -> str:
    return f"part-{index:05d}.parquet"


def _validate_parts(
    directory: Path,
    plans: list[PartPlan],
    source_schema: Any,
) -> None:
    actual_parts = sorted(directory.glob("part-*.parquet"))
    expected_names = [_part_name(index) for index in range(len(plans))]
    if [path.name for path in actual_parts] != expected_names:
        raise RuntimeError(f"Unexpected split files in: {directory}")

    for path, plan in zip(actual_parts, plans, strict=True):
        metadata = pq.read_metadata(path)
        if int(metadata.num_rows) != plan.rows:
            raise RuntimeError(f"Row count validation failed: {path}")
        if not metadata.schema.to_arrow_schema().equals(source_schema):
            raise RuntimeError(f"Schema validation failed: {path}")


def _write_part(
    parquet: Any,
    plan: PartPlan,
    destination: Path,
    compression: str,
    progress: ProgressBar,
    completed_bytes: int,
) -> int:
    writing = destination.with_suffix(destination.suffix + ".writing")
    partial = destination.with_suffix(destination.suffix + ".partial")
    writing.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        writing,
        parquet.schema_arrow,
        compression=compression,
        use_dictionary=True,
        write_statistics=True,
    )
    try:
        for row_group_index in range(
            plan.first_row_group, plan.last_row_group
        ):
            table = parquet.read_row_group(row_group_index, use_threads=True)
            writer.write_table(table)
            completed_bytes += _row_group_size(parquet.metadata, row_group_index)
            progress.update(completed_bytes)
    finally:
        writer.close()
    writing.replace(partial)
    return completed_bytes


def _write_split_metadata(
    directory: Path,
    source_name: str,
    source_rows: int,
    source_size: int,
    target_size: int,
) -> None:
    payload = {
        "format_version": 1,
        "source_file": source_name,
        "source_rows": source_rows,
        "source_size_bytes": source_size,
        "target_size_bytes": target_size,
    }
    (directory / "split-metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def split_file(
    source: Path,
    destination: Path,
    target_size: int,
    compression: str,
) -> int:
    """Split one Parquet file and atomically publish the validated directory."""
    parquet = pq.ParquetFile(source)
    plans = plan_parts(parquet.metadata, target_size)
    if len(plans) <= 1:
        raise RuntimeError(f"File does not require splitting: {source}")

    if destination.exists():
        _validate_parts(destination, plans, parquet.schema_arrow)
        return len(plans)

    staging = destination.with_name(f".{destination.name}.splitting")
    staging.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar("split", sum(plan.compressed_bytes for plan in plans))
    completed_bytes = 0
    for index, plan in enumerate(plans):
        partial = staging / f"{_part_name(index)}.partial"
        if partial.exists():
            metadata = pq.read_metadata(partial)
            if int(metadata.num_rows) != plan.rows:
                raise RuntimeError(f"Invalid partial split file: {partial}")
            if not metadata.schema.to_arrow_schema().equals(parquet.schema_arrow):
                raise RuntimeError(f"Invalid partial split schema: {partial}")
            completed_bytes += plan.compressed_bytes
            progress.update(completed_bytes)
            continue
        completed_bytes = _write_part(
            parquet,
            plan,
            staging / _part_name(index),
            compression,
            progress,
            completed_bytes,
        )

    for partial in sorted(staging.glob("part-*.parquet.partial")):
        partial.rename(partial.with_suffix(""))
    _write_split_metadata(
        staging,
        source.name,
        int(parquet.metadata.num_rows),
        source.stat().st_size,
        target_size,
    )
    _validate_parts(staging, plans, parquet.schema_arrow)
    staging.rename(destination)
    return len(plans)


def split_dataset(
    source_root: Path,
    output_root: Path | None,
    replace_source: bool,
    target_size: int,
    compression: str,
) -> tuple[int, int]:
    """Split every oversized Parquet file and return input/output counts."""
    source_root = source_root.expanduser().resolve()
    if not source_root.is_dir():
        raise RuntimeError(f"Dataset source is not a directory: {source_root}")
    if replace_source == (output_root is not None):
        raise RuntimeError("Choose exactly one of output_root or replace_source")
    if output_root is not None:
        output_root = output_root.expanduser().resolve()
        if output_root == source_root or output_root.is_relative_to(source_root):
            raise RuntimeError("Output directory must be outside the source dataset")

    all_parquet = sorted(
        (path for path in source_root.rglob("*.parquet") if path.is_file()),
        key=lambda path: path.relative_to(source_root).as_posix(),
    )
    if not all_parquet:
        raise RuntimeError(f"No Parquet files found below: {source_root}")
    inputs = [
        path
        for path in all_parquet
        if not (path.parent / "split-metadata.json").is_file()
    ]
    if not inputs:
        print("All source files have already been converted")
        return 0, len(all_parquet)

    split_inputs = 0
    output_files = 0
    for index, source in enumerate(inputs, start=1):
        relative = source.relative_to(source_root)
        size = source.stat().st_size
        print(f"[{index}/{len(inputs)}] {relative} ({size / 1024**2:.1f} MiB)")
        if size <= target_size:
            if output_root is not None:
                destination = output_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source, destination)
            output_files += 1
            print("  already below target; unchanged")
            continue

        base = source.with_suffix("")
        if replace_source:
            destination_directory = base
        else:
            if output_root is None:
                raise RuntimeError("Output directory is required")
            destination_directory = output_root / relative.parent / base.name
        parts = split_file(source, destination_directory, target_size, compression)
        if replace_source:
            source.unlink()
        split_inputs += 1
        output_files += parts
        print(f"  validated {parts} parts")
    return split_inputs, output_files


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split large Parquet files with bounded memory usage."
    )
    parser.add_argument("--source", type=Path, required=True)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument(
        "--replace-source",
        action="store_true",
        help="Delete each original only after its split parts pass validation.",
    )
    parser.add_argument("--target-size-mib", type=int, default=128)
    parser.add_argument("--compression", choices=["zstd", "snappy"], default="zstd")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Parquet splitter command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    target_size = arguments.target_size_mib * 1024 * 1024
    if target_size < 16 * 1024 * 1024:
        parser.error("--target-size-mib must be at least 16")
    try:
        split_inputs, output_files = split_dataset(
            arguments.source,
            arguments.output,
            arguments.replace_source,
            target_size,
            arguments.compression,
        )
        print(
            f"conversion complete: split {split_inputs} inputs into a dataset "
            f"with {output_files} Parquet files"
        )
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")

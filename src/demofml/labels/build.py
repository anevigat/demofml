"""Atomic construction of executable label Parquet files."""

import argparse
import os
import uuid
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.bars.quotes import validate_quote_bar_schema
from demofml.data.progress import ProgressBar
from demofml.labels.executable import (
    DEFAULT_HORIZONS_MINUTES,
    ExecutableLabelBuilder,
)


@dataclass(frozen=True)
class LabelBuildResult:
    """Summary of one executable-label build."""

    input_bars: int
    output_rows: int


def build_labels(
    source: Path,
    output: Path,
    horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
    minimum_return_bps: float = 0.0,
) -> LabelBuildResult:
    """Build labels from one symbol's quote bars and publish atomically."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"Label source is not a file: {source}")
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing labels: {output}")

    parquet = pq.ParquetFile(source)
    validate_quote_bar_schema(parquet.schema_arrow)
    builder = ExecutableLabelBuilder(horizons_minutes, minimum_return_bps)
    progress = ProgressBar("labels", parquet.metadata.num_rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    writer: pq.ParquetWriter | None = None
    input_rows = 0
    output_rows = 0
    pending: pa.Table | None = None
    try:
        writer = pq.ParquetWriter(
            partial,
            builder.schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        for batch in parquet.iter_batches(batch_size=10_000):
            bars = pa.Table.from_batches([batch], schema=parquet.schema_arrow)
            labels = builder.push(bars)
            input_rows += bars.num_rows
            output_rows += labels.num_rows
            if labels.num_rows:
                pending = (
                    labels
                    if pending is None
                    else pa.concat_tables([pending, labels])
                )
            while pending is not None and pending.num_rows >= 10_000:
                writer.write_table(pending.slice(0, 10_000), row_group_size=10_000)
                pending = pending.slice(10_000)
            progress.update(input_rows)
        trailing = builder.finish()
        output_rows += trailing.num_rows
        if trailing.num_rows:
            pending = (
                trailing
                if pending is None
                else pa.concat_tables([pending, trailing])
            )
        while pending is not None and pending.num_rows >= 10_000:
            writer.write_table(pending.slice(0, 10_000), row_group_size=10_000)
            pending = pending.slice(10_000)
        if pending is not None and pending.num_rows:
            writer.write_table(pending, row_group_size=10_000)
        writer.close()
        writer = None
        os.link(partial, output)
    except FileExistsError as error:
        raise RuntimeError(f"Label dataset appeared during build: {output}") from error
    finally:
        if writer is not None:
            with suppress(Exception):
                writer.close()
        partial.unlink(missing_ok=True)
    return LabelBuildResult(input_rows, output_rows)


def _parse_horizons(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "horizons must be comma-separated integers"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build executable bid/ask labels from quote bars."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--horizons-minutes",
        type=_parse_horizons,
        default=DEFAULT_HORIZONS_MINUTES,
    )
    parser.add_argument("--minimum-return-bps", type=float, default=0.0)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the executable-label build command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = build_labels(
            arguments.source,
            arguments.output,
            arguments.horizons_minutes,
            arguments.minimum_return_bps,
        )
        print(
            f"built {result.output_rows} label rows from "
            f"{result.input_bars} bars: {arguments.output}"
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

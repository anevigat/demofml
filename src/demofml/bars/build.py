"""Bounded-memory construction of causal quote bars from Parquet ticks."""

import argparse
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.bars.quotes import QUOTE_BAR_SCHEMA, QuoteBarBuilder
from demofml.bars.quotes_v2 import (
    QUOTE_BAR_V2_SCHEMA,
    QUOTE_BAR_V2_SET_ID,
    QuoteBarV2Builder,
)
from demofml.data.progress import ProgressBar

_BAR_ROW_GROUP_ROWS = 10_000


@dataclass(frozen=True)
class BarBuildResult:
    """Summary of one completed quote-bar build."""

    input_files: int
    input_rows: int
    output_bars: int


def _row_group_is_after_cutoff(
    parquet: pq.ParquetFile,
    row_group_index: int,
    end_exclusive: datetime,
) -> bool:
    timestamp_index = parquet.schema_arrow.get_field_index("timestamp")
    if timestamp_index < 0:
        return False
    statistics = parquet.metadata.row_group(row_group_index).column(
        timestamp_index
    ).statistics
    if statistics is None or not statistics.has_min_max:
        return False
    minimum = statistics.min
    if not isinstance(minimum, datetime):
        return False
    if minimum.tzinfo is None:
        minimum = minimum.replace(tzinfo=UTC)
    return minimum.astimezone(UTC) >= end_exclusive


def build_quote_bars(
    source: Path,
    output: Path,
    symbol: str,
    interval_minutes: int = 5,
    max_row_groups_per_file: int | None = None,
    *,
    bar_set: str = "quote-bars-v1",
    end_exclusive: datetime | None = None,
) -> BarBuildResult:
    """Stream ordered Parquet files into one atomic quote-bar dataset."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.exists():
        raise RuntimeError(f"Bar source does not exist: {source}")
    paths = [source] if source.is_file() else sorted(source.rglob("*.parquet"))
    paths = [path for path in paths if path.is_file()]
    if not paths:
        raise RuntimeError(f"No Parquet files found below: {source}")
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing bar dataset: {output}")
    builder: QuoteBarBuilder | QuoteBarV2Builder
    if bar_set == "quote-bars-v1":
        builder = QuoteBarBuilder(symbol, interval_minutes)
        schema = QUOTE_BAR_SCHEMA
    elif bar_set == QUOTE_BAR_V2_SET_ID:
        builder = QuoteBarV2Builder(symbol, interval_minutes)
        schema = QUOTE_BAR_V2_SCHEMA
    else:
        raise ValueError(f"unsupported bar set: {bar_set}")
    if end_exclusive is not None and (
        end_exclusive.tzinfo is None or end_exclusive.utcoffset() != timedelta(0)
    ):
        raise ValueError("end_exclusive must be timezone-aware UTC")

    row_group_limits: list[int] = []
    for path in paths:
        row_groups = pq.read_metadata(path).num_row_groups
        row_group_limits.append(
            row_groups
            if max_row_groups_per_file is None
            else min(max_row_groups_per_file, row_groups)
        )
    total_row_groups = sum(row_group_limits)
    progress = ProgressBar("bars", total_row_groups)
    input_rows = 0
    output_rows = 0
    completed_row_groups = 0
    pending_bars: list[pa.Table] = []
    pending_rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    writer = pq.ParquetWriter(
        partial,
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )
    try:
        for path, row_group_limit in zip(paths, row_group_limits, strict=True):
            parquet = pq.ParquetFile(path)
            for index in range(row_group_limit):
                if end_exclusive is not None and _row_group_is_after_cutoff(
                    parquet, index, end_exclusive
                ):
                    completed_row_groups += 1
                    progress.update(completed_row_groups)
                    continue
                ticks = parquet.read_row_group(index)
                if end_exclusive is not None:
                    ticks = ticks.filter(
                        pc.less(
                            ticks.column("timestamp"),
                            pa.scalar(
                                end_exclusive,
                                type=pa.timestamp("ns", tz="UTC"),
                            ),
                        )
                    )
                input_rows += ticks.num_rows
                bars = builder.push(ticks)
                if bars.num_rows:
                    pending_bars.append(bars)
                    pending_rows += bars.num_rows
                    output_rows += bars.num_rows
                if pending_rows >= _BAR_ROW_GROUP_ROWS:
                    writer.write_table(
                        pa.concat_tables(pending_bars),
                        row_group_size=_BAR_ROW_GROUP_ROWS,
                    )
                    pending_bars = []
                    pending_rows = 0
                completed_row_groups += 1
                progress.update(completed_row_groups)
        final = builder.finish()
        if final.num_rows:
            pending_bars.append(final)
            pending_rows += final.num_rows
            output_rows += final.num_rows
        if pending_rows:
            writer.write_table(
                pa.concat_tables(pending_bars), row_group_size=_BAR_ROW_GROUP_ROWS
            )
    except Exception:
        writer.close()
        partial.unlink(missing_ok=True)
        raise
    writer.close()
    try:
        os.link(partial, output)
    except FileExistsError as error:
        raise RuntimeError(f"Bar dataset appeared during build: {output}") from error
    finally:
        partial.unlink(missing_ok=True)
    return BarBuildResult(len(paths), input_rows, output_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build causal quote bars from one symbol's ordered ticks."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument(
        "--bar-set",
        choices=("quote-bars-v1", QUOTE_BAR_V2_SET_ID),
        default="quote-bars-v1",
    )
    parser.add_argument(
        "--end-exclusive",
        help="Optional UTC cutoff; ticks at or after it are not aggregated.",
    )
    parser.add_argument(
        "--max-row-groups-per-file",
        type=int,
        default=0,
        help="Use a positive value for a bounded development build.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the quote-bar build command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.interval_minutes <= 0:
        parser.error("--interval-minutes must be positive")
    if arguments.max_row_groups_per_file < 0:
        parser.error("--max-row-groups-per-file cannot be negative")
    limit = arguments.max_row_groups_per_file or None
    end_exclusive: datetime | None = None
    if arguments.end_exclusive:
        try:
            end_exclusive = datetime.fromisoformat(
                arguments.end_exclusive.replace("Z", "+00:00")
            )
            if (
                end_exclusive.tzinfo is None
                or end_exclusive.utcoffset() != timedelta(0)
            ):
                raise ValueError
            end_exclusive = end_exclusive.astimezone(UTC)
        except ValueError:
            parser.error("--end-exclusive must be an ISO-8601 UTC timestamp")
    try:
        result = build_quote_bars(
            arguments.source,
            arguments.output,
            arguments.symbol,
            arguments.interval_minutes,
            limit,
            bar_set=arguments.bar_set,
            end_exclusive=end_exclusive,
        )
        print(
            f"built {result.output_bars} bars from {result.input_rows} ticks "
            f"across {result.input_files} files: {arguments.output}"
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

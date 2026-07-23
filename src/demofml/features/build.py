"""Atomic streaming construction of causal feature Parquet files."""

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
from demofml.features.causal import FEATURE_SCHEMA, CausalFeatureBuilder


@dataclass(frozen=True)
class FeatureBuildResult:
    """Summary of one feature build."""

    input_bars: int
    output_rows: int


def build_features(source: Path, output: Path, symbol: str) -> FeatureBuildResult:
    """Stream one symbol's quote bars into an atomic feature dataset."""
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if not source.is_file():
        raise RuntimeError(f"Feature source is not a file: {source}")
    if output.exists():
        raise RuntimeError(f"Refusing to replace existing features: {output}")

    parquet = pq.ParquetFile(source)
    validate_quote_bar_schema(parquet.schema_arrow)
    progress = ProgressBar("features", parquet.metadata.num_rows)
    builder = CausalFeatureBuilder(symbol)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    input_rows = 0
    output_rows = 0
    pending: pa.Table | None = None
    writer: pq.ParquetWriter | None = None
    try:
        writer = pq.ParquetWriter(
            partial,
            FEATURE_SCHEMA,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        for batch in parquet.iter_batches(batch_size=10_000):
            bars = pa.Table.from_batches([batch], schema=parquet.schema_arrow)
            features = builder.push(bars)
            input_rows += bars.num_rows
            output_rows += features.num_rows
            if features.num_rows:
                pending = (
                    features
                    if pending is None
                    else pa.concat_tables([pending, features])
                )
            while pending is not None and pending.num_rows >= 10_000:
                writer.write_table(pending.slice(0, 10_000), row_group_size=10_000)
                pending = pending.slice(10_000)
            progress.update(input_rows)
        if pending is not None and pending.num_rows:
            writer.write_table(pending, row_group_size=10_000)
        writer.close()
        writer = None
        try:
            os.link(partial, output)
        except FileExistsError as error:
            raise RuntimeError(
                f"Feature dataset appeared during build: {output}"
            ) from error
    finally:
        if writer is not None:
            with suppress(Exception):
                writer.close()
        partial.unlink(missing_ok=True)
    return FeatureBuildResult(input_rows, output_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build causal features from one symbol's quote bars."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--symbol", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the feature build command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = build_features(arguments.source, arguments.output, arguments.symbol)
        print(
            f"built {result.output_rows} feature rows from "
            f"{result.input_bars} bars: {arguments.output}"
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

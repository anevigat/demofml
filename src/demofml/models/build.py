"""Atomic execution of one symbol's development baseline experiment."""

import argparse
import json
import os
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.evaluation.signals import evaluate_predictions
from demofml.models.baseline import load_baseline_config, run_walk_forward
from demofml.validation.splits import load_validation_plan


@dataclass(frozen=True)
class BaselineBuildResult:
    """Summary of an atomically published baseline experiment."""

    prediction_rows: int
    fold_count: int
    symbol: str


def run_baseline_experiment(
    features_path: Path,
    labels_path: Path,
    validation_config_path: Path,
    model_config_path: Path,
    output: Path,
) -> BaselineBuildResult:
    """Run one symbol and publish predictions plus metrics as one directory."""
    features_path = features_path.expanduser().resolve()
    labels_path = labels_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if not features_path.is_file():
        raise RuntimeError(f"Feature input is not a file: {features_path}")
    if not labels_path.is_file():
        raise RuntimeError(f"Label input is not a file: {labels_path}")
    if output.exists():
        raise RuntimeError(f"Refusing to replace baseline experiment: {output}")

    plan = load_validation_plan(validation_config_path)
    config = load_baseline_config(model_config_path)
    predictions = run_walk_forward(
        pq.read_table(features_path),
        pq.read_table(labels_path),
        plan,
        config,
    )
    report = evaluate_predictions(predictions)
    fold_count = len(set(predictions.column("fold_id").to_pylist()))
    symbols = set(predictions.column("symbol").to_pylist())
    if len(symbols) != 1:
        raise RuntimeError("baseline predictions must contain exactly one symbol")
    symbol = next(iter(symbols))
    if not isinstance(symbol, str):
        raise RuntimeError("baseline prediction symbol is invalid")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    partial.mkdir()
    try:
        pq.write_table(
            predictions,
            partial / "predictions.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        (partial / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.rename(partial, output)
    except FileExistsError as error:
        raise RuntimeError(
            f"Baseline experiment appeared during build: {output}"
        ) from error
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return BaselineBuildResult(predictions.num_rows, fold_count, symbol)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a leakage-safe ridge baseline for one symbol."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the baseline experiment command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_baseline_experiment(
            arguments.features,
            arguments.labels,
            arguments.validation_config,
            arguments.model_config,
            arguments.output,
        )
        print(
            f"built {result.prediction_rows} predictions across "
            f"{result.fold_count} folds for {result.symbol}: {arguments.output}"
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

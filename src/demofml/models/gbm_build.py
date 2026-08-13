"""Atomic execution of one symbol's Campaign 3 gradient-boosting experiment."""

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
from demofml.models.baseline import compute_feature_null_diagnostics
from demofml.models.gbm import load_gbm_config, run_walk_forward_gbm
from demofml.validation.splits import load_validation_plan


@dataclass(frozen=True)
class GBMBuildResult:
    """Summary of an atomically published gradient-boosting experiment."""

    prediction_rows: int
    fold_count: int
    symbol: str


def run_gbm_experiment(
    features_path: Path,
    labels_path: Path,
    validation_config_path: Path,
    model_config_path: Path,
    output: Path,
) -> GBMBuildResult:
    """Run one symbol and publish predictions plus metrics as one directory."""
    features_path = features_path.expanduser().resolve()
    labels_path = labels_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if not features_path.is_file():
        raise RuntimeError(f"Feature input is not a file: {features_path}")
    if not labels_path.is_file():
        raise RuntimeError(f"Label input is not a file: {labels_path}")
    if output.exists():
        raise RuntimeError(f"Refusing to replace gbm experiment: {output}")

    plan = load_validation_plan(validation_config_path)
    config = load_gbm_config(model_config_path)
    features_table = pq.read_table(features_path)
    labels_table = pq.read_table(labels_path)
    predictions = run_walk_forward_gbm(features_table, labels_table, plan, config)
    report = evaluate_predictions(predictions)
    null_diagnostics = compute_feature_null_diagnostics(
        features_table, labels_table, plan, config
    )
    selected = sorted(set(predictions.column("selected_candidate").to_pylist()))
    fold_count = len(set(predictions.column("fold_id").to_pylist()))
    symbols = set(predictions.column("symbol").to_pylist())
    if len(symbols) != 1:
        raise RuntimeError("gbm predictions must contain exactly one symbol")
    symbol = next(iter(symbols))
    if not isinstance(symbol, str):
        raise RuntimeError("gbm prediction symbol is invalid")

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
        (partial / "feature_null_diagnostics.json").write_text(
            json.dumps(null_diagnostics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # Diagnostic sidecars, deliberately outside metrics.json and outside
        # the fingerprinted stage outputs: the acceptance gate compares stored
        # metrics against a recomputation with strict equality, so anything
        # added there would break every future run's gate.
        (partial / "selected_candidates.json").write_text(
            json.dumps(
                {"model_set": config.id, "selected_candidates": selected},
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(partial, output)
    except FileExistsError as error:
        raise RuntimeError(f"GBM experiment appeared during build: {output}") from error
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return GBMBuildResult(predictions.num_rows, fold_count, symbol)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a leakage-safe gradient-boosting fold set for one symbol."
    )
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the gradient-boosting experiment command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_gbm_experiment(
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

"""Frozen promotion gates for the pre-2022 causal-v2 research screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tomllib
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.evaluation.portfolio import PORTFOLIO_HORIZONS, PORTFOLIO_SYMBOLS
from demofml.models.baseline import (
    MODEL_SET_V3_ID,
    PREDICTION_SET_V4_ID,
)
from demofml.validation.splits import (
    SCREEN_VALIDATION_SET_ID,
    ValidationPlan,
    load_validation_plan,
)

SCREEN_ACCEPTANCE_SET_ID = "microstructure-screen-acceptance-v1"


@dataclass(frozen=True)
class ScreenAcceptanceConfig:
    """Immutable scientific gates for the single 2021 holdout."""

    id: str
    pipeline_set: str
    prediction_set: str
    model_set: str
    validation_set: str
    symbols: tuple[str, ...]
    horizons_minutes: tuple[int, ...]
    minimum_mean_return_bps_exclusive: float
    minimum_positive_symbols_per_horizon: int
    minimum_trades_per_symbol_horizon: int
    expected_fold_count: int
    expected_stage_count: int
    locked_test_policy: str

    def __post_init__(self) -> None:
        if (
            self.id != SCREEN_ACCEPTANCE_SET_ID
            or self.pipeline_set != "microstructure-screen-pipeline-v1"
            or self.prediction_set != PREDICTION_SET_V4_ID
            or self.model_set != MODEL_SET_V3_ID
            or self.validation_set != SCREEN_VALIDATION_SET_ID
            or self.symbols != PORTFOLIO_SYMBOLS
            or self.horizons_minutes != PORTFOLIO_HORIZONS
            or self.minimum_mean_return_bps_exclusive != 0.0
            or self.minimum_positive_symbols_per_horizon != 6
            or self.minimum_trades_per_symbol_horizon != 100
            or self.expected_fold_count != 1
            or self.expected_stage_count != 43
            or self.locked_test_policy != "forbidden"
        ):
            raise ValueError(
                "microstructure screen acceptance contract is incompatible"
            )


def load_screen_acceptance_config(path: Path) -> ScreenAcceptanceConfig:
    """Load the exact pre-registered screen gates."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Screen acceptance config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    try:
        return ScreenAcceptanceConfig(
            id=str(values["id"]),
            pipeline_set=str(values["pipeline_set"]),
            prediction_set=str(values["prediction_set"]),
            model_set=str(values["model_set"]),
            validation_set=str(values["validation_set"]),
            symbols=tuple(str(value) for value in values["symbols"]),
            horizons_minutes=tuple(
                int(value) for value in values["horizons_minutes"]
            ),
            minimum_mean_return_bps_exclusive=float(
                values["minimum_mean_return_bps_exclusive"]
            ),
            minimum_positive_symbols_per_horizon=int(
                values["minimum_positive_symbols_per_horizon"]
            ),
            minimum_trades_per_symbol_horizon=int(
                values["minimum_trades_per_symbol_horizon"]
            ),
            expected_fold_count=int(values["expected_fold_count"]),
            expected_stage_count=int(values["expected_stage_count"]),
            locked_test_policy=str(values["locked_test_policy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid screen acceptance config: {error}") from error


_PREDICTION_FIELDS = {
    "model_set": pa.string(),
    "validation_set": pa.string(),
    "fold_id": pa.string(),
    "symbol": pa.string(),
    "decision_time": pa.timestamp("ns", tz="UTC"),
    "entry_time": pa.timestamp("ns", tz="UTC"),
    "exit_time": pa.timestamp("ns", tz="UTC"),
    "horizon_minutes": pa.int16(),
    "predicted_long_return": pa.float64(),
    "predicted_short_return": pa.float64(),
    "action": pa.string(),
    "realized_return": pa.float64(),
}
_PREDICTION_METADATA = {
    b"demofml.prediction_set": PREDICTION_SET_V4_ID.encode(),
    b"demofml.model_set": MODEL_SET_V3_ID.encode(),
    b"demofml.feature_set": b"causal-v2",
    b"demofml.label_set": b"executable-v2",
    b"demofml.validation_set": SCREEN_VALIDATION_SET_ID.encode(),
    b"demofml.horizons_minutes": b"15,30,60",
    b"demofml.action_threshold_bps": b"0.0",
    b"demofml.random_seed": b"1729",
    b"demofml.selection_policy": b"static-threshold-v1",
    b"demofml.calibration_window_months": b"0",
    b"demofml.calibration_purge_minutes": b"0",
    b"demofml.minimum_calibration_rows": b"0",
    b"demofml.calibration_regression": b"none",
}


def _validate_table(table: pa.Table) -> str:
    if table.schema.names != list(_PREDICTION_FIELDS):
        raise ValueError("screen prediction columns do not match predictions-v4")
    for name, expected_type in _PREDICTION_FIELDS.items():
        field = table.schema.field(name)
        if field.type != expected_type or field.nullable:
            raise ValueError(f"screen prediction field {name} is invalid")
    if table.schema.metadata != _PREDICTION_METADATA:
        raise ValueError("screen prediction metadata is incompatible")
    symbols = set(table.column("symbol").to_pylist())
    if len(symbols) != 1:
        raise ValueError("each screen prediction table must contain one symbol")
    symbol = next(iter(symbols))
    if not isinstance(symbol, str):
        raise ValueError("screen prediction symbol is invalid")
    return symbol


def _number(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"screen {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"screen {name} must be finite")
    return number


def evaluate_microstructure_screen(
    prediction_tables: Sequence[pa.Table],
    config: ScreenAcceptanceConfig,
    plan: ValidationPlan,
) -> dict[str, Any]:
    """Recompute the three pre-registered gates from eight prediction tables."""
    if plan.id != config.validation_set:
        raise ValueError("screen validation plan is incompatible")
    if len(prediction_tables) != len(config.symbols):
        raise ValueError("screen requires exactly eight prediction tables")
    table_symbols = [_validate_table(table) for table in prediction_tables]
    if set(table_symbols) != set(config.symbols) or len(set(table_symbols)) != len(
        table_symbols
    ):
        raise ValueError(
            "screen predictions do not contain the canonical eight symbols"
        )

    fold = plan.folds()[0]
    cell_sums: dict[tuple[str, int], float] = defaultdict(float)
    cell_observations: dict[tuple[str, int], int] = defaultdict(int)
    cell_trades: dict[tuple[str, int], int] = defaultdict(int)
    pooled_sums: dict[int, float] = defaultdict(float)
    pooled_observations: dict[int, int] = defaultdict(int)
    previous_decisions: dict[tuple[str, int], datetime] = {}
    for table, expected_symbol in zip(
        prediction_tables, table_symbols, strict=True
    ):
        for batch in table.to_batches(max_chunksize=10_000):
            for row in pa.Table.from_batches([batch]).to_pylist():
                symbol = str(row["symbol"])
                model_set = str(row["model_set"])
                validation_set = str(row["validation_set"])
                fold_id = str(row["fold_id"])
                horizon = int(_number(row["horizon_minutes"], "horizon"))
                decision_time = row["decision_time"]
                entry_time = row["entry_time"]
                exit_time = row["exit_time"]
                predicted_long = _number(
                    row["predicted_long_return"], "predicted_long_return"
                )
                predicted_short = _number(
                    row["predicted_short_return"], "predicted_short_return"
                )
                realized_return = _number(
                    row["realized_return"], "realized_return"
                )
                action = str(row["action"])
                if (
                    symbol != expected_symbol
                    or model_set != config.model_set
                    or validation_set != config.validation_set
                    or fold_id != "wf-2021-01"
                    or horizon not in config.horizons_minutes
                ):
                    raise ValueError("screen prediction row provenance is invalid")
                if (
                    not isinstance(decision_time, datetime)
                    or not isinstance(entry_time, datetime)
                    or not isinstance(exit_time, datetime)
                    or not fold.validation_start
                    <= decision_time
                    < fold.validation_end_exclusive
                    or not decision_time
                    <= entry_time
                    <= decision_time + timedelta(minutes=5)
                    or not decision_time + timedelta(minutes=horizon)
                    <= exit_time
                    <= decision_time + timedelta(minutes=horizon + 5)
                    or exit_time >= plan.development_end_exclusive
                ):
                    raise ValueError(
                        "screen prediction timestamps cross the research cutoff"
                    )
                key = (symbol, horizon)
                previous = previous_decisions.get(key)
                if previous is not None and decision_time <= previous:
                    raise ValueError(
                        "screen prediction keys must be unique and ordered"
                    )
                previous_decisions[key] = decision_time
                expected_action = (
                    "flat"
                    if max(predicted_long, predicted_short) <= 0.0
                    else "long"
                    if predicted_long > predicted_short
                    else "short"
                )
                if action != expected_action or (
                    action == "flat" and realized_return != 0.0
                ):
                    raise ValueError("screen prediction action cannot be reproduced")
                cell_sums[key] += realized_return
                cell_observations[key] += 1
                cell_trades[key] += action != "flat"
                pooled_sums[horizon] += realized_return
                pooled_observations[horizon] += 1

    cells: list[dict[str, object]] = []
    positive_symbols = {horizon: 0 for horizon in config.horizons_minutes}
    minimum_trade_gate = True
    for symbol in config.symbols:
        for horizon in config.horizons_minutes:
            key = (symbol, horizon)
            observations = cell_observations[key]
            if not observations:
                raise ValueError(f"screen cell {symbol}/{horizon} is empty")
            mean_bps = cell_sums[key] / observations * 10_000.0
            trades = cell_trades[key]
            positive_symbols[horizon] += mean_bps > 0.0
            minimum_trade_gate &= trades >= config.minimum_trades_per_symbol_horizon
            cells.append(
                {
                    "symbol": symbol,
                    "horizon_minutes": horizon,
                    "observations": observations,
                    "trades": trades,
                    "mean_executable_return_bps": mean_bps,
                }
            )

    horizon_rows: list[dict[str, object]] = []
    mean_gate = True
    positive_symbol_gate = True
    for horizon in config.horizons_minutes:
        mean_bps = (
            pooled_sums[horizon] / pooled_observations[horizon] * 10_000.0
        )
        mean_passed = mean_bps > config.minimum_mean_return_bps_exclusive
        symbols_passed = (
            positive_symbols[horizon]
            >= config.minimum_positive_symbols_per_horizon
        )
        mean_gate &= mean_passed
        positive_symbol_gate &= symbols_passed
        horizon_rows.append(
            {
                "horizon_minutes": horizon,
                "mean_executable_return_bps": mean_bps,
                "positive_symbols": positive_symbols[horizon],
                "mean_gate_passed": mean_passed,
                "positive_symbol_gate_passed": symbols_passed,
            }
        )
    accepted = mean_gate and positive_symbol_gate and minimum_trade_gate
    return {
        "format_version": 1,
        "acceptance_set": config.id,
        "prediction_set": config.prediction_set,
        "model_set": config.model_set,
        "validation_set": config.validation_set,
        "accepted": accepted,
        "promotion_authorized": False,
        "report_scope": "scientific_result_only",
        "locked_test_policy": config.locked_test_policy,
        "evaluation_interval": {
            "start": fold.validation_start.isoformat().replace("+00:00", "Z"),
            "decision_end_exclusive": fold.validation_end_exclusive.isoformat().replace(
                "+00:00", "Z"
            ),
            "data_end_exclusive": plan.development_end_exclusive.isoformat().replace(
                "+00:00", "Z"
            ),
        },
        "gates": {
            "positive_mean_all_horizons": mean_gate,
            "minimum_positive_symbols_all_horizons": positive_symbol_gate,
            "minimum_trades_all_cells": minimum_trade_gate,
        },
        "horizons": horizon_rows,
        "cells": cells,
        "next_action": (
            "freeze_causal_v2_before_one_2022_2024_walk_forward"
            if accepted
            else "stop_microstructure_research_line"
        ),
    }


def publish_microstructure_screen(
    prediction_paths: Sequence[Path],
    acceptance_config_path: Path,
    validation_config_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Read prepared artifacts and atomically publish the screen decision."""
    if not prediction_paths:
        raise RuntimeError("At least one prediction path is required")
    paths = [path.expanduser().resolve() for path in prediction_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Prediction input is not a file: {missing[0]}")
    output = output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace screen report: {output}")
    config = load_screen_acceptance_config(acceptance_config_path)
    plan = load_validation_plan(validation_config_path)
    report = evaluate_microstructure_screen(
        [pq.read_table(path) for path in paths], config, plan
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    try:
        partial.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(partial, output)
    except FileExistsError as error:
        raise RuntimeError(f"Screen report appeared during build: {output}") from error
    finally:
        partial.unlink(missing_ok=True)
    return report


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{description} is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{description} is invalid: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _expected_stages(symbols: Sequence[str]) -> Counter[tuple[str, str | None]]:
    stages: Counter[tuple[str, str | None]] = Counter(
        {("validation", None): 1, ("portfolio", None): 1, ("screen", None): 1}
    )
    for symbol in symbols:
        for stage in ("bars", "features", "labels", "development", "baseline"):
            stages[(stage, symbol)] = 1
    return stages


def publish_screen_acceptance(
    run_root: Path,
    pipeline_config_path: Path,
    acceptance_config_path: Path,
    scientific_result_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Bind a scientific screen result to its immutable pipeline execution."""
    run_root = run_root.expanduser().resolve()
    scientific_result_path = scientific_result_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if scientific_result_path != run_root / "screen" / "result.json":
        raise ValueError("scientific screen result path is not canonical")
    config = load_screen_acceptance_config(acceptance_config_path)
    result = _read_json_object(scientific_result_path, "Scientific screen result")
    run = _read_json_object(run_root / "run.json", "Pipeline run record")
    execution = _read_json_object(
        run_root / "execution-report.json", "Pipeline execution report"
    )
    run_id = str(run.get("run_id", ""))
    if (
        run.get("pipeline_set") != config.pipeline_set
        or run.get("acceptance_set") != config.id
        or run.get("symbols") != list(config.symbols)
        or run.get("development_only") is not True
        or not run_id.startswith("sha256-")
        or result.get("acceptance_set") != config.id
        or result.get("prediction_set") != config.prediction_set
        or result.get("model_set") != config.model_set
        or result.get("validation_set") != config.validation_set
        or result.get("promotion_authorized") is not False
        or result.get("report_scope") != "scientific_result_only"
    ):
        raise ValueError("screen result and pipeline run provenance differ")
    stages = execution.get("stages")
    if (
        execution.get("pipeline_run_id") != run_id
        or execution.get("status") != "COMPUTE_SUCCEEDED"
        or execution.get("report_scope")
        != "computational_stages_through_microstructure_screen"
        or not isinstance(stages, list)
        or len(stages) != config.expected_stage_count
    ):
        raise ValueError("screen execution report is incompatible")
    actual_stages: Counter[tuple[str, str | None]] = Counter()
    for stage in stages:
        if not isinstance(stage, dict):
            raise ValueError("screen execution stage is invalid")
        name = stage.get("stage")
        symbol = stage.get("symbol")
        if not isinstance(name, str) or not (
            symbol is None or isinstance(symbol, str)
        ):
            raise ValueError("screen execution stage identity is invalid")
        actual_stages[(name, symbol)] += 1
    if actual_stages != _expected_stages(config.symbols):
        raise ValueError("screen execution stages are incomplete")

    marker = _read_json_object(run_root / "screen.stage.json", "Screen stage marker")
    expected_fingerprint = hashlib.sha256(
        f"{run_id}:screen:portfolio".encode()
    ).hexdigest()
    expected_output = {
        "path": "screen/result.json",
        "size_bytes": scientific_result_path.stat().st_size,
        "sha256": _file_sha256(scientific_result_path),
    }
    if marker.get("fingerprint") != expected_fingerprint or marker.get(
        "outputs"
    ) != [expected_output]:
        raise ValueError("scientific screen stage marker is incompatible")

    report = {
        **result,
        "report_scope": "pipeline_acceptance",
        "promotion_authorized": bool(result.get("accepted")),
        "pipeline_set": config.pipeline_set,
        "pipeline_run_id": run_id,
        "code_reference": run.get("code_reference"),
        "pipeline_config_sha256": _file_sha256(
            pipeline_config_path.expanduser().resolve()
        ),
        "acceptance_config_sha256": _file_sha256(
            acceptance_config_path.expanduser().resolve()
        ),
        "execution_stage_count": len(stages),
        "scientific_result_sha256": expected_output["sha256"],
    }
    if output.exists():
        existing = _read_json_object(output, "Screen acceptance report")
        if existing != report:
            raise RuntimeError("Screen acceptance report differs")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    try:
        partial.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(partial, output)
    except FileExistsError as error:
        raise RuntimeError(
            f"Screen acceptance report appeared during build: {output}"
        ) from error
    finally:
        partial.unlink(missing_ok=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the frozen pre-2022 microstructure screen gates."
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--acceptance-config", type=Path, required=True)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the screen evaluator only after the eight predictions exist."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        report = publish_microstructure_screen(
            arguments.predictions,
            arguments.acceptance_config,
            arguments.validation_config,
            arguments.output,
        )
        print(
            f"microstructure screen accepted={report['accepted']}: "
            f"{arguments.output}"
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

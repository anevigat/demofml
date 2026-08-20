"""Frozen development-only acceptance gates for completed research runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tomllib
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.evaluation.portfolio import (
    PORTFOLIO_HORIZONS,
    PORTFOLIO_SET_ID,
    PORTFOLIO_SET_V2_ID,
    PORTFOLIO_SET_V4_ID,
    PORTFOLIO_SET_V5_ID,
    PORTFOLIO_SYMBOLS,
    load_portfolio_config,
    simulate_portfolio,
)
from demofml.evaluation.signals import evaluate_predictions
from demofml.models.baseline import (
    MODEL_SET_ID,
    MODEL_SET_V2_ID,
    PREDICTION_SET_ID,
    PREDICTION_SET_V3_ID,
)
from demofml.models.gbm import (
    GBM_MODEL_SET_ID,
    GBM_MODEL_SET_V2_ID,
    GBM_PREDICTION_SET_ID,
    GBM_PREDICTION_SET_V2_ID,
)
from demofml.reporting.portfolio import portfolio_report
from demofml.research.envelope import (
    SealedEnvelope,
    load_sealed_envelope,
    verify_sealed_envelope,
)
from demofml.validation.splits import (
    CAMPAIGN_3_VALIDATION_SET_ID,
    VALIDATION_SET_ID,
    ValidationPlan,
    load_validation_plan,
)

ACCEPTANCE_SET_ID = "development-acceptance-v1"
PIPELINE_SET_ID = "development-pipeline-v2"
ACCEPTANCE_SET_V2_ID = "development-acceptance-v2"
PIPELINE_SET_V3_ID = "development-pipeline-v3"
ACCEPTANCE_SET_V3_ID = "campaign-3-lightgbm-causal-v2-acceptance-v1"
PIPELINE_SET_V4_ID = "campaign-3-lightgbm-causal-v2-pipeline-v1"
ACCEPTANCE_SET_V4_ID = "campaign-3-lightgbm-gated-causal-v2-acceptance-v1"
PIPELINE_SET_V5_ID = "campaign-3-lightgbm-gated-causal-v2-pipeline-v1"
DATASET_SET_ID = "cleaned-ticks-development-v1"
_ACCEPTANCE_PROVENANCE = {
    ACCEPTANCE_SET_ID: (
        PIPELINE_SET_ID,
        MODEL_SET_ID,
        PORTFOLIO_SET_ID,
        PREDICTION_SET_ID,
    ),
    ACCEPTANCE_SET_V2_ID: (
        PIPELINE_SET_V3_ID,
        MODEL_SET_V2_ID,
        PORTFOLIO_SET_V2_ID,
        PREDICTION_SET_V3_ID,
    ),
    ACCEPTANCE_SET_V3_ID: (
        PIPELINE_SET_V4_ID,
        GBM_MODEL_SET_ID,
        PORTFOLIO_SET_V4_ID,
        GBM_PREDICTION_SET_ID,
    ),
    ACCEPTANCE_SET_V4_ID: (
        PIPELINE_SET_V5_ID,
        GBM_MODEL_SET_V2_ID,
        PORTFOLIO_SET_V5_ID,
        GBM_PREDICTION_SET_V2_ID,
    ),
}
# Campaign 1/2 all ran on purged-walk-forward-v1; Campaign 3 runs the same
# folds over causal-v2 under its own validation id, so the expected validation
# contract is now per-acceptance rather than a single constant.
_ACCEPTANCE_VALIDATION_SETS = {
    ACCEPTANCE_SET_ID: VALIDATION_SET_ID,
    ACCEPTANCE_SET_V2_ID: VALIDATION_SET_ID,
    ACCEPTANCE_SET_V3_ID: CAMPAIGN_3_VALIDATION_SET_ID,
    ACCEPTANCE_SET_V4_ID: CAMPAIGN_3_VALIDATION_SET_ID,
}
# The run directory and stage name holding a run's predictions. Campaign 1/2
# runs on disk use "baseline" and must keep replaying unchanged; Campaign 3
# uses "model", because a directory named "baseline" holding gradient-boosting
# output would misdescribe every artifact inside it.
_ACCEPTANCE_MODEL_STAGES = {
    ACCEPTANCE_SET_ID: "baseline",
    ACCEPTANCE_SET_V2_ID: "baseline",
    ACCEPTANCE_SET_V3_ID: "model",
    ACCEPTANCE_SET_V4_ID: "model",
}
_HASH_BLOCK_SIZE = 8 * 1024 * 1024
# purged-walk-forward-v1's locked-test start, used only as the fallback for
# acceptance configs that predate the optional `validation_config` field (see
# AcceptanceConfig.validation_config) and therefore cannot derive it from a
# real ValidationPlan. Campaign 1/2 configs rely on this fallback; do not
# change it, since it must keep matching purged-walk-forward-v1 exactly.
_LEGACY_LOCKED_TEST_START = datetime(2025, 1, 1, tzinfo=UTC)
_RUN_ID_PATTERN = re.compile(r"^sha256-[0-9a-f]{64}$")
_CODE_REFERENCE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class AcceptanceConfig:
    """Immutable criteria fixed before observing full development results."""

    id: str
    pipeline_set: str
    dataset_set: str
    validation_set: str
    model_set: str
    portfolio_set: str
    portfolio_config: Path
    symbols: tuple[str, ...]
    horizons_minutes: tuple[int, ...]
    expected_fold_count: int
    expected_stage_count: int
    expected_authorized_files: int
    expected_source_rows: int
    locked_test_policy: str
    minimum_positive_folds_per_horizon: int
    minimum_positive_symbols_per_horizon: int
    minimum_trades_per_symbol_horizon: int
    minimum_mean_executable_return_bps_exclusive: float
    minimum_total_return_exclusive: float
    maximum_drawdown_exclusive: float
    minimum_realized_annual_volatility: float
    maximum_realized_annual_volatility: float
    maximum_gross_leverage: float
    require_no_drawdown_halt: bool
    reconciliation_tolerance_usd: float
    validation_config: Path | None = None
    sealed_envelope: Path | None = None


def load_acceptance_config(path: Path) -> AcceptanceConfig:
    """Load and strictly validate the Phase 12 acceptance contract."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Acceptance config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    try:
        if int(values["format_version"]) != 1:
            raise ValueError("acceptance format_version must be 1")
        model = values["model"]
        portfolio = values["portfolio"]
        config = AcceptanceConfig(
            id=str(values["id"]),
            pipeline_set=str(values["pipeline_set"]),
            dataset_set=str(values["dataset_set"]),
            validation_set=str(values["validation_set"]),
            model_set=str(values["model_set"]),
            portfolio_set=str(values["portfolio_set"]),
            portfolio_config=(path.parent / str(values["portfolio_config"])).resolve(),
            symbols=tuple(str(value) for value in values["symbols"]),
            horizons_minutes=tuple(
                int(value) for value in values["horizons_minutes"]
            ),
            expected_fold_count=int(values["expected_fold_count"]),
            expected_stage_count=int(values["expected_stage_count"]),
            expected_authorized_files=int(values["expected_authorized_files"]),
            expected_source_rows=int(values["expected_source_rows"]),
            locked_test_policy=str(values["locked_test_policy"]),
            minimum_positive_folds_per_horizon=int(
                model["minimum_positive_folds_per_horizon"]
            ),
            minimum_positive_symbols_per_horizon=int(
                model["minimum_positive_symbols_per_horizon"]
            ),
            minimum_trades_per_symbol_horizon=int(
                model["minimum_trades_per_symbol_horizon"]
            ),
            minimum_mean_executable_return_bps_exclusive=float(
                model["minimum_mean_executable_return_bps_exclusive"]
            ),
            minimum_total_return_exclusive=float(
                portfolio["minimum_total_return_exclusive"]
            ),
            maximum_drawdown_exclusive=float(
                portfolio["maximum_drawdown_exclusive"]
            ),
            minimum_realized_annual_volatility=float(
                portfolio["minimum_realized_annual_volatility"]
            ),
            maximum_realized_annual_volatility=float(
                portfolio["maximum_realized_annual_volatility"]
            ),
            maximum_gross_leverage=float(portfolio["maximum_gross_leverage"]),
            require_no_drawdown_halt=bool(
                portfolio["require_no_drawdown_halt"]
            ),
            reconciliation_tolerance_usd=float(
                portfolio["reconciliation_tolerance_usd"]
            ),
            validation_config=(
                (path.parent / str(values["validation_config"])).resolve()
                if "validation_config" in values
                else None
            ),
            sealed_envelope=(
                (path.parent / str(values["sealed_envelope"])).resolve()
                if "sealed_envelope" in values
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid acceptance config field: {error}") from error
    if config.validation_config is not None and not config.validation_config.is_file():
        raise RuntimeError(
            f"Acceptance validation config is not a file: {config.validation_config}"
        )
    if config.sealed_envelope is not None and not config.sealed_envelope.is_file():
        raise RuntimeError(
            f"Acceptance sealed envelope is not a file: {config.sealed_envelope}"
        )
    provenance = _ACCEPTANCE_PROVENANCE.get(config.id)
    if provenance is None:
        raise ValueError("acceptance id is not supported")
    pipeline_set, model_set, portfolio_set, _ = provenance
    expected_validation_set = _ACCEPTANCE_VALIDATION_SETS[config.id]
    if (
        config.pipeline_set != pipeline_set
        or config.dataset_set != DATASET_SET_ID
        or config.validation_set != expected_validation_set
        or config.model_set != model_set
        or config.portfolio_set != portfolio_set
    ):
        raise ValueError("acceptance provenance is incompatible")
    if not config.portfolio_config.is_file():
        raise RuntimeError(
            f"Acceptance portfolio config is not a file: {config.portfolio_config}"
        )
    if config.symbols != PORTFOLIO_SYMBOLS:
        raise ValueError("acceptance symbols must be the canonical eight")
    if config.horizons_minutes != PORTFOLIO_HORIZONS:
        raise ValueError("acceptance horizons must be 15, 30, and 60 minutes")
    if config.locked_test_policy != "forbidden":
        raise ValueError("acceptance locked test policy must remain forbidden")
    positive_integers = (
        config.expected_fold_count,
        config.expected_stage_count,
        config.expected_authorized_files,
        config.expected_source_rows,
        config.minimum_positive_folds_per_horizon,
        config.minimum_positive_symbols_per_horizon,
        config.minimum_trades_per_symbol_horizon,
    )
    if any(value <= 0 for value in positive_integers):
        raise ValueError("acceptance counts must be positive")
    if config.minimum_positive_folds_per_horizon > config.expected_fold_count:
        raise ValueError("positive-fold threshold exceeds the expected folds")
    if config.minimum_positive_symbols_per_horizon > len(config.symbols):
        raise ValueError("positive-symbol threshold exceeds the universe")
    finite_values = (
        config.minimum_mean_executable_return_bps_exclusive,
        config.minimum_total_return_exclusive,
        config.maximum_drawdown_exclusive,
        config.minimum_realized_annual_volatility,
        config.maximum_realized_annual_volatility,
        config.maximum_gross_leverage,
        config.reconciliation_tolerance_usd,
    )
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("acceptance thresholds must be finite")
    if not (
        0.0 < config.maximum_drawdown_exclusive < 1.0
        and 0.0 <= config.minimum_realized_annual_volatility
        < config.maximum_realized_annual_volatility
        and config.maximum_gross_leverage > 0.0
        and config.reconciliation_tolerance_usd > 0.0
    ):
        raise ValueError("acceptance risk thresholds are invalid")
    return config


def _resolved_validation_plan(config: AcceptanceConfig) -> ValidationPlan | None:
    """Load the acceptance contract's real validation plan, if it declares one."""
    if config.validation_config is None:
        return None
    return load_validation_plan(config.validation_config)


def _verified_sealed_envelope(
    config: AcceptanceConfig, acceptance_config_path: Path
) -> SealedEnvelope | None:
    """Verify the seal pre-registered over this run's four sealed documents.

    A broken seal raises instead of emitting a failed check: it means the
    contract being evaluated is not the contract that was committed before the
    first fold ran, so neither a pass nor a fail from this gate would carry the
    meaning the acceptance report claims for it. Acceptance configs that
    declare no envelope (every Campaign 1/2 contract) are unaffected.
    """
    if config.sealed_envelope is None:
        return None
    envelope = load_sealed_envelope(config.sealed_envelope)
    verify_sealed_envelope(envelope)
    resolved = acceptance_config_path.expanduser().resolve()
    sealed_acceptance = envelope.resolved_path("acceptance")
    if resolved != sealed_acceptance:
        raise RuntimeError(
            f"Acceptance contract is not the sealed one: {resolved} "
            f"was sealed as {sealed_acceptance}"
        )
    return envelope


def _locked_test_start(plan: ValidationPlan | None) -> datetime:
    """Return the acceptance run's locked-test boundary.

    Derived from the acceptance contract's own validation plan when one is
    declared; falls back to purged-walk-forward-v1's boundary otherwise, which
    keeps every acceptance config written before `validation_config` existed
    behaving exactly as before.
    """
    if plan is None:
        return _LEGACY_LOCKED_TEST_START
    return plan.locked_test_start


def _expected_fold_ids(
    plan: ValidationPlan | None, expected_fold_count: int
) -> set[str]:
    """Return the fold ids the acceptance contract's validation plan requires."""
    if plan is None:
        return {
            f"wf-{2022 + index // 12}-{index % 12 + 1:02d}"
            for index in range(expected_fold_count)
        }
    fold_ids = {fold.id for fold in plan.folds()}
    if len(fold_ids) != expected_fold_count:
        raise RuntimeError(
            "validation plan fold count does not match the acceptance contract"
        )
    return fold_ids


def _locked_start_ns(locked_test_start: datetime) -> np.datetime64:
    return np.datetime64(locked_test_start.replace(tzinfo=None), "ns")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{description} is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{description} is invalid: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{description} must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _model_artifact(
    root: Path, symbol: str, config: AcceptanceConfig, name: str
) -> Path:
    """Return one symbol's model-stage artifact for this acceptance contract."""
    return root / "symbols" / symbol / _ACCEPTANCE_MODEL_STAGES[config.id] / name


def _stage_specs(
    root: Path, symbols: Sequence[str], model_stage: str = "baseline"
) -> list[tuple[str, str | None, Path, list[Path]]]:
    specs: list[tuple[str, str | None, Path, list[Path]]] = [
        (
            "validation",
            None,
            root / "validation" / "stage.json",
            [root / "validation" / "manifest.json"],
        )
    ]
    for symbol in symbols:
        symbol_root = root / "symbols" / symbol
        specs.extend(
            [
                (
                    "bars",
                    symbol,
                    symbol_root / "bars.stage.json",
                    [symbol_root / "bars.parquet"],
                ),
                (
                    "features",
                    symbol,
                    symbol_root / "features.stage.json",
                    [symbol_root / "features-full.parquet"],
                ),
                (
                    "labels",
                    symbol,
                    symbol_root / "labels.stage.json",
                    [symbol_root / "labels-full.parquet"],
                ),
                (
                    "development",
                    symbol,
                    symbol_root / "development.stage.json",
                    [
                        symbol_root / "development" / "features.parquet",
                        symbol_root / "development" / "labels.parquet",
                    ],
                ),
                (
                    model_stage,
                    symbol,
                    symbol_root / f"{model_stage}.stage.json",
                    [
                        symbol_root / model_stage / "predictions.parquet",
                        symbol_root / model_stage / "metrics.json",
                    ],
                ),
            ]
        )
    specs.append(
        (
            "portfolio",
            None,
            root / "portfolio.stage.json",
            [
                root / "portfolio" / "ledger.parquet",
                root / "portfolio" / "equity.parquet",
                root / "portfolio" / "period-returns.parquet",
                root / "portfolio" / "metrics.json",
            ],
        )
    )
    return specs


def _verify_stages(
    root: Path, run_id: str, symbols: Sequence[str], model_stage: str
) -> int:
    specs = _stage_specs(root, symbols, model_stage)
    for stage, symbol, marker, outputs in specs:
        record = _read_json(marker, "Stage marker")
        expected_fingerprint = hashlib.sha256(
            f"{run_id}:{stage}:{symbol or 'portfolio'}".encode()
        ).hexdigest()
        if record.get("format_version") != 1:
            raise RuntimeError("Stage marker format is incompatible")
        if record.get("fingerprint") != expected_fingerprint:
            raise RuntimeError("Stage marker fingerprint differs")
        expected_outputs = []
        for output in outputs:
            if not output.is_file():
                raise RuntimeError("Stage output is missing")
            expected_outputs.append(
                {
                    "path": output.relative_to(root).as_posix(),
                    "size_bytes": output.stat().st_size,
                    "sha256": _sha256(output),
                }
            )
        if record.get("outputs") != expected_outputs:
            raise RuntimeError("Stage output hashes differ")
    return len(specs)


def _valid_execution_stages(
    rows: object, config: AcceptanceConfig, attempt_mode: object
) -> bool:
    if not isinstance(rows, list) or len(rows) != config.expected_stage_count:
        return False
    expected_specs = _stage_specs(
        Path("/run"), config.symbols, _ACCEPTANCE_MODEL_STAGES[config.id]
    )
    expected_outputs = {
        (stage, symbol): {
            output.relative_to("/run").as_posix() for output in outputs
        }
        for stage, symbol, _, outputs in expected_specs
    }
    seen: set[tuple[str, str | None]] = set()
    for row in rows:
        if not isinstance(row, dict):
            return False
        if set(row) != {
            "stage",
            "symbol",
            "action",
            "resumed",
            "elapsed_ns",
            "build_elapsed_ns",
            "peak_rss_bytes_at_end",
            "outputs",
        }:
            return False
        stage = row.get("stage")
        symbol = row.get("symbol")
        key = (str(stage), str(symbol) if symbol is not None else None)
        if key in seen or key not in expected_outputs:
            return False
        seen.add(key)
        action = row.get("action")
        build_elapsed = row.get("build_elapsed_ns")
        if action not in {
            "executed",
            "verified_skipped",
            "checkpoint_recovered",
        }:
            return False
        if not isinstance(row.get("resumed"), bool):
            return False
        if action in {"verified_skipped", "checkpoint_recovered"} and not row[
            "resumed"
        ]:
            return False
        if not isinstance(row.get("elapsed_ns"), int) or row["elapsed_ns"] <= 0:
            return False
        if (
            not isinstance(row.get("peak_rss_bytes_at_end"), int)
            or row["peak_rss_bytes_at_end"] <= 0
        ):
            return False
        if action == "executed":
            if not isinstance(build_elapsed, int) or build_elapsed <= 0:
                return False
        elif build_elapsed is not None:
            return False
        outputs = row.get("outputs")
        if not isinstance(outputs, list | tuple):
            return False
        paths: set[str] = set()
        for output in outputs:
            if not isinstance(output, dict):
                return False
            if set(output) != {"path", "size_bytes", "rows"}:
                return False
            path_value = output.get("path")
            if not isinstance(path_value, str):
                return False
            path = PurePosixPath(path_value)
            if path.is_absolute() or ".." in path.parts:
                return False
            size = output.get("size_bytes")
            rows_value = output.get("rows")
            if not isinstance(size, int) or size < 0:
                return False
            if rows_value is not None and (
                not isinstance(rows_value, int) or rows_value < 0
            ):
                return False
            paths.add(path.as_posix())
        if len(paths) != len(outputs) or paths != expected_outputs[key]:
            return False
    if seen != set(expected_outputs):
        return False
    if attempt_mode == "fresh":
        return all(
            row["action"] == "executed" and row["resumed"] is False
            for row in rows
        )
    if attempt_mode == "resumed_incomplete":
        return any(
            row["resumed"] is True or row["action"] != "executed"
            for row in rows
        )
    if attempt_mode == "verify_completed":
        return all(row["action"] == "verified_skipped" for row in rows)
    return False


def _check(
    identifier: str,
    passed: bool,
    observed: object,
    operator: str,
    threshold: object,
) -> dict[str, object]:
    return {
        "id": identifier,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
    }


def _model_observations(
    root: Path,
    config: AcceptanceConfig,
    expected_fold_ids: set[str],
) -> tuple[dict[int, float], dict[int, int], dict[int, int], int, int]:
    weighted_horizon_returns: dict[int, float] = defaultdict(float)
    horizon_observations: dict[int, int] = defaultdict(int)
    positive_symbols: dict[int, int] = defaultdict(int)
    weighted_fold_returns: dict[tuple[str, int], float] = defaultdict(float)
    fold_observations: dict[tuple[str, int], int] = defaultdict(int)
    minimum_trades: int | None = None
    metric_cells = 0
    expected_horizons = set(config.horizons_minutes)
    for symbol in config.symbols:
        predictions = pq.read_table(
            _model_artifact(root, symbol, config, "predictions.parquet")
        )
        symbols = set(predictions.column("symbol").to_pylist())
        if symbols != {symbol}:
            raise RuntimeError("Prediction symbol differs from its run directory")
        recomputed = evaluate_predictions(predictions)
        report = _read_json(
            _model_artifact(root, symbol, config, "metrics.json"),
            "Baseline metrics",
        )
        if report != recomputed:
            raise RuntimeError("Baseline metrics differ from predictions")
        if (
            report.get("format_version") != 1
            or report.get("model_set") != config.model_set
            or report.get("validation_set") != config.validation_set
        ):
            raise RuntimeError("Baseline metric provenance is incompatible")
        aggregate = report.get("aggregate")
        folds = report.get("folds")
        if not isinstance(aggregate, list) or not isinstance(folds, list):
            raise RuntimeError("Baseline metric groups are invalid")
        aggregate_by_horizon: dict[int, dict[str, Any]] = {}
        for row in aggregate:
            horizon = int(row["horizon_minutes"])
            if horizon in aggregate_by_horizon:
                raise RuntimeError("Baseline aggregate horizons are duplicated")
            aggregate_by_horizon[horizon] = row
        if set(aggregate_by_horizon) != expected_horizons:
            raise RuntimeError("Baseline aggregate horizons are incomplete")

        folds_by_key: dict[tuple[str, int], dict[str, Any]] = {}
        for row in folds:
            key = (str(row["fold_id"]), int(row["horizon_minutes"]))
            if key in folds_by_key:
                raise RuntimeError("Baseline fold metrics are duplicated")
            observations = int(row["observations"])
            trades = int(row["trades"])
            mean = float(row["mean_executable_return_bps"])
            if (
                observations <= 0
                or not 0 <= trades <= observations
                or not math.isfinite(mean)
            ):
                raise RuntimeError("Baseline fold metric is invalid")
            folds_by_key[key] = row
        expected_keys = {
            (fold_id, horizon)
            for fold_id in expected_fold_ids
            for horizon in config.horizons_minutes
        }
        if set(folds_by_key) != expected_keys:
            raise RuntimeError("Baseline fold metric cells are incomplete")

        for horizon, row in aggregate_by_horizon.items():
            observations = int(row["observations"])
            trades = int(row["trades"])
            mean = float(row["mean_executable_return_bps"])
            if (
                observations <= 0
                or not 0 <= trades <= observations
                or not math.isfinite(mean)
            ):
                raise RuntimeError("Baseline aggregate metric is invalid")
            horizon_folds = [
                fold_row
                for (fold_id, fold_horizon), fold_row in folds_by_key.items()
                if fold_horizon == horizon
            ]
            fold_observation_total = sum(
                int(fold_row["observations"]) for fold_row in horizon_folds
            )
            fold_trade_total = sum(
                int(fold_row["trades"]) for fold_row in horizon_folds
            )
            fold_weighted_mean = sum(
                float(fold_row["mean_executable_return_bps"])
                * int(fold_row["observations"])
                for fold_row in horizon_folds
            ) / fold_observation_total
            if (
                observations != fold_observation_total
                or trades != fold_trade_total
                or not math.isclose(mean, fold_weighted_mean, abs_tol=1e-9)
            ):
                raise RuntimeError("Baseline aggregate and folds do not reconcile")
            weighted_horizon_returns[horizon] += mean * observations
            horizon_observations[horizon] += observations
            positive_symbols[horizon] += int(
                mean > config.minimum_mean_executable_return_bps_exclusive
            )
            minimum_trades = trades if minimum_trades is None else min(
                minimum_trades, trades
            )
        for (fold_id, horizon), row in folds_by_key.items():
            observations = int(row["observations"])
            trades = int(row["trades"])
            mean = float(row["mean_executable_return_bps"])
            if (
                observations <= 0
                or not 0 <= trades <= observations
                or not math.isfinite(mean)
            ):
                raise RuntimeError("Baseline fold metric is invalid")
            weighted_fold_returns[(fold_id, horizon)] += mean * observations
            fold_observations[(fold_id, horizon)] += observations
        metric_cells += len(aggregate_by_horizon) + len(folds_by_key)
    horizon_means = {
        horizon: weighted_horizon_returns[horizon] / horizon_observations[horizon]
        for horizon in config.horizons_minutes
    }
    positive_folds: dict[int, int] = defaultdict(int)
    for (fold_id, horizon), weighted_return in weighted_fold_returns.items():
        mean = weighted_return / fold_observations[(fold_id, horizon)]
        positive_folds[horizon] += int(
            mean > config.minimum_mean_executable_return_bps_exclusive
        )
    return (
        horizon_means,
        positive_folds,
        positive_symbols,
        minimum_trades or 0,
        metric_cells,
    )


def _prediction_timestamps_are_safe(
    root: Path, config: AcceptanceConfig, locked_test_start: datetime
) -> bool:
    locked_start = _locked_start_ns(locked_test_start)
    prediction_set = _ACCEPTANCE_PROVENANCE[config.id][3]
    for symbol in config.symbols:
        parquet = pq.ParquetFile(
            _model_artifact(root, symbol, config, "predictions.parquet")
        )
        metadata = parquet.schema_arrow.metadata or {}
        if (
            metadata.get(b"demofml.prediction_set")
            != prediction_set.encode()
            or
            metadata.get(b"demofml.model_set") != config.model_set.encode()
            or metadata.get(b"demofml.validation_set")
            != config.validation_set.encode()
        ):
            return False
        for batch in parquet.iter_batches(
            columns=["decision_time", "entry_time", "exit_time"],
            batch_size=100_000,
            use_threads=False,
        ):
            for column in batch.columns:
                if column.null_count:
                    return False
                values = column.to_numpy(zero_copy_only=False).astype(
                    "datetime64[ns]"
                )
                if values.size and np.max(values) >= locked_start:
                    return False
    return True


def _portfolio_timestamps_are_safe(root: Path, locked_test_start: datetime) -> bool:
    locked_start = _locked_start_ns(locked_test_start)
    files = (
        (
            root / "portfolio" / "ledger.parquet",
            ("decision_time", "entry_time", "exit_time"),
        ),
        (root / "portfolio" / "equity.parquet", ("event_time",)),
    )
    for path, columns in files:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(
            columns=list(columns), batch_size=100_000, use_threads=False
        ):
            for column in batch.columns:
                if column.null_count:
                    return False
                values = column.to_numpy(zero_copy_only=False).astype(
                    "datetime64[ns]"
                )
                if values.size and np.max(values) >= locked_start:
                    return False
    return True


def _portfolio_recomputes(
    root: Path,
    config: AcceptanceConfig,
    report: dict[str, Any],
    locked_test_start: datetime,
) -> bool:
    portfolio_config = load_portfolio_config(config.portfolio_config)
    predictions = (
        pq.read_table(
            _model_artifact(root, symbol, config, "predictions.parquet")
        )
        for symbol in config.symbols
    )
    simulation = simulate_portfolio(
        predictions,
        portfolio_config,
        locked_test_start,
    )
    stored_ledger = pq.read_table(root / "portfolio" / "ledger.parquet")
    stored_equity = pq.read_table(root / "portfolio" / "equity.parquet")
    period_returns = tuple(
        float(value)
        for value in pq.read_table(
            root / "portfolio" / "period-returns.parquet",
            columns=["portfolio_return"],
        )
        .column("portfolio_return")
        .to_pylist()
    )
    return (
        simulation.ledger.equals(stored_ledger)
        and simulation.equity.equals(stored_equity)
        and simulation.period_returns == period_returns
        and portfolio_report(simulation, portfolio_config) == report
    )


def _portfolio_artifact_observations(
    root: Path, locked_test_start: datetime
) -> dict[str, float | bool]:
    locked_start = _locked_start_ns(locked_test_start)
    ledger = pq.ParquetFile(root / "portfolio" / "ledger.parquet")
    ledger_pnl = 0.0
    maximum_risk_leverage = 0.0
    maximum_pnl_error = 0.0
    timestamps_safe = True
    for batch in ledger.iter_batches(
        columns=[
            "notional_usd",
            "realized_return",
            "pnl_usd",
            "risk_leverage",
            "decision_time",
            "entry_time",
            "exit_time",
        ],
        batch_size=100_000,
        use_threads=False,
    ):
        notional = np.asarray(batch.column(0), dtype=float)
        realized = np.asarray(batch.column(1), dtype=float)
        pnl = np.asarray(batch.column(2), dtype=float)
        risk = np.asarray(batch.column(3), dtype=float)
        if not all(
            np.isfinite(values).all()
            for values in (notional, realized, pnl, risk)
        ):
            raise RuntimeError("Portfolio ledger contains non-finite values")
        if (notional < 0.0).any() or (risk < 0.0).any():
            raise RuntimeError("Portfolio ledger contains negative exposure")
        ledger_pnl += float(np.sum(pnl, dtype=np.float64))
        if risk.size:
            maximum_risk_leverage = max(maximum_risk_leverage, float(np.max(risk)))
            maximum_pnl_error = max(
                maximum_pnl_error,
                float(np.max(np.abs(pnl - notional * realized))),
            )
        for column in batch.columns[4:]:
            if column.null_count:
                timestamps_safe = False
                continue
            values = column.to_numpy(zero_copy_only=False).astype("datetime64[ns]")
            if values.size and np.max(values) >= locked_start:
                timestamps_safe = False

    equity = pq.ParquetFile(root / "portfolio" / "equity.parquet")
    first_equity: float | None = None
    last_equity: float | None = None
    maximum_drawdown = 0.0
    maximum_drawdown_error = 0.0
    maximum_peak_error_usd = 0.0
    maximum_observed_gross_leverage = 0.0
    running_peak = float("-inf")
    halted = False
    for batch in equity.iter_batches(
        columns=[
            "event_time",
            "equity_usd",
            "running_peak_usd",
            "drawdown",
            "active_positions",
            "gross_notional_usd",
            "halted",
        ],
        batch_size=100_000,
        use_threads=False,
    ):
        event_time = batch.column(0)
        equity_values = np.asarray(batch.column(1), dtype=float)
        reported_peaks = np.asarray(batch.column(2), dtype=float)
        drawdowns = np.asarray(batch.column(3), dtype=float)
        active_positions = np.asarray(batch.column(4), dtype=np.int64)
        gross_notional = np.asarray(batch.column(5), dtype=float)
        halted_values = np.asarray(batch.column(6), dtype=bool)
        if event_time.null_count or not (
            np.isfinite(equity_values).all()
            and np.isfinite(reported_peaks).all()
            and np.isfinite(drawdowns).all()
            and np.isfinite(gross_notional).all()
        ):
            raise RuntimeError("Portfolio equity contains invalid values")
        if (
            (equity_values <= 0.0).any()
            or (active_positions < 0).any()
            or (gross_notional < 0.0).any()
        ):
            raise RuntimeError("Portfolio equity contains invalid risk state")
        if equity_values.size:
            first_equity = (
                float(equity_values[0]) if first_equity is None else first_equity
            )
            last_equity = float(equity_values[-1])
            for equity_value, reported_peak, reported_drawdown, gross in zip(
                equity_values,
                reported_peaks,
                drawdowns,
                gross_notional,
                strict=True,
            ):
                running_peak = max(running_peak, float(equity_value))
                expected_drawdown = 1.0 - float(equity_value) / running_peak
                maximum_drawdown = max(maximum_drawdown, expected_drawdown)
                maximum_drawdown_error = max(
                    maximum_drawdown_error,
                    abs(float(reported_drawdown) - expected_drawdown),
                )
                maximum_peak_error_usd = max(
                    maximum_peak_error_usd,
                    abs(float(reported_peak) - running_peak),
                )
                maximum_observed_gross_leverage = max(
                    maximum_observed_gross_leverage,
                    float(gross) / float(equity_value),
                )
            halted = halted or bool(halted_values.any())
        event_values = event_time.to_numpy(zero_copy_only=False).astype(
            "datetime64[ns]"
        )
        if event_values.size and np.max(event_values) >= locked_start:
            timestamps_safe = False
    if first_equity is None or last_equity is None:
        raise RuntimeError("Portfolio equity is empty")

    period_returns = pq.ParquetFile(
        root / "portfolio" / "period-returns.parquet"
    )
    return_values: list[np.ndarray[Any, np.dtype[np.float64]]] = []
    expected_index = 0
    for batch in period_returns.iter_batches(
        columns=["period_index", "portfolio_return"],
        batch_size=100_000,
        use_threads=False,
    ):
        indices = np.asarray(batch.column(0), dtype=np.int64)
        values = np.asarray(batch.column(1), dtype=float)
        if not np.isfinite(values).all() or not np.array_equal(
            indices, np.arange(expected_index, expected_index + len(indices))
        ):
            raise RuntimeError("Portfolio period returns are invalid")
        expected_index += len(indices)
        return_values.append(values)
    if expected_index < 2:
        raise RuntimeError("Portfolio period returns are insufficient")
    all_returns = np.concatenate(return_values)
    realized_volatility = float(np.std(all_returns, ddof=1) * math.sqrt(72_576))
    return {
        "first_equity_usd": first_equity,
        "last_equity_usd": last_equity,
        "ledger_pnl_usd": ledger_pnl,
        "maximum_drawdown": maximum_drawdown,
        "maximum_drawdown_error": maximum_drawdown_error,
        "maximum_peak_error_usd": maximum_peak_error_usd,
        "maximum_risk_leverage": maximum_risk_leverage,
        "maximum_observed_gross_leverage": maximum_observed_gross_leverage,
        "maximum_pnl_error_usd": maximum_pnl_error,
        "realized_annual_volatility": realized_volatility,
        "halted": halted,
        "timestamps_safe": timestamps_safe,
    }


def evaluate_development_run(
    run_root: Path, acceptance_config_path: Path
) -> dict[str, Any]:
    """Evaluate a completed development run without reading locked-test data."""
    root = run_root.expanduser().resolve()
    config = load_acceptance_config(acceptance_config_path)
    envelope = _verified_sealed_envelope(config, acceptance_config_path)
    plan = _resolved_validation_plan(config)
    locked_test_start = _locked_test_start(plan)
    run = _read_json(root / "run.json", "Pipeline run record")
    run_id = str(run.get("run_id", ""))
    checks: list[dict[str, object]] = []
    provenance_passed = (
        run.get("format_version") == 1
        and run.get("pipeline_set") == config.pipeline_set
        and run.get("dataset_set") == config.dataset_set
        and run.get("development_only") is True
        and tuple(run.get("symbols", [])) == config.symbols
        and run.get("dataset_file_count") == config.expected_authorized_files
        and run.get("dataset_rows") == config.expected_source_rows
        and run.get("locked_test_start") == locked_test_start.isoformat()
        and run.get("acceptance_set") == config.id
        and root.name == run_id
        and _RUN_ID_PATTERN.fullmatch(run_id) is not None
        and _RUN_ID_PATTERN.fullmatch(str(run.get("dataset_version", "")))
        is not None
        and _CODE_REFERENCE_PATTERN.fullmatch(str(run.get("code_reference", "")))
        is not None
    )
    checks.append(
        _check(
            "contract.provenance",
            provenance_passed,
            bool(provenance_passed),
            "eq",
            True,
        )
    )

    try:
        stage_count = _verify_stages(
            root, run_id, config.symbols, _ACCEPTANCE_MODEL_STAGES[config.id]
        )
        stage_verified = stage_count == config.expected_stage_count
    except (OSError, RuntimeError, ValueError):
        stage_count = 0
        stage_verified = False
    checks.append(
        _check(
            "execution.verified_stages",
            stage_verified,
            stage_count,
            "eq",
            config.expected_stage_count,
        )
    )
    execution = _read_json(root / "execution-report.json", "Execution report")
    execution_stages = execution.get("stages")
    profile_valid = (
        set(execution)
        == {
            "format_version",
            "pipeline_run_id",
            "status",
            "report_scope",
            "attempt_mode",
            "compute_elapsed_ns",
            "process_lifetime_peak_rss_bytes_at_end",
            "stages",
        }
        and
        execution.get("format_version") == 1
        and execution.get("pipeline_run_id") == run_id
        and execution.get("status") == "COMPUTE_SUCCEEDED"
        and execution.get("report_scope")
        == "computational_stages_through_portfolio"
        and _valid_execution_stages(
            execution_stages, config, execution.get("attempt_mode")
        )
        and int(execution.get("compute_elapsed_ns", 0)) > 0
        and int(execution.get("process_lifetime_peak_rss_bytes_at_end", 0)) > 0
    )
    checks.append(
        _check("execution.profile", profile_valid, bool(profile_valid), "eq", True)
    )
    if not profile_valid:
        raise RuntimeError("Execution profile is invalid")

    validation = _read_json(root / "validation" / "manifest.json", "Validation")
    folds = validation.get("folds")
    fold_count = len(folds) if isinstance(folds, list) else 0
    expected_fold_ids = _expected_fold_ids(plan, config.expected_fold_count)
    observed_fold_ids = (
        {str(fold.get("id")) for fold in folds if isinstance(fold, dict)}
        if isinstance(folds, list)
        else set()
    )
    locked_test = validation.get("locked_test")
    expected_locked_start = locked_test_start.isoformat().replace("+00:00", "Z")
    validation_valid = (
        validation.get("id") == config.validation_set
        and validation.get("purge_minutes") == 65
        and validation.get("maximum_information_window_minutes") == 65
        and isinstance(locked_test, dict)
        and locked_test.get("start") == expected_locked_start
        and fold_count == config.expected_fold_count
        and observed_fold_ids == expected_fold_ids
    )
    checks.append(
        _check(
            "validation.fold_count",
            validation_valid,
            fold_count,
            "eq",
            config.expected_fold_count,
        )
    )

    prediction_timestamps_safe = _prediction_timestamps_are_safe(
        root, config, locked_test_start
    )
    if not prediction_timestamps_safe:
        raise RuntimeError("Prediction timestamps reach the locked test")
    checks.append(
        _check(
            "model.development_timestamps",
            True,
            True,
            "eq",
            True,
        )
    )
    (
        horizon_means,
        positive_folds,
        positive_symbols,
        minimum_trades,
        metric_cells,
    ) = _model_observations(
        root,
        config,
        expected_fold_ids,
    )
    checks.extend(
        [
            _check(
                "model.positive_horizon_means",
                all(
                    value
                    > config.minimum_mean_executable_return_bps_exclusive
                    for value in horizon_means.values()
                ),
                {str(key): value for key, value in horizon_means.items()},
                "gt_each",
                config.minimum_mean_executable_return_bps_exclusive,
            ),
            _check(
                "model.positive_folds",
                all(
                    positive_folds[horizon]
                    >= config.minimum_positive_folds_per_horizon
                    for horizon in config.horizons_minutes
                ),
                {str(key): value for key, value in positive_folds.items()},
                "gte_each",
                config.minimum_positive_folds_per_horizon,
            ),
            _check(
                "model.positive_symbols",
                all(
                    positive_symbols[horizon]
                    >= config.minimum_positive_symbols_per_horizon
                    for horizon in config.horizons_minutes
                ),
                {str(key): value for key, value in positive_symbols.items()},
                "gte_each",
                config.minimum_positive_symbols_per_horizon,
            ),
            _check(
                "model.minimum_trades",
                minimum_trades >= config.minimum_trades_per_symbol_horizon,
                minimum_trades,
                "gte",
                config.minimum_trades_per_symbol_horizon,
            ),
            _check(
                "model.metric_cells",
                metric_cells
                == len(config.symbols)
                * len(config.horizons_minutes)
                * (config.expected_fold_count + 1),
                metric_cells,
                "eq",
                len(config.symbols)
                * len(config.horizons_minutes)
                * (config.expected_fold_count + 1),
            ),
        ]
    )
    portfolio_timestamps_safe = _portfolio_timestamps_are_safe(root, locked_test_start)
    if not portfolio_timestamps_safe:
        raise RuntimeError("Portfolio timestamps reach the locked test")
    portfolio = _read_json(root / "portfolio" / "metrics.json", "Portfolio metrics")
    initial = float(portfolio["initial_capital_usd"])
    final = float(portfolio["final_equity_usd"])
    total_return = float(portfolio["total_return"])
    drawdown = float(portfolio["maximum_drawdown"])
    volatility = float(portfolio["realized_annual_volatility"])
    leverage = float(portfolio["maximum_gross_leverage"])
    if not all(
        math.isfinite(value)
        for value in (initial, final, total_return, drawdown, volatility, leverage)
    ):
        raise RuntimeError("Portfolio metrics contain non-finite values")
    if initial <= 0.0:
        raise RuntimeError("Portfolio initial capital must be positive")
    artifacts = _portfolio_artifact_observations(root, locked_test_start)
    portfolio_recomputed = _portfolio_recomputes(
        root, config, portfolio, locked_test_start
    )
    equity_reconciled = (
        math.isclose(
            initial,
            float(artifacts["first_equity_usd"]),
            rel_tol=0.0,
            abs_tol=config.reconciliation_tolerance_usd,
        )
        and math.isclose(
            final,
            float(artifacts["last_equity_usd"]),
            rel_tol=0.0,
            abs_tol=config.reconciliation_tolerance_usd,
        )
        and math.isclose(
            final - initial,
            float(artifacts["ledger_pnl_usd"]),
            rel_tol=0.0,
            abs_tol=config.reconciliation_tolerance_usd,
        )
        and math.isclose(
            total_return,
            final / initial - 1.0,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and math.isclose(
            drawdown,
            float(artifacts["maximum_drawdown"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and float(artifacts["maximum_pnl_error_usd"])
        <= config.reconciliation_tolerance_usd
        and float(artifacts["maximum_peak_error_usd"])
        <= config.reconciliation_tolerance_usd
        and float(artifacts["maximum_drawdown_error"]) <= 1e-12
        and math.isclose(
            volatility,
            float(artifacts["realized_annual_volatility"]),
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        and bool(portfolio.get("drawdown_halt_triggered"))
        == bool(artifacts["halted"])
    )
    checks.extend(
        [
            _check(
                "portfolio.total_return",
                total_return > config.minimum_total_return_exclusive,
                total_return,
                "gt",
                config.minimum_total_return_exclusive,
            ),
            _check(
                "portfolio.maximum_drawdown",
                drawdown < config.maximum_drawdown_exclusive,
                drawdown,
                "lt",
                config.maximum_drawdown_exclusive,
            ),
            _check(
                "portfolio.realized_volatility",
                config.minimum_realized_annual_volatility
                <= volatility
                <= config.maximum_realized_annual_volatility,
                volatility,
                "between_inclusive",
                [
                    config.minimum_realized_annual_volatility,
                    config.maximum_realized_annual_volatility,
                ],
            ),
            _check(
                "portfolio.maximum_gross_leverage",
                leverage <= config.maximum_gross_leverage
                and leverage
                >= float(artifacts["maximum_observed_gross_leverage"])
                and float(artifacts["maximum_risk_leverage"])
                <= config.maximum_gross_leverage,
                {
                    "gross": leverage,
                    "risk": float(artifacts["maximum_risk_leverage"]),
                    "observed_gross": float(
                        artifacts["maximum_observed_gross_leverage"]
                    ),
                },
                "lte",
                config.maximum_gross_leverage,
            ),
            _check(
                "portfolio.drawdown_halt",
                not config.require_no_drawdown_halt
                or portfolio.get("drawdown_halt_triggered") is False,
                bool(portfolio.get("drawdown_halt_triggered")),
                "eq",
                False,
            ),
            _check(
                "portfolio.ledger_equity_reconciliation",
                equity_reconciled,
                bool(equity_reconciled),
                "eq",
                True,
            ),
            _check(
                "portfolio.development_timestamps",
                portfolio_timestamps_safe and bool(artifacts["timestamps_safe"]),
                portfolio_timestamps_safe and bool(artifacts["timestamps_safe"]),
                "eq",
                True,
            ),
            _check(
                "portfolio.full_recomputation",
                portfolio_recomputed,
                portfolio_recomputed,
                "eq",
                True,
            ),
        ]
    )
    attribution = portfolio.get("attribution")
    if not isinstance(attribution, dict):
        raise RuntimeError("Portfolio attribution is invalid")
    pnl = final - initial
    trades = int(portfolio["trades"])
    dimensions = ("symbols", "horizons", "folds")
    reconciled = all(
        math.isclose(
            sum(float(row["pnl_usd"]) for row in attribution[dimension]),
            pnl,
            rel_tol=0.0,
            abs_tol=config.reconciliation_tolerance_usd,
        )
        and sum(int(row["trades"]) for row in attribution[dimension]) == trades
        for dimension in dimensions
    )
    checks.append(
        _check(
            "portfolio.attribution_reconciliation",
            reconciled,
            bool(reconciled),
            "eq",
            True,
        )
    )
    provenance = (
        portfolio.get("format_version") == 1
        and portfolio.get("portfolio_set") == config.portfolio_set
        and portfolio.get("model_set") == config.model_set
        and portfolio.get("validation_set") == config.validation_set
        and portfolio.get("development_only") is True
    )
    checks.append(
        _check(
            "portfolio.provenance", provenance, bool(provenance), "eq", True
        )
    )

    counts = Counter(str(check["status"]) for check in checks)
    accepted = counts["fail"] == 0 and counts["blocked"] == 0
    report: dict[str, Any] = {
        "format_version": 1,
        "acceptance_set": config.id,
        "development_only": True,
        "run_id": run_id,
        "acceptance_config_sha256": _sha256(
            acceptance_config_path.expanduser().resolve()
        ),
        "checks": checks,
        "summary": {
            "pass": counts["pass"],
            "fail": counts["fail"],
            "blocked": counts["blocked"],
            "accepted": accepted,
        },
    }
    if envelope is not None and config.sealed_envelope is not None:
        # Only present for contracts that pre-registered a seal, so acceptance
        # reports for Campaign 1/2 runs keep exactly the shape they had.
        report["sealed_envelope"] = {
            "id": envelope.id,
            "campaign": envelope.campaign,
            "sealed_at": envelope.sealed_at.isoformat(),
            "sha256": _sha256(config.sealed_envelope),
            "documents": {
                document.role: document.sha256 for document in envelope.documents
            },
        }
    return report


def publish_acceptance_report(
    run_root: Path, acceptance_config_path: Path, output: Path
) -> dict[str, Any]:
    """Evaluate and atomically publish one immutable acceptance report."""
    output = output.expanduser().resolve()
    report = evaluate_development_run(run_root, acceptance_config_path)
    if output.exists():
        existing = _read_json(output, "Acceptance report")
        if existing != report:
            raise RuntimeError(f"Acceptance report differs: {output}")
        return report
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    try:
        partial.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.link(partial, output)
    except FileExistsError as error:
        existing = _read_json(output, "Acceptance report")
        if existing != report:
            raise RuntimeError(f"Acceptance report appeared: {output}") from error
    finally:
        partial.unlink(missing_ok=True)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Phase 12 development acceptance command line interface."""
    parser = argparse.ArgumentParser(
        description="Evaluate frozen development-only acceptance criteria."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--acceptance-config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    config = load_acceptance_config(arguments.acceptance_config)
    output = arguments.output or (
        arguments.run_root / "acceptance" / f"{config.id}.json"
    )
    try:
        report = publish_acceptance_report(
            arguments.run_root, arguments.acceptance_config, output
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(f"acceptance report: {output}")
    if not bool(report["summary"]["accepted"]):
        parser.exit(2, "development acceptance rejected\n")

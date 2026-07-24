"""Blind scoring and trusted outcome attachment for the locked test."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from demofml.features.causal import FEATURE_SCHEMA
from demofml.models.baseline import FEATURE_COLUMNS, BaselineConfig
from demofml.models.frozen import (
    FROZEN_MODEL_SET_ID,
    load_frozen_ridge,
    predict_frozen_ridge,
)
from demofml.validation.splits import (
    ValidationPlan,
    select_locked_test_rows,
    validate_feature_label_schemas,
)

LOCKED_FOLD_ID = "locked-test-v1"
LOCKED_SCORE_SET_ID = "locked-test-scores-v1"
LOCKED_PREDICTION_SET_ID = "locked-test-predictions-v1"
_TIMESTAMP = pa.timestamp("ns", tz="UTC")


@dataclass(frozen=True)
class LockedScoringResult:
    """Blind scores and the executable subset joined to hidden outcomes."""

    scores: pa.Table
    evaluated_predictions: pa.Table
    unresolved_executions: int


def locked_score_schema(candidate_id: str, config: BaselineConfig) -> pa.Schema:
    """Build the outcome-free score schema bound to one candidate."""
    return pa.schema(
        [
            pa.field("candidate_id", pa.string(), nullable=False),
            pa.field("model_artifact_set", pa.string(), nullable=False),
            pa.field("model_set", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("decision_time", _TIMESTAMP, nullable=False),
            pa.field("horizon_minutes", pa.int16(), nullable=False),
            pa.field("predicted_long_return", pa.float64(), nullable=False),
            pa.field("predicted_short_return", pa.float64(), nullable=False),
            pa.field("action", pa.string(), nullable=False),
        ],
        metadata={
            b"demofml.score_set": LOCKED_SCORE_SET_ID.encode(),
            b"demofml.candidate_id": candidate_id.encode(),
            b"demofml.model_artifact_set": FROZEN_MODEL_SET_ID.encode(),
            b"demofml.model_set": config.id.encode(),
            b"demofml.feature_set": config.feature_set.encode(),
        },
    )


def locked_prediction_schema(candidate_id: str, config: BaselineConfig) -> pa.Schema:
    """Build trusted executable outcomes for portfolio accounting."""
    return pa.schema(
        [
            pa.field("model_set", pa.string(), nullable=False),
            pa.field("validation_set", pa.string(), nullable=False),
            pa.field("fold_id", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("decision_time", _TIMESTAMP, nullable=False),
            pa.field("entry_time", _TIMESTAMP, nullable=False),
            pa.field("exit_time", _TIMESTAMP, nullable=False),
            pa.field("horizon_minutes", pa.int16(), nullable=False),
            pa.field("predicted_long_return", pa.float64(), nullable=False),
            pa.field("predicted_short_return", pa.float64(), nullable=False),
            pa.field("action", pa.string(), nullable=False),
            pa.field("execution_available", pa.bool_(), nullable=False),
            pa.field("realized_return", pa.float64(), nullable=False),
        ],
        metadata={
            b"demofml.prediction_set": LOCKED_PREDICTION_SET_ID.encode(),
            b"demofml.candidate_id": candidate_id.encode(),
            b"demofml.model_artifact_set": FROZEN_MODEL_SET_ID.encode(),
            b"demofml.model_set": config.id.encode(),
            b"demofml.feature_set": config.feature_set.encode(),
            b"demofml.label_set": config.label_set.encode(),
            b"demofml.validation_set": config.validation_set.encode(),
            b"demofml.horizons_minutes": ",".join(
                str(value) for value in config.horizons_minutes
            ).encode(),
            b"demofml.action_threshold_bps": str(config.action_threshold_bps).encode(),
        },
    )


def _feature_matrix(features: pa.Table, config: BaselineConfig) -> np.ndarray:
    for name in ("symbol", "bar_end", *FEATURE_COLUMNS):
        if name not in features.column_names:
            raise ValueError(f"locked feature schema is missing {name}")
        expected = FEATURE_SCHEMA.field(name)
        actual = features.schema.field(name)
        if actual.type != expected.type or actual.nullable != expected.nullable:
            raise ValueError(f"locked feature field {name} differs")
    columns = []
    for name in config.features:
        values = np.asarray(
            features.column(name).to_numpy(zero_copy_only=False), dtype=float
        )
        if np.isinf(values).any():
            raise ValueError(f"locked feature {name} contains infinity")
        columns.append(values)
    return np.column_stack(columns)


def _action(long_prediction: float, short_prediction: float, threshold: float) -> str:
    if max(long_prediction, short_prediction) <= threshold:
        return "flat"
    return "long" if long_prediction > short_prediction else "short"


def score_locked_features(
    features: pa.Table,
    plan: ValidationPlan,
    config: BaselineConfig,
    candidate_root: Path,
    candidate_id: str,
    symbol: str,
) -> pa.Table:
    """Score every eligible feature key without receiving an outcome table."""
    if features.num_rows == 0:
        raise ValueError("locked features must be non-empty")
    metadata = features.schema.metadata or {}
    if metadata.get(b"demofml.feature_set") != plan.feature_set.encode():
        raise ValueError("locked feature schema does not match the validation plan")
    if set(features.column("symbol").to_pylist()) != {symbol}:
        raise ValueError("locked features must contain exactly the requested symbol")
    decision_times = tuple(features.column("bar_end").to_pylist())
    if any(not isinstance(value, datetime) for value in decision_times):
        raise ValueError("locked decision times cannot be null")
    locked_indices = select_locked_test_rows(decision_times, plan)
    if not locked_indices:
        raise ValueError(f"{symbol} has no eligible locked-test decisions")
    selected = np.asarray(locked_indices, dtype=np.int64)
    matrix = _feature_matrix(features, config)[selected]
    score_rows: list[dict[str, object]] = []
    for horizon in config.horizons_minutes:
        model = load_frozen_ridge(
            candidate_root / "models" / symbol / f"{horizon}m.json"
        )
        if (
            model.symbol != symbol
            or model.horizon_minutes != horizon
            or model.training_end >= plan.development_decision_end
        ):
            raise RuntimeError("frozen model training provenance is unsafe")
        predictions = predict_frozen_ridge(model, matrix)
        for position, index in enumerate(locked_indices):
            decision = decision_times[index]
            if not isinstance(decision, datetime):
                raise ValueError("locked decision times cannot be null")
            predicted_long = float(predictions[position, 0])
            predicted_short = float(predictions[position, 1])
            action = _action(predicted_long, predicted_short, config.action_threshold)
            score_rows.append(
                {
                    "candidate_id": candidate_id,
                    "model_artifact_set": FROZEN_MODEL_SET_ID,
                    "model_set": config.id,
                    "symbol": symbol,
                    "decision_time": decision,
                    "horizon_minutes": horizon,
                    "predicted_long_return": predicted_long,
                    "predicted_short_return": predicted_short,
                    "action": action,
                }
            )
    return pa.Table.from_pylist(
        score_rows, schema=locked_score_schema(candidate_id, config)
    )


def attach_locked_outcomes(
    scores: pa.Table,
    labels: pa.Table,
    plan: ValidationPlan,
    config: BaselineConfig,
    candidate_id: str,
    symbol: str,
) -> LockedScoringResult:
    """Attach hidden executable outcomes only after blind scores are frozen."""
    metadata = scores.schema.metadata or {}
    if (
        metadata.get(b"demofml.score_set") != LOCKED_SCORE_SET_ID.encode()
        or metadata.get(b"demofml.candidate_id") != candidate_id.encode()
        or scores.schema != locked_score_schema(candidate_id, config)
    ):
        raise ValueError("locked score provenance is incompatible")
    validate_feature_label_schemas(FEATURE_SCHEMA, labels.schema, plan)
    label_symbols = labels.column("symbol").to_pylist()
    decision_times = tuple(labels.column("decision_time").to_pylist())
    if set(label_symbols) != {symbol}:
        raise ValueError("locked labels must contain exactly the requested symbol")
    if any(not isinstance(value, datetime) for value in decision_times):
        raise ValueError("locked decision times cannot be null")
    locked_indices = select_locked_test_rows(decision_times, plan)
    expected_keys = {
        (decision_times[index], horizon)
        for index in locked_indices
        for horizon in config.horizons_minutes
    }
    score_by_key: dict[tuple[datetime, int], dict[str, object]] = {}
    for row in scores.to_pylist():
        decision = row["decision_time"]
        horizon = row["horizon_minutes"]
        if not isinstance(decision, datetime) or not isinstance(horizon, int):
            raise ValueError("locked score keys are invalid")
        key = (decision, horizon)
        if key in score_by_key:
            raise ValueError("locked score keys are duplicated")
        score_by_key[key] = row
    if set(score_by_key) != expected_keys:
        raise ValueError("locked score keys do not match eligible labels")

    evaluated_rows: list[dict[str, object]] = []
    unresolved = 0
    entry_times = labels.column("entry_time").to_pylist()
    for horizon in config.horizons_minutes:
        exit_times = labels.column(f"exit_time_{horizon}m").to_pylist()
        long_targets = labels.column(f"long_return_{horizon}m").to_pylist()
        short_targets = labels.column(f"short_return_{horizon}m").to_pylist()
        for index in locked_indices:
            decision = decision_times[index]
            if not isinstance(decision, datetime):
                raise ValueError("locked decision times cannot be null")
            score = score_by_key[(decision, horizon)]
            raw_long = score["predicted_long_return"]
            raw_short = score["predicted_short_return"]
            if (
                not isinstance(raw_long, int | float)
                or not isinstance(raw_short, int | float)
                or not math.isfinite(float(raw_long))
                or not math.isfinite(float(raw_short))
            ):
                raise ValueError("locked score predictions must be finite")
            predicted_long = float(raw_long)
            predicted_short = float(raw_short)
            action = _action(predicted_long, predicted_short, config.action_threshold)
            if score["action"] != action:
                raise ValueError("locked score action differs from frozen predictions")
            entry = entry_times[index]
            exit_time = exit_times[index]
            long_return = long_targets[index]
            short_return = short_targets[index]
            if (
                not isinstance(entry, datetime)
                or not isinstance(exit_time, datetime)
                or not isinstance(long_return, int | float)
                or not isinstance(short_return, int | float)
                or not math.isfinite(float(long_return))
                or not math.isfinite(float(short_return))
            ):
                unresolved += 1
                evaluated_rows.append(
                    {
                        "model_set": config.id,
                        "validation_set": config.validation_set,
                        "fold_id": LOCKED_FOLD_ID,
                        "symbol": symbol,
                        "decision_time": decision,
                        "entry_time": decision,
                        "exit_time": decision + timedelta(minutes=horizon),
                        "horizon_minutes": horizon,
                        "predicted_long_return": predicted_long,
                        "predicted_short_return": predicted_short,
                        "action": "flat",
                        "execution_available": False,
                        "realized_return": 0.0,
                    }
                )
                continue
            if not decision <= entry <= decision + timedelta(minutes=5):
                raise RuntimeError("locked entry time violates executable latency")
            exit_target = decision + timedelta(minutes=horizon)
            exit_deadline = decision + timedelta(minutes=horizon + 5)
            if not exit_target <= exit_time <= exit_deadline:
                raise RuntimeError("locked exit time violates executable latency")
            if exit_time >= plan.locked_test_end_exclusive:
                raise RuntimeError("locked outcome exits outside the test interval")
            realized = (
                float(long_return)
                if action == "long"
                else float(short_return)
                if action == "short"
                else 0.0
            )
            evaluated_rows.append(
                {
                    "model_set": config.id,
                    "validation_set": config.validation_set,
                    "fold_id": LOCKED_FOLD_ID,
                    "symbol": symbol,
                    "decision_time": decision,
                    "entry_time": entry,
                    "exit_time": exit_time,
                    "horizon_minutes": horizon,
                    "predicted_long_return": predicted_long,
                    "predicted_short_return": predicted_short,
                    "action": action,
                    "execution_available": True,
                    "realized_return": realized,
                }
            )
    return LockedScoringResult(
        scores,
        pa.Table.from_pylist(
            evaluated_rows, schema=locked_prediction_schema(candidate_id, config)
        ),
        unresolved,
    )


def score_locked_test(
    features: pa.Table,
    labels: pa.Table,
    plan: ValidationPlan,
    config: BaselineConfig,
    candidate_root: Path,
    candidate_id: str,
    symbol: str,
) -> LockedScoringResult:
    """Convenience wrapper preserving the strict two-phase implementation."""
    if features.num_rows != labels.num_rows:
        raise ValueError("locked feature and label rows must match and be non-empty")
    if not features.column("symbol").equals(labels.column("symbol")):
        raise ValueError("locked feature and label symbols are not aligned")
    if not features.column("bar_end").equals(labels.column("decision_time")):
        raise ValueError("locked feature and label decisions are not aligned")
    scores = score_locked_features(
        features, plan, config, candidate_root, candidate_id, symbol
    )
    return attach_locked_outcomes(scores, labels, plan, config, candidate_id, symbol)

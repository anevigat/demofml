"""Cost-aware metrics for executable walk-forward predictions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean, pstdev
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

from demofml.models.baseline import (
    PREDICTION_SET_ID,
    PREDICTION_SET_V3_ID,
    PREDICTION_SET_V4_ID,
)
from demofml.models.gbm import GBM_PREDICTION_SET_ID
from demofml.models.locked import LOCKED_PREDICTION_SET_ID

EVALUATION_SET_ID = "executable-signal-metrics-v1"
_REQUIRED_COLUMNS = (
    "model_set",
    "validation_set",
    "fold_id",
    "symbol",
    "decision_time",
    "horizon_minutes",
    "predicted_long_return",
    "predicted_short_return",
    "action",
    "realized_return",
)
_ACTIONS = frozenset({"long", "short", "flat"})


def _number(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    return float(value)


def _metrics(rows: Iterable[dict[str, object]]) -> dict[str, int | float | None]:
    observations = list(rows)
    returns = [
        _number(row["realized_return"], "realized_return") for row in observations
    ]
    actions = [str(row["action"]) for row in observations]
    if not returns:
        raise ValueError("cannot evaluate an empty prediction group")
    if any(action not in _ACTIONS for action in actions):
        raise ValueError("prediction group contains an invalid action")
    if not all(math.isfinite(value) for value in returns):
        raise ValueError("prediction group contains non-finite returns")
    traded_returns = [
        value
        for value, action in zip(returns, actions, strict=True)
        if action != "flat"
    ]
    trades = len(traded_returns)
    return {
        "observations": len(observations),
        "trades": trades,
        "trade_rate": trades / len(observations),
        "mean_executable_return_bps": fmean(returns) * 10_000.0,
        "return_stddev_bps": pstdev(returns) * 10_000.0,
        "hit_rate": (
            sum(value > 0.0 for value in traded_returns) / trades if trades else None
        ),
    }


def evaluate_predictions(predictions: pa.Table) -> dict[str, Any]:
    """Evaluate fold and aggregate returns without overlapping-position claims."""
    missing = set(_REQUIRED_COLUMNS).difference(predictions.column_names)
    if missing:
        raise ValueError(f"prediction schema is missing {sorted(missing)}")
    metadata = predictions.schema.metadata or {}
    prediction_set = metadata.get(b"demofml.prediction_set", b"").decode()
    if prediction_set not in {
        PREDICTION_SET_ID,
        PREDICTION_SET_V3_ID,
        PREDICTION_SET_V4_ID,
        GBM_PREDICTION_SET_ID,
    }:
        raise ValueError("prediction metadata is not a development prediction set")
    if predictions.num_rows == 0:
        raise ValueError("cannot evaluate empty predictions")
    if prediction_set == PREDICTION_SET_V3_ID:
        _validate_calibrated_actions(predictions, metadata)
    rows = predictions.select(list(_REQUIRED_COLUMNS)).to_pylist()
    aggregate: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_fold: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        horizon = int(_number(row["horizon_minutes"], "horizon_minutes"))
        fold_id = str(row["fold_id"])
        aggregate[horizon].append(row)
        by_fold[(fold_id, horizon)].append(row)
    return {
        "format_version": 1,
        "evaluation_set": EVALUATION_SET_ID,
        "prediction_set": prediction_set,
        "model_set": metadata.get(b"demofml.model_set", b"").decode(),
        "validation_set": metadata.get(b"demofml.validation_set", b"").decode(),
        "aggregate": [
            {"horizon_minutes": horizon, **_metrics(aggregate[horizon])}
            for horizon in sorted(aggregate)
        ],
        "folds": [
            {
                "fold_id": fold_id,
                "horizon_minutes": horizon,
                **_metrics(by_fold[(fold_id, horizon)]),
            }
            for fold_id, horizon in sorted(by_fold)
        ],
        "always_flat_comparator": {
            "mean_executable_return_bps": 0.0,
            "trade_rate": 0.0,
        },
        "interpretation": "development_only_no_overlapping_position_accounting",
    }


def _validate_calibrated_actions(
    predictions: pa.Table, metadata: dict[bytes, bytes]
) -> None:
    fields = (
        "calibrated_selected_return",
        "calibration_intercept",
        "calibration_slope",
        "calibration_rows",
    )
    missing = set(fields).difference(predictions.column_names)
    if missing:
        raise ValueError(f"calibrated prediction schema is missing {sorted(missing)}")
    try:
        threshold = float(metadata[b"demofml.action_threshold_bps"]) / 10_000.0
        minimum_rows = int(metadata[b"demofml.minimum_calibration_rows"])
    except (KeyError, ValueError) as error:
        raise ValueError("calibrated prediction metadata is invalid") from error
    rows = predictions.select(
        [
            "fold_id",
            "symbol",
            "horizon_minutes",
            "predicted_long_return",
            "predicted_short_return",
            *fields,
            "action",
        ]
    ).to_pylist()
    calibrators: dict[tuple[str, str, int], tuple[float, float, int]] = {}
    for row in rows:
        predicted_long = _number(
            row["predicted_long_return"], "predicted_long_return"
        )
        predicted_short = _number(
            row["predicted_short_return"], "predicted_short_return"
        )
        calibrated = _number(
            row["calibrated_selected_return"], "calibrated_selected_return"
        )
        intercept = _number(row["calibration_intercept"], "calibration_intercept")
        slope = _number(row["calibration_slope"], "calibration_slope")
        calibration_rows = int(_number(row["calibration_rows"], "calibration_rows"))
        if not all(
            math.isfinite(value)
            for value in (predicted_long, predicted_short, calibrated, intercept, slope)
        ):
            raise ValueError("calibrated predictions must be finite")
        if slope < 0.0 or calibration_rows < minimum_rows:
            raise ValueError("calibration contract is invalid")
        expected_calibrated = intercept + slope * max(
            predicted_long, predicted_short
        )
        if not math.isclose(calibrated, expected_calibrated, abs_tol=1e-15):
            raise ValueError("calibrated return cannot be reproduced")
        expected_action = (
            "flat"
            if calibrated <= threshold
            else "long"
            if predicted_long > predicted_short
            else "short"
        )
        if row["action"] != expected_action:
            raise ValueError("calibrated action cannot be reproduced")
        key = (str(row["fold_id"]), str(row["symbol"]), int(row["horizon_minutes"]))
        calibrator = (intercept, slope, calibration_rows)
        if key in calibrators and calibrators[key] != calibrator:
            raise ValueError("calibrator changes within a prediction cell")
        calibrators[key] = calibrator


def evaluate_locked_predictions(
    predictions: pa.Table, candidate_id: str
) -> dict[str, Any]:
    """Evaluate trusted locked outcomes without accepting caller-supplied provenance."""
    missing = set(_REQUIRED_COLUMNS).difference(predictions.column_names)
    if missing:
        raise ValueError(f"prediction schema is missing {sorted(missing)}")
    metadata = predictions.schema.metadata or {}
    if metadata.get(b"demofml.prediction_set") != LOCKED_PREDICTION_SET_ID.encode():
        raise ValueError("prediction metadata is not the locked prediction set")
    if metadata.get(b"demofml.candidate_id") != candidate_id.encode():
        raise ValueError("prediction candidate identity differs")
    if predictions.num_rows == 0:
        raise ValueError("cannot evaluate empty locked predictions")
    rows = predictions.select(list(_REQUIRED_COLUMNS)).to_pylist()
    aggregate: dict[int, list[dict[str, object]]] = defaultdict(list)
    by_symbol: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        horizon = int(_number(row["horizon_minutes"], "horizon_minutes"))
        symbol = str(row["symbol"])
        aggregate[horizon].append(row)
        by_symbol[(symbol, horizon)].append(row)
    return {
        "format_version": 1,
        "evaluation_set": "locked-test-signal-metrics-v1",
        "prediction_set": LOCKED_PREDICTION_SET_ID,
        "candidate_id": candidate_id,
        "model_set": metadata.get(b"demofml.model_set", b"").decode(),
        "validation_set": metadata.get(b"demofml.validation_set", b"").decode(),
        "development_only": False,
        "locked_test": True,
        "aggregate": [
            {"horizon_minutes": horizon, **_metrics(aggregate[horizon])}
            for horizon in sorted(aggregate)
        ],
        "symbols": [
            {
                "symbol": symbol,
                "horizon_minutes": horizon,
                **_metrics(by_symbol[(symbol, horizon)]),
            }
            for symbol, horizon in sorted(by_symbol)
        ],
        "always_flat_comparator": {
            "mean_executable_return_bps": 0.0,
            "trade_rate": 0.0,
        },
        "interpretation": "one_shot_locked_test_executable_outcomes",
    }

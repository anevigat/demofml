"""Cost-aware metrics for executable walk-forward predictions."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from statistics import fmean, pstdev
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]

from demofml.models.baseline import PREDICTION_SET_ID

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
    if metadata.get(b"demofml.prediction_set") != PREDICTION_SET_ID.encode():
        raise ValueError(f"prediction metadata is not {PREDICTION_SET_ID}")
    if predictions.num_rows == 0:
        raise ValueError("cannot evaluate empty predictions")
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
        "prediction_set": PREDICTION_SET_ID,
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

"""Deterministic ridge baseline trained inside purged temporal folds."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from demofml.features.causal import FEATURE_SCHEMA, FEATURE_SET_ID
from demofml.labels.executable import LABEL_SET_ID
from demofml.validation.splits import (
    VALIDATION_SET_ID,
    ValidationPlan,
    select_fold_rows,
    validate_feature_label_schemas,
)

MODEL_SET_ID = "baseline-ridge-v1"
PREDICTION_SET_ID = "walk-forward-predictions-v1"
FEATURE_COLUMNS = tuple(FEATURE_SCHEMA.names[2:])


@dataclass(frozen=True)
class BaselineConfig:
    """Immutable behavior of the development ridge baseline."""

    id: str
    feature_set: str
    label_set: str
    validation_set: str
    horizons_minutes: tuple[int, ...]
    training_scope: str
    model_type: str
    alpha: float
    solver: str
    imputation: str
    standardize: bool
    action_threshold_bps: float
    minimum_training_rows: int
    random_seed: int
    locked_test_policy: str
    features: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.id != MODEL_SET_ID:
            raise ValueError(f"model id must be {MODEL_SET_ID}")
        if self.feature_set != FEATURE_SET_ID or self.label_set != LABEL_SET_ID:
            raise ValueError("baseline feature and label sets are incompatible")
        if self.validation_set != VALIDATION_SET_ID:
            raise ValueError("baseline requires purged-walk-forward-v1")
        if (
            not self.horizons_minutes
            or tuple(sorted(set(self.horizons_minutes))) != self.horizons_minutes
        ):
            raise ValueError("horizons must be unique and increasing")
        if self.training_scope != "per_symbol":
            raise ValueError("baseline training_scope must be per_symbol")
        if self.model_type != "ridge" or self.solver != "lsqr":
            raise ValueError("baseline model must be ridge with the lsqr solver")
        if not math.isfinite(self.alpha) or self.alpha <= 0.0:
            raise ValueError("alpha must be finite and positive")
        if self.imputation != "training_median" or not self.standardize:
            raise ValueError("baseline requires training median and standardization")
        if not math.isfinite(self.action_threshold_bps):
            raise ValueError("action_threshold_bps must be finite")
        if self.minimum_training_rows < 2:
            raise ValueError("minimum_training_rows must be at least two")
        if self.random_seed < 0:
            raise ValueError("random_seed cannot be negative")
        if self.locked_test_policy != "forbidden":
            raise ValueError("locked test policy must remain forbidden")
        if self.features != FEATURE_COLUMNS:
            raise ValueError("baseline features do not match causal-v1")

    @property
    def action_threshold(self) -> float:
        return self.action_threshold_bps / 10_000.0


@dataclass(frozen=True)
class AlignedResearchData:
    """One symbol's aligned feature matrix and executable targets."""

    symbol: str
    decision_times: tuple[datetime, ...]
    features: NDArray[np.float64]
    long_targets: dict[int, NDArray[np.float64]]
    short_targets: dict[int, NDArray[np.float64]]


def load_baseline_config(path: Path) -> BaselineConfig:
    """Load and strictly validate the versioned baseline definition."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Baseline config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    try:
        return BaselineConfig(
            id=str(values["id"]),
            feature_set=str(values["feature_set"]),
            label_set=str(values["label_set"]),
            validation_set=str(values["validation_set"]),
            horizons_minutes=tuple(int(value) for value in values["horizons_minutes"]),
            training_scope=str(values["training_scope"]),
            model_type=str(values["model_type"]),
            alpha=float(values["alpha"]),
            solver=str(values["solver"]),
            imputation=str(values["imputation"]),
            standardize=bool(values["standardize"]),
            action_threshold_bps=float(values["action_threshold_bps"]),
            minimum_training_rows=int(values["minimum_training_rows"]),
            random_seed=int(values["random_seed"]),
            locked_test_policy=str(values["locked_test_policy"]),
            features=tuple(str(value) for value in values["features"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid baseline config field: {error}") from error


def prediction_schema(config: BaselineConfig) -> pa.Schema:
    """Build the prediction schema with complete model provenance."""
    return pa.schema(
        [
            pa.field("model_set", pa.string(), nullable=False),
            pa.field("validation_set", pa.string(), nullable=False),
            pa.field("fold_id", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("decision_time", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("horizon_minutes", pa.int16(), nullable=False),
            pa.field("predicted_long_return", pa.float64(), nullable=False),
            pa.field("predicted_short_return", pa.float64(), nullable=False),
            pa.field("action", pa.string(), nullable=False),
            pa.field("realized_return", pa.float64(), nullable=False),
        ],
        metadata={
            b"demofml.prediction_set": PREDICTION_SET_ID.encode(),
            b"demofml.model_set": config.id.encode(),
            b"demofml.feature_set": config.feature_set.encode(),
            b"demofml.label_set": config.label_set.encode(),
            b"demofml.validation_set": config.validation_set.encode(),
            b"demofml.horizons_minutes": ",".join(
                str(value) for value in config.horizons_minutes
            ).encode(),
            b"demofml.action_threshold_bps": str(config.action_threshold_bps).encode(),
            b"demofml.random_seed": str(config.random_seed).encode(),
        },
    )


def _validate_feature_fields(schema: pa.Schema) -> None:
    for name in ("symbol", "bar_end", *FEATURE_COLUMNS):
        if name not in schema.names:
            raise ValueError(f"feature schema is missing {name}")
        actual = schema.field(name)
        expected = FEATURE_SCHEMA.field(name)
        if actual.type != expected.type or actual.nullable != expected.nullable:
            raise ValueError(f"feature field {name} does not match causal-v1")


def _float_column(table: pa.Table, name: str) -> NDArray[np.float64]:
    values = np.asarray(table.column(name).to_numpy(zero_copy_only=False), dtype=float)
    if np.isinf(values).any():
        raise ValueError(f"{name} contains infinite values")
    return values


def align_research_tables(
    features: pa.Table,
    labels: pa.Table,
    plan: ValidationPlan,
    config: BaselineConfig,
) -> AlignedResearchData:
    """Align exact feature/label keys and reject locked-test observations."""
    if config.feature_set != plan.feature_set or config.label_set != plan.label_set:
        raise ValueError("model and validation data contracts differ")
    if max(config.horizons_minutes) != plan.max_horizon_minutes:
        raise ValueError("model horizons and validation purge differ")
    validate_feature_label_schemas(features.schema, labels.schema, plan)
    _validate_feature_fields(features.schema)
    required_labels = {
        "symbol",
        "decision_time",
        *(
            f"{side}_return_{horizon}m"
            for horizon in config.horizons_minutes
            for side in ("long", "short")
        ),
    }
    missing_labels = required_labels.difference(labels.column_names)
    if missing_labels:
        raise ValueError(f"label schema is missing {sorted(missing_labels)}")
    if features.num_rows == 0 or features.num_rows != labels.num_rows:
        raise ValueError("feature and label row counts must match and be non-zero")
    if not features.column("symbol").equals(labels.column("symbol")):
        raise ValueError("feature and label symbols are not aligned")
    if not features.column("bar_end").equals(labels.column("decision_time")):
        raise ValueError("feature and label decision times are not aligned")

    symbols = set(features.column("symbol").to_pylist())
    if len(symbols) != 1:
        raise ValueError("baseline inputs must contain exactly one symbol")
    symbol_value = next(iter(symbols))
    if not isinstance(symbol_value, str) or not symbol_value:
        raise ValueError("baseline symbol must be a non-empty string")
    decision_times = tuple(features.column("bar_end").to_pylist())
    previous: datetime | None = None
    for decision_time in decision_times:
        if not isinstance(decision_time, datetime):
            raise ValueError("decision_time cannot be null")
        if previous is not None and decision_time <= previous:
            raise ValueError("research rows must be strictly ordered")
        if decision_time >= plan.locked_test_start:
            raise ValueError("locked-test rows are forbidden during development")
        previous = decision_time

    matrix = np.column_stack(
        [_float_column(features, name) for name in config.features]
    )
    long_targets = {
        horizon: _float_column(labels, f"long_return_{horizon}m")
        for horizon in config.horizons_minutes
    }
    short_targets = {
        horizon: _float_column(labels, f"short_return_{horizon}m")
        for horizon in config.horizons_minutes
    }
    return AlignedResearchData(
        symbol_value,
        decision_times,
        matrix,
        long_targets,
        short_targets,
    )


def _fit_predict(
    training_features: NDArray[np.float64],
    training_targets: NDArray[np.float64],
    validation_features: NDArray[np.float64],
    config: BaselineConfig,
) -> NDArray[np.float64]:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=config.alpha, solver="lsqr"),
    )
    model.fit(training_features, training_targets)
    return np.asarray(model.predict(validation_features), dtype=float)


def _action(long_prediction: float, short_prediction: float, threshold: float) -> str:
    if max(long_prediction, short_prediction) <= threshold:
        return "flat"
    return "long" if long_prediction > short_prediction else "short"


def run_walk_forward(
    features: pa.Table,
    labels: pa.Table,
    plan: ValidationPlan,
    config: BaselineConfig,
) -> pa.Table:
    """Train and score every development fold without touching the lock."""
    data = align_research_tables(features, labels, plan, config)
    rows: list[dict[str, object]] = []
    for fold in plan.folds():
        selection = select_fold_rows(data.decision_times, fold)
        if not selection.validation:
            raise ValueError(f"fold {fold.id} has no validation rows")
        training_indices = np.asarray(selection.train, dtype=np.int64)
        validation_indices = np.asarray(selection.validation, dtype=np.int64)
        for horizon in config.horizons_minutes:
            long_targets = data.long_targets[horizon]
            short_targets = data.short_targets[horizon]
            training_target_mask = np.isfinite(
                long_targets[training_indices]
            ) & np.isfinite(short_targets[training_indices])
            usable_training = training_indices[training_target_mask]
            if usable_training.size < config.minimum_training_rows:
                raise ValueError(
                    f"fold {fold.id} horizon {horizon} has insufficient training rows"
                )
            validation_target_mask = np.isfinite(
                long_targets[validation_indices]
            ) & np.isfinite(short_targets[validation_indices])
            usable_validation = validation_indices[validation_target_mask]
            if usable_validation.size == 0:
                raise ValueError(
                    f"fold {fold.id} horizon {horizon} has no resolved "
                    "validation labels"
                )
            training_targets = np.column_stack(
                [
                    long_targets[usable_training],
                    short_targets[usable_training],
                ]
            )
            predictions = _fit_predict(
                data.features[usable_training],
                training_targets,
                data.features[usable_validation],
                config,
            )
            for row_index, prediction in zip(
                usable_validation, predictions, strict=True
            ):
                predicted_long = float(prediction[0])
                predicted_short = float(prediction[1])
                if not math.isfinite(predicted_long) or not math.isfinite(
                    predicted_short
                ):
                    raise RuntimeError("ridge produced a non-finite prediction")
                action = _action(
                    predicted_long, predicted_short, config.action_threshold
                )
                realized = (
                    float(long_targets[row_index])
                    if action == "long"
                    else float(short_targets[row_index])
                    if action == "short"
                    else 0.0
                )
                rows.append(
                    {
                        "model_set": config.id,
                        "validation_set": plan.id,
                        "fold_id": fold.id,
                        "symbol": data.symbol,
                        "decision_time": data.decision_times[row_index],
                        "horizon_minutes": horizon,
                        "predicted_long_return": predicted_long,
                        "predicted_short_return": predicted_short,
                        "action": action,
                        "realized_return": realized,
                    }
                )
    return pa.Table.from_pylist(rows, schema=prediction_schema(config))

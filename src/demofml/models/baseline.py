"""Deterministic ridge baseline trained inside purged temporal folds."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import Pipeline, make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from demofml.features.causal import FEATURE_SCHEMA, FEATURE_SET_ID
from demofml.features.causal_v2 import (
    FEATURE_SET_V2_ID,
    FEATURE_V2_COLUMNS,
    FEATURE_V2_SCHEMA,
)
from demofml.labels.executable import (
    LABEL_SET_ID,
    LABEL_SET_V2_ID,
    label_schema,
)
from demofml.validation.splits import (
    SCREEN_VALIDATION_SET_ID,
    VALIDATION_SET_ID,
    ValidationPlan,
    select_fold_rows,
    validate_feature_label_schemas,
)

MODEL_SET_ID = "baseline-ridge-v1"
PREDICTION_SET_ID = "walk-forward-predictions-v2"
MODEL_SET_V2_ID = "baseline-ridge-v2"
PREDICTION_SET_V3_ID = "walk-forward-predictions-v3"
MODEL_SET_V3_ID = "baseline-ridge-v3"
PREDICTION_SET_V4_ID = "walk-forward-predictions-v4"
FEATURE_COLUMNS = tuple(FEATURE_SCHEMA.names[2:])
_MODEL_CONTRACTS = {
    MODEL_SET_ID: (
        FEATURE_SET_ID,
        LABEL_SET_ID,
        VALIDATION_SET_ID,
        PREDICTION_SET_ID,
    ),
    MODEL_SET_V2_ID: (
        FEATURE_SET_ID,
        LABEL_SET_ID,
        VALIDATION_SET_ID,
        PREDICTION_SET_V3_ID,
    ),
    MODEL_SET_V3_ID: (
        FEATURE_SET_V2_ID,
        LABEL_SET_V2_ID,
        SCREEN_VALIDATION_SET_ID,
        PREDICTION_SET_V4_ID,
    ),
}
_FEATURE_CONTRACTS = {
    FEATURE_SET_ID: (FEATURE_SCHEMA, FEATURE_COLUMNS),
    FEATURE_SET_V2_ID: (FEATURE_V2_SCHEMA, FEATURE_V2_COLUMNS),
}


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
    selection_policy: str = "static-threshold-v1"
    calibration_window_months: int = 0
    calibration_purge_minutes: int = 0
    minimum_calibration_rows: int = 0
    calibration_regression: str = "none"

    def __post_init__(self) -> None:
        if self.id not in _MODEL_CONTRACTS:
            raise ValueError("model id is not supported")
        feature_set, label_set, validation_set, _prediction_set = _MODEL_CONTRACTS[
            self.id
        ]
        if self.validation_set != validation_set:
            if self.id in {MODEL_SET_ID, MODEL_SET_V2_ID}:
                raise ValueError("baseline requires purged-walk-forward-v1")
            raise ValueError("baseline validation contract is incompatible")
        if self.feature_set != feature_set or self.label_set != label_set:
            raise ValueError("baseline data contracts are incompatible")
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
        if self.features != _FEATURE_CONTRACTS[self.feature_set][1]:
            raise ValueError(
                f"baseline features do not match {self.feature_set}"
            )
        expected_selection = (
            "purged-tail-monotone-affine-v1"
            if self.id == MODEL_SET_V2_ID
            else "static-threshold-v1"
        )
        if self.selection_policy != expected_selection:
            raise ValueError("baseline selection policy does not match model id")
        if expected_selection == "static-threshold-v1":
            if (
                self.calibration_window_months != 0
                or self.calibration_purge_minutes != 0
                or self.minimum_calibration_rows != 0
                or self.calibration_regression != "none"
            ):
                raise ValueError("static ridge cannot define calibration")
        else:
            if (
                self.calibration_window_months != 1
                or self.calibration_purge_minutes != 65
                or self.minimum_calibration_rows < 2
                or self.calibration_regression != "ols_nonnegative_slope"
            ):
                raise ValueError(
                    "baseline-ridge-v2 calibration contract is incompatible"
                )

    @property
    def action_threshold(self) -> float:
        return self.action_threshold_bps / 10_000.0

    @property
    def prediction_set(self) -> str:
        """Return the prediction contract emitted by this model version."""
        return _MODEL_CONTRACTS[self.id][3]


@dataclass(frozen=True)
class AlignedResearchData:
    """One symbol's aligned feature matrix and executable targets."""

    symbol: str
    decision_times: tuple[datetime, ...]
    entry_times: tuple[datetime | None, ...]
    exit_times: dict[int, tuple[datetime | None, ...]]
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
            selection_policy=str(
                values.get("selection_policy", "static-threshold-v1")
            ),
            calibration_window_months=int(
                values.get("calibration_window_months", 0)
            ),
            calibration_purge_minutes=int(
                values.get("calibration_purge_minutes", 0)
            ),
            minimum_calibration_rows=int(values.get("minimum_calibration_rows", 0)),
            calibration_regression=str(values.get("calibration_regression", "none")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid baseline config field: {error}") from error


def prediction_schema(config: BaselineConfig) -> pa.Schema:
    """Build the prediction schema with complete model provenance."""
    calibration_fields = (
        [
            pa.field("calibrated_selected_return", pa.float64(), nullable=False),
            pa.field("calibration_intercept", pa.float64(), nullable=False),
            pa.field("calibration_slope", pa.float64(), nullable=False),
            pa.field("calibration_rows", pa.int32(), nullable=False),
        ]
        if config.selection_policy == "purged-tail-monotone-affine-v1"
        else []
    )
    return pa.schema(
        [
            pa.field("model_set", pa.string(), nullable=False),
            pa.field("validation_set", pa.string(), nullable=False),
            pa.field("fold_id", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("decision_time", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("entry_time", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("exit_time", pa.timestamp("ns", tz="UTC"), nullable=False),
            pa.field("horizon_minutes", pa.int16(), nullable=False),
            pa.field("predicted_long_return", pa.float64(), nullable=False),
            pa.field("predicted_short_return", pa.float64(), nullable=False),
            *calibration_fields,
            pa.field("action", pa.string(), nullable=False),
            pa.field("realized_return", pa.float64(), nullable=False),
        ],
        metadata={
            b"demofml.prediction_set": config.prediction_set.encode(),
            b"demofml.model_set": config.id.encode(),
            b"demofml.feature_set": config.feature_set.encode(),
            b"demofml.label_set": config.label_set.encode(),
            b"demofml.validation_set": config.validation_set.encode(),
            b"demofml.horizons_minutes": ",".join(
                str(value) for value in config.horizons_minutes
            ).encode(),
            b"demofml.action_threshold_bps": str(config.action_threshold_bps).encode(),
            b"demofml.random_seed": str(config.random_seed).encode(),
            b"demofml.selection_policy": config.selection_policy.encode(),
            b"demofml.calibration_window_months": str(
                config.calibration_window_months
            ).encode(),
            b"demofml.calibration_purge_minutes": str(
                config.calibration_purge_minutes
            ).encode(),
            b"demofml.minimum_calibration_rows": str(
                config.minimum_calibration_rows
            ).encode(),
            b"demofml.calibration_regression": config.calibration_regression.encode(),
        },
    )


def _validate_feature_fields(schema: pa.Schema, config: BaselineConfig) -> None:
    expected_schema, expected_columns = _FEATURE_CONTRACTS[config.feature_set]
    if config.feature_set == FEATURE_SET_V2_ID and not schema.equals(
        expected_schema, check_metadata=True
    ):
        raise ValueError("feature schema does not exactly match causal-v2")
    for name in ("symbol", "bar_end", *expected_columns):
        if name not in schema.names:
            raise ValueError(f"feature schema is missing {name}")
        actual = schema.field(name)
        expected = expected_schema.field(name)
        if actual.type != expected.type or actual.nullable != expected.nullable:
            raise ValueError(
                f"feature field {name} does not match {config.feature_set}"
            )


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
    _validate_feature_fields(features.schema, config)
    required_labels = {
        "symbol",
        "decision_time",
        "entry_time",
        *(
            f"{field}_{horizon}m"
            for horizon in config.horizons_minutes
            for field in ("exit_time", "long_return", "short_return")
        ),
    }
    missing_labels = required_labels.difference(labels.column_names)
    if missing_labels:
        raise ValueError(f"label schema is missing {sorted(missing_labels)}")
    expected_labels = label_schema(
        config.horizons_minutes,
        0.0,
        label_set=config.label_set,
    )
    if not labels.schema.equals(expected_labels, check_metadata=True):
        raise ValueError(
            f"label schema does not exactly match {config.label_set}"
        )
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
        if decision_time > plan.development_end_exclusive:
            raise ValueError("rows beyond the research cutoff are forbidden")
        previous = decision_time

    matrix = np.column_stack(
        [_float_column(features, name) for name in config.features]
    )
    entry_times = tuple(labels.column("entry_time").to_pylist())
    exit_times = {
        horizon: tuple(labels.column(f"exit_time_{horizon}m").to_pylist())
        for horizon in config.horizons_minutes
    }
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
        entry_times,
        exit_times,
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


def _fit_ridge(
    training_features: NDArray[np.float64],
    training_targets: NDArray[np.float64],
    config: BaselineConfig,
) -> Pipeline:
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=config.alpha, solver="lsqr"),
    )
    model.fit(training_features, training_targets)
    return model


def _predict(
    model: Pipeline, features: NDArray[np.float64]
) -> NDArray[np.float64]:
    predictions = np.asarray(model.predict(features), dtype=float)
    if predictions.ndim != 2 or predictions.shape[1] != 2:
        raise RuntimeError("ridge produced an invalid prediction matrix")
    if not np.isfinite(predictions).all():
        raise RuntimeError("ridge produced a non-finite prediction")
    return predictions


def _fit_monotone_affine_calibrator(
    scores: NDArray[np.float64], outcomes: NDArray[np.float64]
) -> tuple[float, float]:
    """Fit deterministic OLS while forbidding a negative score slope."""
    if scores.ndim != 1 or outcomes.ndim != 1 or scores.shape != outcomes.shape:
        raise ValueError("calibration scores and outcomes must be aligned vectors")
    if (
        scores.size == 0
        or not np.isfinite(scores).all()
        or not np.isfinite(outcomes).all()
    ):
        raise ValueError("calibration values must be non-empty and finite")
    score_mean = float(np.mean(scores))
    outcome_mean = float(np.mean(outcomes))
    centered_scores = scores - score_mean
    denominator = float(centered_scores @ centered_scores)
    slope = (
        0.0
        if denominator == 0.0
        else max(
            float(centered_scores @ (outcomes - outcome_mean)) / denominator,
            0.0,
        )
    )
    intercept = outcome_mean - slope * score_mean
    if not math.isfinite(intercept) or not math.isfinite(slope):
        raise RuntimeError("calibration produced non-finite coefficients")
    return intercept, slope


def _previous_month_start(value: datetime) -> datetime:
    if value.month == 1:
        return value.replace(year=value.year - 1, month=12)
    return value.replace(month=value.month - 1)


def _resolved_indices(
    indices: NDArray[np.int64],
    long_targets: NDArray[np.float64],
    short_targets: NDArray[np.float64],
) -> NDArray[np.int64]:
    mask = np.isfinite(long_targets[indices]) & np.isfinite(short_targets[indices])
    return indices[mask]


def _action(long_prediction: float, short_prediction: float, threshold: float) -> str:
    if max(long_prediction, short_prediction) <= threshold:
        return "flat"
    return "long" if long_prediction > short_prediction else "short"


def _run_walk_forward_v1(
    data: AlignedResearchData,
    plan: ValidationPlan,
    config: BaselineConfig,
) -> pa.Table:
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
                entry_time = data.entry_times[row_index]
                exit_time = data.exit_times[horizon][row_index]
                decision_time = data.decision_times[row_index]
                if (
                    not isinstance(entry_time, datetime)
                    or not isinstance(exit_time, datetime)
                    or not decision_time <= entry_time < exit_time
                ):
                    raise RuntimeError("resolved label execution times are invalid")
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
                        "decision_time": decision_time,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "horizon_minutes": horizon,
                        "predicted_long_return": predicted_long,
                        "predicted_short_return": predicted_short,
                        "action": action,
                        "realized_return": realized,
                    }
                )
    return pa.Table.from_pylist(rows, schema=prediction_schema(config))


def _run_walk_forward_v2(
    data: AlignedResearchData,
    plan: ValidationPlan,
    config: BaselineConfig,
) -> pa.Table:
    rows: list[dict[str, object]] = []
    for fold in plan.folds():
        selection = select_fold_rows(data.decision_times, fold)
        if not selection.validation:
            raise ValueError(f"fold {fold.id} has no validation rows")
        calibration_start = _previous_month_start(fold.validation_start)
        fit_end = calibration_start - timedelta(
            minutes=config.calibration_purge_minutes
        )
        fit_candidates = np.asarray(
            [
                index
                for index in selection.train
                if data.decision_times[index] < fit_end
            ],
            dtype=np.int64,
        )
        calibration_candidates = np.asarray(
            [
                index
                for index in selection.train
                if data.decision_times[index] >= calibration_start
            ],
            dtype=np.int64,
        )
        validation_indices = np.asarray(selection.validation, dtype=np.int64)
        for horizon in config.horizons_minutes:
            long_targets = data.long_targets[horizon]
            short_targets = data.short_targets[horizon]

            usable_fit = _resolved_indices(
                fit_candidates, long_targets, short_targets
            )
            usable_calibration = _resolved_indices(
                calibration_candidates, long_targets, short_targets
            )
            usable_validation = _resolved_indices(
                validation_indices, long_targets, short_targets
            )
            if usable_fit.size < config.minimum_training_rows:
                raise ValueError(
                    f"fold {fold.id} horizon {horizon} has insufficient fit rows"
                )
            if usable_calibration.size < config.minimum_calibration_rows:
                raise ValueError(
                    f"fold {fold.id} horizon {horizon} has insufficient "
                    "calibration rows"
                )
            if usable_validation.size == 0:
                raise ValueError(
                    f"fold {fold.id} horizon {horizon} has no resolved "
                    "validation labels"
                )
            fit_targets = np.column_stack(
                [long_targets[usable_fit], short_targets[usable_fit]]
            )
            model = _fit_ridge(data.features[usable_fit], fit_targets, config)
            calibration_predictions = _predict(
                model, data.features[usable_calibration]
            )
            calibration_long = (
                calibration_predictions[:, 0] > calibration_predictions[:, 1]
            )
            calibration_scores = np.max(calibration_predictions, axis=1)
            calibration_outcomes = np.where(
                calibration_long,
                long_targets[usable_calibration],
                short_targets[usable_calibration],
            )
            intercept, slope = _fit_monotone_affine_calibrator(
                calibration_scores, calibration_outcomes
            )
            predictions = _predict(model, data.features[usable_validation])
            for row_index, prediction in zip(
                usable_validation, predictions, strict=True
            ):
                predicted_long = float(prediction[0])
                predicted_short = float(prediction[1])
                calibrated = intercept + slope * max(
                    predicted_long, predicted_short
                )
                action = (
                    "flat"
                    if calibrated <= config.action_threshold
                    else "long"
                    if predicted_long > predicted_short
                    else "short"
                )
                entry_time = data.entry_times[row_index]
                exit_time = data.exit_times[horizon][row_index]
                decision_time = data.decision_times[row_index]
                if (
                    not isinstance(entry_time, datetime)
                    or not isinstance(exit_time, datetime)
                    or not decision_time <= entry_time < exit_time
                ):
                    raise RuntimeError("resolved label execution times are invalid")
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
                        "decision_time": decision_time,
                        "entry_time": entry_time,
                        "exit_time": exit_time,
                        "horizon_minutes": horizon,
                        "predicted_long_return": predicted_long,
                        "predicted_short_return": predicted_short,
                        "calibrated_selected_return": calibrated,
                        "calibration_intercept": intercept,
                        "calibration_slope": slope,
                        "calibration_rows": int(usable_calibration.size),
                        "action": action,
                        "realized_return": realized,
                    }
                )
    return pa.Table.from_pylist(rows, schema=prediction_schema(config))


def run_walk_forward(
    features: pa.Table,
    labels: pa.Table,
    plan: ValidationPlan,
    config: BaselineConfig,
) -> pa.Table:
    """Train and score every development fold without touching the lock."""
    data = align_research_tables(features, labels, plan, config)
    if config.selection_policy == "static-threshold-v1":
        return _run_walk_forward_v1(data, plan, config)
    if config.selection_policy == "purged-tail-monotone-affine-v1":
        return _run_walk_forward_v2(data, plan, config)
    raise ValueError("baseline selection policy is not supported")

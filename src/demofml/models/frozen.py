"""Portable, versioned ridge state for production inference."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from demofml.evaluation.portfolio import PORTFOLIO_HORIZONS, PORTFOLIO_SYMBOLS
from demofml.features.causal import FEATURE_SET_ID
from demofml.labels.executable import LABEL_SET_ID
from demofml.models.baseline import FEATURE_COLUMNS, MODEL_SET_ID, BaselineConfig
from demofml.validation.splits import VALIDATION_SET_ID

FROZEN_RIDGE_FORMAT_VERSION = 1
FROZEN_MODEL_SET_ID = "frozen-ridge-model-v1"

_FIELDS = frozenset(
    {
        "format_version",
        "model_artifact_set",
        "model_set",
        "feature_set",
        "label_set",
        "validation_set",
        "training_scope",
        "model_type",
        "solver",
        "imputation",
        "standardize",
        "symbol",
        "horizon_minutes",
        "training_rows",
        "training_start",
        "training_end",
        "features",
        "alpha",
        "imputer_statistics",
        "scaler_mean",
        "scaler_scale",
        "ridge_coefficients",
        "ridge_intercept",
    }
)


def _finite_tuple(values: object, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} must contain exactly {length} values")
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must contain only numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"{name} must contain only finite values")
        result.append(number)
    return tuple(result)


def _require_utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware UTC")
    if value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")


@dataclass(frozen=True)
class FrozenRidgeModel:
    """Finite numerical state and provenance for a two-target ridge model."""

    format_version: int
    model_artifact_set: str
    model_set: str
    feature_set: str
    label_set: str
    validation_set: str
    training_scope: str
    model_type: str
    solver: str
    imputation: str
    standardize: bool
    symbol: str
    horizon_minutes: int
    training_rows: int
    training_start: datetime
    training_end: datetime
    features: tuple[str, ...]
    alpha: float
    imputer_statistics: tuple[float, ...]
    scaler_mean: tuple[float, ...]
    scaler_scale: tuple[float, ...]
    ridge_coefficients: tuple[tuple[float, ...], tuple[float, ...]]
    ridge_intercept: tuple[float, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.format_version, bool)
            or not isinstance(self.format_version, int)
            or self.format_version != FROZEN_RIDGE_FORMAT_VERSION
        ):
            raise ValueError("unsupported frozen ridge format version")
        if self.model_artifact_set != FROZEN_MODEL_SET_ID:
            raise ValueError("unsupported frozen ridge artifact set")
        if self.model_set != MODEL_SET_ID:
            raise ValueError("unsupported frozen ridge model provenance")
        if (
            self.feature_set != FEATURE_SET_ID
            or self.label_set != LABEL_SET_ID
            or self.validation_set != VALIDATION_SET_ID
            or self.training_scope != "per_symbol"
            or self.model_type != "ridge"
            or self.solver != "lsqr"
            or self.imputation != "training_median"
            or self.standardize is not True
        ):
            raise ValueError("unsupported frozen ridge provenance")
        if self.symbol not in PORTFOLIO_SYMBOLS:
            raise ValueError(f"unsupported frozen ridge symbol: {self.symbol}")
        if isinstance(self.horizon_minutes, bool) or not isinstance(
            self.horizon_minutes, int
        ):
            raise ValueError("frozen ridge horizon must be an integer")
        if self.horizon_minutes not in PORTFOLIO_HORIZONS:
            raise ValueError(
                f"unsupported frozen ridge horizon: {self.horizon_minutes}"
            )
        if (
            isinstance(self.training_rows, bool)
            or not isinstance(self.training_rows, int)
            or self.training_rows < 2
        ):
            raise ValueError("training_rows must be at least two")
        _require_utc(self.training_start, "training_start")
        _require_utc(self.training_end, "training_end")
        if self.training_start >= self.training_end:
            raise ValueError("frozen ridge training window must be increasing")
        if self.features != FEATURE_COLUMNS:
            raise ValueError("frozen ridge features are missing or reordered")
        if isinstance(self.alpha, bool) or not isinstance(self.alpha, (int, float)):
            raise ValueError("frozen ridge alpha must be finite and positive")
        if not math.isfinite(self.alpha):
            raise ValueError("frozen ridge alpha must be finite and positive")
        if self.alpha <= 0.0:
            raise ValueError("frozen ridge alpha must be finite and positive")

        feature_count = len(FEATURE_COLUMNS)
        if not all(
            isinstance(values, tuple)
            for values in (
                self.imputer_statistics,
                self.scaler_mean,
                self.scaler_scale,
                self.ridge_coefficients,
                self.ridge_intercept,
            )
        ):
            raise ValueError("frozen ridge numerical state must use immutable tuples")
        _finite_tuple(self.imputer_statistics, "imputer_statistics", feature_count)
        _finite_tuple(self.scaler_mean, "scaler_mean", feature_count)
        scales = _finite_tuple(self.scaler_scale, "scaler_scale", feature_count)
        if any(value <= 0.0 for value in scales):
            raise ValueError("scaler_scale must contain only positive values")
        if (
            not isinstance(self.ridge_coefficients, tuple)
            or len(self.ridge_coefficients) != 2
        ):
            raise ValueError("ridge_coefficients must have two rows")
        for row in self.ridge_coefficients:
            if not isinstance(row, tuple):
                raise ValueError(
                    "frozen ridge numerical state must use immutable tuples"
                )
            _finite_tuple(row, "ridge_coefficients", feature_count)
        _finite_tuple(self.ridge_intercept, "ridge_intercept", 2)


def _numeric_matrix(
    values: NDArray[np.generic], name: str, columns: int
) -> NDArray[np.float64]:
    if not isinstance(values, np.ndarray) or values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional ndarray")
    if values.shape[0] == 0 or values.shape[1] != columns:
        raise ValueError(f"{name} must have shape (rows, {columns}) with rows > 0")
    if values.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain numeric values")
    matrix = np.asarray(values, dtype=np.float64)
    if np.isinf(matrix).any():
        raise ValueError(f"{name} cannot contain infinite values")
    return matrix


def fit_frozen_ridge(
    features: NDArray[np.generic],
    targets: NDArray[np.generic],
    config: BaselineConfig,
    symbol: str,
    horizon_minutes: int,
    training_decision_times: Sequence[datetime],
) -> FrozenRidgeModel:
    """Fit the baseline pipeline and extract portable finite numerical state."""
    if not isinstance(config, BaselineConfig):
        raise TypeError("config must be a BaselineConfig")
    if symbol not in PORTFOLIO_SYMBOLS:
        raise ValueError(f"unsupported frozen ridge symbol: {symbol}")
    if (
        horizon_minutes not in PORTFOLIO_HORIZONS
        or horizon_minutes not in config.horizons_minutes
    ):
        raise ValueError(f"unsupported frozen ridge horizon: {horizon_minutes}")

    feature_matrix = _numeric_matrix(features, "features", len(FEATURE_COLUMNS))
    target_matrix = _numeric_matrix(targets, "targets", 2)
    if feature_matrix.shape[0] != target_matrix.shape[0]:
        raise ValueError("feature and target row counts must match")
    if feature_matrix.shape[0] < config.minimum_training_rows:
        raise ValueError("insufficient rows to fit frozen ridge")
    if np.isnan(target_matrix).any():
        raise ValueError("targets must contain only finite values")

    decision_times = tuple(training_decision_times)
    if len(decision_times) != feature_matrix.shape[0]:
        raise ValueError("training decision times must match training rows")
    previous: datetime | None = None
    for decision_time in decision_times:
        _require_utc(decision_time, "training decision time")
        if previous is not None and decision_time <= previous:
            raise ValueError("training decision times must be strictly ordered")
        previous = decision_time

    pipeline = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=config.alpha, solver="lsqr"),
    )
    pipeline.fit(feature_matrix, target_matrix)
    imputer = pipeline.named_steps["simpleimputer"]
    scaler = pipeline.named_steps["standardscaler"]
    ridge = pipeline.named_steps["ridge"]

    statistics = _finite_tuple(
        tuple(float(value) for value in imputer.statistics_),
        "imputer_statistics",
        len(FEATURE_COLUMNS),
    )
    means = _finite_tuple(
        tuple(float(value) for value in scaler.mean_),
        "scaler_mean",
        len(FEATURE_COLUMNS),
    )
    scales = _finite_tuple(
        tuple(float(value) for value in scaler.scale_),
        "scaler_scale",
        len(FEATURE_COLUMNS),
    )
    coefficients = tuple(
        _finite_tuple(
            tuple(float(value) for value in row),
            "ridge_coefficients",
            len(FEATURE_COLUMNS),
        )
        for row in np.asarray(ridge.coef_, dtype=np.float64)
    )
    if len(coefficients) != 2:
        raise RuntimeError("ridge did not produce two-target coefficients")
    intercept = _finite_tuple(
        tuple(float(value) for value in np.asarray(ridge.intercept_)),
        "ridge_intercept",
        2,
    )

    return FrozenRidgeModel(
        format_version=FROZEN_RIDGE_FORMAT_VERSION,
        model_artifact_set=FROZEN_MODEL_SET_ID,
        model_set=config.id,
        feature_set=config.feature_set,
        label_set=config.label_set,
        validation_set=config.validation_set,
        training_scope=config.training_scope,
        model_type=config.model_type,
        solver=config.solver,
        imputation=config.imputation,
        standardize=config.standardize,
        symbol=symbol,
        horizon_minutes=horizon_minutes,
        training_rows=feature_matrix.shape[0],
        training_start=decision_times[0],
        training_end=decision_times[-1],
        features=config.features,
        alpha=config.alpha,
        imputer_statistics=statistics,
        scaler_mean=means,
        scaler_scale=scales,
        ridge_coefficients=(coefficients[0], coefficients[1]),
        ridge_intercept=(intercept[0], intercept[1]),
    )


def predict_frozen_ridge(
    model: FrozenRidgeModel, features: NDArray[np.generic]
) -> NDArray[np.float64]:
    """Predict with frozen state only; no sklearn estimator is created or fit."""
    if not isinstance(model, FrozenRidgeModel):
        raise TypeError("model must be a FrozenRidgeModel")
    matrix = _numeric_matrix(features, "features", len(model.features))
    statistics = np.asarray(model.imputer_statistics, dtype=np.float64)
    imputed = np.where(np.isnan(matrix), statistics, matrix)
    scaled = (imputed - np.asarray(model.scaler_mean)) / np.asarray(model.scaler_scale)
    predictions = scaled @ np.asarray(
        model.ridge_coefficients, dtype=np.float64
    ).T + np.asarray(model.ridge_intercept, dtype=np.float64)
    if not np.isfinite(predictions).all():
        raise RuntimeError("frozen ridge produced non-finite predictions")
    return np.asarray(predictions, dtype=np.float64)


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _payload(model: FrozenRidgeModel) -> dict[str, object]:
    return {
        "format_version": model.format_version,
        "model_artifact_set": model.model_artifact_set,
        "model_set": model.model_set,
        "feature_set": model.feature_set,
        "label_set": model.label_set,
        "validation_set": model.validation_set,
        "training_scope": model.training_scope,
        "model_type": model.model_type,
        "solver": model.solver,
        "imputation": model.imputation,
        "standardize": model.standardize,
        "symbol": model.symbol,
        "horizon_minutes": model.horizon_minutes,
        "training_rows": model.training_rows,
        "training_start": _timestamp(model.training_start),
        "training_end": _timestamp(model.training_end),
        "features": list(model.features),
        "alpha": model.alpha,
        "imputer_statistics": list(model.imputer_statistics),
        "scaler_mean": list(model.scaler_mean),
        "scaler_scale": list(model.scaler_scale),
        "ridge_coefficients": [list(row) for row in model.ridge_coefficients],
        "ridge_intercept": list(model.ridge_intercept),
    }


def write_frozen_ridge(model: FrozenRidgeModel, path: Path) -> None:
    """Atomically publish a canonical JSON artifact without replacing a path."""
    if not isinstance(model, FrozenRidgeModel):
        raise TypeError("model must be a FrozenRidgeModel")
    path = path.expanduser().resolve()
    if path.exists():
        raise RuntimeError(f"Refusing to replace frozen ridge artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = (
        json.dumps(_payload(model), allow_nan=False, indent=2, sort_keys=True) + "\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as destination:
            destination.write(serialized)
            destination.flush()
            os.fsync(destination.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise RuntimeError(
                f"Refusing to replace frozen ridge artifact: {path}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _string(values: dict[str, Any], name: str) -> str:
    value = values[name]
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return value


def _integer(values: dict[str, Any], name: str) -> int:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _number(values: dict[str, Any], name: str) -> float:
    value = values[name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _parse_timestamp(values: dict[str, Any], name: str) -> datetime:
    value = _string(values, name)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp") from error
    _require_utc(parsed, name)
    return parsed.astimezone(UTC)


def load_frozen_ridge(path: Path) -> FrozenRidgeModel:
    """Load a frozen ridge artifact using an exact, versioned JSON schema."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Frozen ridge artifact is not a file: {path}")
    try:
        values = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"invalid frozen ridge JSON: {error}") from error
    if not isinstance(values, dict):
        raise ValueError("frozen ridge JSON root must be an object")
    actual_fields = set(values)
    if actual_fields != _FIELDS:
        missing = sorted(_FIELDS - actual_fields)
        unknown = sorted(actual_fields - _FIELDS)
        raise ValueError(
            f"frozen ridge fields differ; missing={missing}, unknown={unknown}"
        )
    standardize = values["standardize"]
    if not isinstance(standardize, bool):
        raise ValueError("standardize must be a boolean")
    features = values["features"]
    if not isinstance(features, list) or not all(
        isinstance(value, str) for value in features
    ):
        raise ValueError("features must be a list of strings")
    feature_count = len(FEATURE_COLUMNS)
    coefficients = values["ridge_coefficients"]
    if not isinstance(coefficients, list) or len(coefficients) != 2:
        raise ValueError("ridge_coefficients must have two rows")

    ridge_intercept = _finite_tuple(values["ridge_intercept"], "ridge_intercept", 2)
    return FrozenRidgeModel(
        format_version=_integer(values, "format_version"),
        model_artifact_set=_string(values, "model_artifact_set"),
        model_set=_string(values, "model_set"),
        feature_set=_string(values, "feature_set"),
        label_set=_string(values, "label_set"),
        validation_set=_string(values, "validation_set"),
        training_scope=_string(values, "training_scope"),
        model_type=_string(values, "model_type"),
        solver=_string(values, "solver"),
        imputation=_string(values, "imputation"),
        standardize=standardize,
        symbol=_string(values, "symbol"),
        horizon_minutes=_integer(values, "horizon_minutes"),
        training_rows=_integer(values, "training_rows"),
        training_start=_parse_timestamp(values, "training_start"),
        training_end=_parse_timestamp(values, "training_end"),
        features=tuple(features),
        alpha=_number(values, "alpha"),
        imputer_statistics=_finite_tuple(
            values["imputer_statistics"], "imputer_statistics", feature_count
        ),
        scaler_mean=_finite_tuple(values["scaler_mean"], "scaler_mean", feature_count),
        scaler_scale=_finite_tuple(
            values["scaler_scale"], "scaler_scale", feature_count
        ),
        ridge_coefficients=(
            _finite_tuple(coefficients[0], "ridge_coefficients", feature_count),
            _finite_tuple(coefficients[1], "ridge_coefficients", feature_count),
        ),
        ridge_intercept=(ridge_intercept[0], ridge_intercept[1]),
    )

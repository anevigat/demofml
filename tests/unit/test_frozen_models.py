import json
from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from demofml.models.baseline import FEATURE_COLUMNS, load_baseline_config
from demofml.models.frozen import (
    FROZEN_MODEL_SET_ID,
    FROZEN_RIDGE_FORMAT_VERSION,
    FrozenRidgeModel,
    fit_frozen_ridge,
    load_frozen_ridge,
    predict_frozen_ridge,
    write_frozen_ridge,
)

PROJECT_ROOT = Path(__file__).parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs/experiments/baseline-ridge-v1.toml"


def _training_data() -> tuple[np.ndarray, np.ndarray, tuple[datetime, ...]]:
    generator = np.random.default_rng(1729)
    features = generator.normal(size=(40, len(FEATURE_COLUMNS)))
    features[::7, 2] = np.nan
    features[::11, 8] = np.nan
    features[:, 12] = np.nan
    targets = np.column_stack(
        (
            np.nan_to_num(features[:, 0]) * 0.3 - features[:, 4] * 0.1,
            np.nan_to_num(features[:, 0]) * -0.2 + features[:, 6] * 0.4,
        )
    )
    start = datetime(2025, 1, 2, tzinfo=UTC)
    times = tuple(start + timedelta(minutes=5 * row) for row in range(len(features)))
    return features, targets, times


def _model() -> FrozenRidgeModel:
    features, targets, times = _training_data()
    return fit_frozen_ridge(
        features,
        targets,
        load_baseline_config(MODEL_CONFIG),
        "EURUSD",
        30,
        times,
    )


def _artifact(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    path = tmp_path / "ridge.json"
    write_frozen_ridge(_model(), path)
    return path, json.loads(path.read_text())


def test_predictions_match_exact_sklearn_pipeline_before_and_after_json(
    tmp_path: Path,
) -> None:
    training_features, targets, times = _training_data()
    config = load_baseline_config(MODEL_CONFIG)
    model = fit_frozen_ridge(training_features, targets, config, "EURUSD", 30, times)
    reference = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=config.alpha, solver="lsqr"),
    ).fit(training_features, targets)
    inference = training_features[[1, 2, 3, 4]].copy()
    inference[0, 0] = np.nan
    inference[1, -1] = np.nan

    expected = reference.predict(inference)
    assert_allclose(predict_frozen_ridge(model, inference), expected, rtol=1e-13)

    path = tmp_path / "nested" / "ridge.json"
    write_frozen_ridge(model, path)
    loaded = load_frozen_ridge(path)
    assert loaded == model
    assert_allclose(predict_frozen_ridge(loaded, inference), expected, rtol=1e-13)
    assert "NaN" not in path.read_text()
    assert not list(path.parent.glob("*.partial"))


def test_fitted_model_records_provenance_and_is_immutable() -> None:
    model = _model()

    assert model.format_version == FROZEN_RIDGE_FORMAT_VERSION
    assert model.model_artifact_set == FROZEN_MODEL_SET_ID
    assert model.features == FEATURE_COLUMNS
    assert model.training_rows == 40
    assert model.symbol == "EURUSD"
    assert model.horizon_minutes == 30
    assert all(np.isfinite(model.imputer_statistics))
    with pytest.raises(FrozenInstanceError):
        model.alpha = 2.0  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("format_version", 2, "format version"),
        ("model_artifact_set", "future-v2", "artifact set"),
        ("symbol", "BTCUSD", "unsupported.*symbol"),
        ("horizon_minutes", 45, "unsupported.*horizon"),
        ("model_set", "future-ridge-v2", "provenance"),
        ("feature_set", "future-v2", "provenance"),
        ("solver", "auto", "provenance"),
    ],
)
def test_loader_rejects_unsupported_identity_and_provenance(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path, payload = _artifact(tmp_path)
    payload[field] = value
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_frozen_ridge(path)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_loader_requires_exact_fields(tmp_path: Path, mutation: str) -> None:
    path, payload = _artifact(tmp_path)
    if mutation == "missing":
        del payload["alpha"]
    else:
        payload["surprise"] = 1
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=r"missing=.*unknown="):
        load_frozen_ridge(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("features", list(reversed(FEATURE_COLUMNS)), "missing or reordered"),
        ("scaler_mean", [0.0], "exactly 15"),
        ("scaler_scale", [0.0] * len(FEATURE_COLUMNS), "positive"),
        ("ridge_coefficients", [[0.0] * len(FEATURE_COLUMNS)], "two rows"),
        ("ridge_intercept", [0.0], "exactly 2"),
    ],
)
def test_loader_rejects_reordered_features_and_bad_state_dimensions(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path, payload = _artifact(tmp_path)
    payload[field] = value
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_frozen_ridge(path)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_loader_rejects_non_finite_json_numbers(tmp_path: Path, constant: str) -> None:
    path, _ = _artifact(tmp_path)
    text = path.read_text().replace('"alpha": 1.0', f'"alpha": {constant}')
    path.write_text(text)

    with pytest.raises(ValueError, match="non-finite JSON"):
        load_frozen_ridge(path)


def test_loader_rejects_duplicate_fields_and_non_object_json(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"format_version": 1, "format_version": 1}')
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_frozen_ridge(duplicate)

    duplicate.write_text("[]")
    with pytest.raises(ValueError, match="root must be an object"):
        load_frozen_ridge(duplicate)


def test_writer_refuses_to_replace_existing_artifact(tmp_path: Path) -> None:
    path = tmp_path / "ridge.json"
    write_frozen_ridge(_model(), path)
    original = path.read_bytes()

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        write_frozen_ridge(_model(), path)

    assert path.read_bytes() == original


@pytest.mark.parametrize(
    ("feature_change", "target_change", "message"),
    [
        (lambda value: value[:, :-1], lambda value: value, r"shape \(rows, 15\)"),
        (lambda value: value, lambda value: value[:, :1], r"shape \(rows, 2\)"),
        (lambda value: value[:30], lambda value: value, "row counts"),
        (
            lambda value: np.where(
                np.arange(value.size).reshape(value.shape) == 0, np.inf, value
            ),
            lambda value: value,
            "infinite",
        ),
        (
            lambda value: value,
            lambda value: np.where(
                np.arange(value.size).reshape(value.shape) == 0, np.nan, value
            ),
            "finite",
        ),
    ],
)
def test_fit_rejects_invalid_dimensions_and_non_finite_values(
    feature_change: Any, target_change: Any, message: str
) -> None:
    features, targets, times = _training_data()

    with pytest.raises(ValueError, match=message):
        fit_frozen_ridge(
            feature_change(features),
            target_change(targets),
            load_baseline_config(MODEL_CONFIG),
            "EURUSD",
            15,
            times,
        )


def test_fit_rejects_invalid_training_identity_and_times() -> None:
    features, targets, times = _training_data()
    config = load_baseline_config(MODEL_CONFIG)
    with pytest.raises(ValueError, match="unsupported.*symbol"):
        fit_frozen_ridge(features, targets, config, "BTCUSD", 15, times)
    with pytest.raises(ValueError, match="unsupported.*horizon"):
        fit_frozen_ridge(features, targets, config, "EURUSD", 45, times)
    with pytest.raises(ValueError, match="match training rows"):
        fit_frozen_ridge(features, targets, config, "EURUSD", 15, times[:-1])
    with pytest.raises(ValueError, match="strictly ordered"):
        fit_frozen_ridge(
            features,
            targets,
            config,
            "EURUSD",
            15,
            (*times[:-1], times[-2]),
        )
    non_utc = (*times[:-1], times[-1].replace(tzinfo=None))
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        fit_frozen_ridge(features, targets, config, "EURUSD", 15, non_utc)


def test_fit_honors_config_minimum_rows_and_predict_validates_input() -> None:
    features, targets, times = _training_data()
    config = replace(load_baseline_config(MODEL_CONFIG), minimum_training_rows=50)
    with pytest.raises(ValueError, match="insufficient rows"):
        fit_frozen_ridge(features, targets, config, "EURUSD", 15, times)

    model = _model()
    with pytest.raises(ValueError, match=r"shape \(rows, 15\)"):
        predict_frozen_ridge(model, features[:, :-1])
    infinite = features[:1].copy()
    infinite[0, 0] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        predict_frozen_ridge(model, infinite)


def test_load_rejects_missing_or_malformed_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not a file"):
        load_frozen_ridge(tmp_path / "missing.json")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")
    with pytest.raises(ValueError, match="invalid frozen ridge JSON"):
        load_frozen_ridge(malformed)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"horizon_minutes": True}, "horizon must be an integer"),
        ({"training_start": datetime(2030, 1, 1, tzinfo=UTC)}, "window"),
        ({"alpha": True}, "alpha must be finite"),
        ({"alpha": 0.0}, "alpha must be finite"),
        ({"imputer_statistics": [0.0] * len(FEATURE_COLUMNS)}, "immutable tuples"),
        ({"ridge_intercept": [0.0, 0.0]}, "immutable tuples"),
        (
            {
                "ridge_coefficients": (
                    [0.0] * len(FEATURE_COLUMNS),
                    (0.0,) * len(FEATURE_COLUMNS),
                )
            },
            "immutable tuples",
        ),
    ],
)
def test_frozen_model_rejects_mutable_or_ambiguous_state(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_model(), **change)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_artifact_set", 1, "must be a string"),
        ("format_version", "1", "must be an integer"),
        ("alpha", "1", "must be a number"),
        ("training_start", "not-a-time", "valid ISO-8601"),
        ("standardize", "true", "must be a boolean"),
        ("features", "mid_return_1", "list of strings"),
    ],
)
def test_loader_rejects_wrong_json_field_types(
    tmp_path: Path, field: str, value: object, message: str
) -> None:
    path, payload = _artifact(tmp_path)
    payload[field] = value
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=message):
        load_frozen_ridge(path)


def test_fit_and_predict_reject_non_array_and_overflowing_values() -> None:
    features, targets, times = _training_data()
    config = load_baseline_config(MODEL_CONFIG)
    non_array: Any = [[0.0] * len(FEATURE_COLUMNS)]
    with pytest.raises(ValueError, match="two-dimensional ndarray"):
        fit_frozen_ridge(non_array, targets, config, "EURUSD", 15, times)
    text = features.astype(str)
    with pytest.raises(ValueError, match="numeric values"):
        fit_frozen_ridge(text, targets, config, "EURUSD", 15, times)
    wrong_config: Any = object()
    with pytest.raises(TypeError, match="BaselineConfig"):
        fit_frozen_ridge(features, targets, wrong_config, "EURUSD", 15, times)

    huge_row = (1e308,) * len(FEATURE_COLUMNS)
    model = replace(_model(), ridge_coefficients=(huge_row, huge_row))
    huge = np.full((1, len(FEATURE_COLUMNS)), 1e308)
    with pytest.raises(RuntimeError, match="non-finite predictions"):
        predict_frozen_ridge(model, huge)

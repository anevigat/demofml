import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import demofml.models.baseline as baseline_module
from demofml.evaluation.signals import (
    EVALUATION_SET_ID,
    evaluate_predictions,
)
from demofml.features.causal import FEATURE_SCHEMA
from demofml.labels.executable import label_schema
from demofml.models.baseline import (
    FEATURE_COLUMNS,
    MODEL_SET_ID,
    PREDICTION_SET_ID,
    BaselineConfig,
    align_research_tables,
    load_baseline_config,
    run_walk_forward,
)
from demofml.models.build import run_baseline_experiment
from demofml.validation.splits import ValidationPlan, load_validation_plan

PROJECT_ROOT = Path(__file__).parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs/experiments/baseline-ridge-v1.toml"
VALIDATION_CONFIG = PROJECT_ROOT / "configs/experiments/purged-walk-forward-v1.toml"


def _config() -> BaselineConfig:
    return load_baseline_config(MODEL_CONFIG)


def _plan() -> ValidationPlan:
    return replace(
        load_validation_plan(VALIDATION_CONFIG),
        train_start=datetime(2021, 12, 1, tzinfo=UTC),
        first_validation_start=datetime(2022, 1, 1, tzinfo=UTC),
        development_end_exclusive=datetime(2022, 2, 1, tzinfo=UTC),
        locked_test_start=datetime(2022, 2, 1, tzinfo=UTC),
        locked_test_end_exclusive=datetime(2022, 3, 1, tzinfo=UTC),
    )


def _tables() -> tuple[pa.Table, pa.Table]:
    training_start = datetime(2021, 12, 31, 20, 0, tzinfo=UTC)
    validation_start = datetime(2022, 1, 1, tzinfo=UTC)
    times = [training_start + timedelta(minutes=5 * index) for index in range(24)]
    times.extend(validation_start + timedelta(minutes=5 * index) for index in range(6))
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for index, decision_time in enumerate(times):
        signal = float(index - 12)
        feature_row: dict[str, object] = {
            "symbol": "EURUSD",
            "bar_end": decision_time,
        }
        for feature_index, name in enumerate(FEATURE_COLUMNS):
            feature_row[name] = signal + feature_index / 100.0
        if index == 0:
            feature_row["mid_return_12"] = None
        feature_rows.append(feature_row)

        long_return = signal / 10_000.0
        short_return = -signal / 10_000.0
        label_row: dict[str, object] = {
            "symbol": "EURUSD",
            "decision_time": decision_time,
            "entry_time": decision_time,
            "entry_bid": 1.0,
            "entry_ask": 1.0001,
        }
        for horizon in (15, 30, 60):
            suffix = f"{horizon}m"
            label_row[f"exit_time_{suffix}"] = decision_time + timedelta(
                minutes=horizon
            )
            label_row[f"long_return_{suffix}"] = long_return
            label_row[f"short_return_{suffix}"] = short_return
            label_row[f"action_{suffix}"] = (
                "long" if long_return > short_return else "short"
            )
        label_rows.append(label_row)
    return (
        pa.Table.from_pylist(feature_rows, schema=FEATURE_SCHEMA),
        pa.Table.from_pylist(label_rows, schema=label_schema((15, 30, 60))),
    )


def test_walk_forward_trains_only_on_purged_development_rows() -> None:
    features, labels = _tables()

    predictions = run_walk_forward(features, labels, _plan(), _config())

    assert predictions.num_rows == 18
    assert set(predictions.column("fold_id").to_pylist()) == {"wf-2022-01"}
    assert set(predictions.column("horizon_minutes").to_pylist()) == {15, 30, 60}
    assert set(predictions.column("action").to_pylist()) == {"long"}
    assert min(predictions.column("decision_time").to_pylist()) == datetime(
        2022, 1, 1, tzinfo=UTC
    )
    assert predictions.schema.metadata is not None
    assert (
        predictions.schema.metadata[b"demofml.prediction_set"]
        == PREDICTION_SET_ID.encode()
    )
    assert predictions.column("realized_return")[0].as_py() > 0.0
    assert predictions.column("entry_time")[0].as_py() == datetime(
        2022, 1, 1, tzinfo=UTC
    )
    assert predictions.column("exit_time")[0].as_py() == datetime(
        2022, 1, 1, 0, 15, tzinfo=UTC
    )


def test_future_validation_row_cannot_change_earlier_prediction() -> None:
    features, labels = _tables()
    original = run_walk_forward(features, labels, _plan(), _config())
    rows = features.to_pylist()
    for name in FEATURE_COLUMNS:
        rows[-1][name] = 1_000_000.0
    changed_features = pa.Table.from_pylist(rows, schema=FEATURE_SCHEMA)

    changed = run_walk_forward(changed_features, labels, _plan(), _config())

    assert changed.slice(0, 5).equals(original.slice(0, 5))


def test_positive_threshold_can_abstain_to_flat() -> None:
    features, labels = _tables()
    config = replace(_config(), action_threshold_bps=1_000.0)

    predictions = run_walk_forward(features, labels, _plan(), config)

    assert set(predictions.column("action").to_pylist()) == {"flat"}
    assert set(predictions.column("realized_return").to_pylist()) == {0.0}
    assert evaluate_predictions(predictions)["aggregate"][0]["hit_rate"] is None


def test_alignment_rejects_mismatched_and_locked_rows() -> None:
    features, labels = _tables()
    wrong_symbol_rows = labels.to_pylist()
    wrong_symbol_rows[0]["symbol"] = "GBPUSD"
    wrong_symbols = pa.Table.from_pylist(wrong_symbol_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="symbols are not aligned"):
        align_research_tables(features, wrong_symbols, _plan(), _config())

    locked_feature_rows = features.to_pylist()
    locked_label_rows = labels.to_pylist()
    locked_time = _plan().locked_test_start
    locked_feature_rows[-1]["bar_end"] = locked_time
    locked_label_rows[-1]["decision_time"] = locked_time
    locked_features = pa.Table.from_pylist(locked_feature_rows, schema=features.schema)
    locked_labels = pa.Table.from_pylist(locked_label_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="locked-test rows are forbidden"):
        align_research_tables(locked_features, locked_labels, _plan(), _config())


def test_alignment_rejects_invalid_features_and_keys() -> None:
    features, labels = _tables()
    with pytest.raises(ValueError, match="row counts"):
        align_research_tables(features.slice(0, 1), labels, _plan(), _config())
    shifted_rows = labels.to_pylist()
    shifted_rows[0]["decision_time"] += timedelta(minutes=1)
    shifted = pa.Table.from_pylist(shifted_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="decision times are not aligned"):
        align_research_tables(features, shifted, _plan(), _config())

    rows = features.to_pylist()
    rows[0]["spread_bps"] = float("inf")
    infinite = pa.Table.from_pylist(rows, schema=features.schema)
    with pytest.raises(ValueError, match="infinite values"):
        align_research_tables(infinite, labels, _plan(), _config())

    with pytest.raises(ValueError, match="feature schema is missing"):
        align_research_tables(
            features.drop(["mid_return_1"]), labels, _plan(), _config()
        )


def test_alignment_rejects_contract_types_symbols_and_order() -> None:
    features, labels = _tables()
    feature_index = features.schema.get_field_index("mid_return_1")
    wrong_type = features.set_column(
        feature_index,
        pa.field("mid_return_1", pa.float32()),
        pa.array(features.column("mid_return_1").to_pylist(), type=pa.float32()),
    )
    with pytest.raises(ValueError, match="does not match causal-v1"):
        align_research_tables(wrong_type, labels, _plan(), _config())

    feature_rows = features.to_pylist()
    label_rows = labels.to_pylist()
    feature_rows[-1]["symbol"] = "GBPUSD"
    label_rows[-1]["symbol"] = "GBPUSD"
    mixed_features = pa.Table.from_pylist(feature_rows, schema=features.schema)
    mixed_labels = pa.Table.from_pylist(label_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="exactly one symbol"):
        align_research_tables(mixed_features, mixed_labels, _plan(), _config())

    feature_rows = features.to_pylist()
    label_rows = labels.to_pylist()
    feature_rows[1]["bar_end"] = feature_rows[0]["bar_end"]
    label_rows[1]["decision_time"] = label_rows[0]["decision_time"]
    duplicate_features = pa.Table.from_pylist(feature_rows, schema=features.schema)
    duplicate_labels = pa.Table.from_pylist(label_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="strictly ordered"):
        align_research_tables(duplicate_features, duplicate_labels, _plan(), _config())


def test_alignment_rejects_missing_targets_and_plan_mismatch() -> None:
    features, labels = _tables()
    with pytest.raises(ValueError, match="label schema is missing"):
        align_research_tables(
            features, labels.drop(["long_return_15m"]), _plan(), _config()
        )
    with pytest.raises(ValueError, match="data contracts differ"):
        align_research_tables(
            features,
            labels,
            replace(_plan(), feature_set="future-v2"),
            _config(),
        )
    with pytest.raises(ValueError, match="validation purge differ"):
        align_research_tables(
            features,
            labels,
            replace(_plan(), max_horizon_minutes=90, purge_minutes=95),
            _config(),
        )


def test_walk_forward_requires_usable_fold_rows() -> None:
    features, labels = _tables()
    with pytest.raises(ValueError, match="insufficient training rows"):
        run_walk_forward(
            features,
            labels,
            _plan(),
            replace(_config(), minimum_training_rows=25),
        )

    rows = labels.to_pylist()
    for row in rows[24:]:
        row["long_return_15m"] = None
        row["short_return_15m"] = None
    unresolved = pa.Table.from_pylist(rows, schema=labels.schema)
    with pytest.raises(ValueError, match="no resolved validation labels"):
        run_walk_forward(features, unresolved, _plan(), _config())

    later_plan = replace(
        _plan(),
        first_validation_start=datetime(2022, 1, 2, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="has no validation rows"):
        run_walk_forward(features, labels, later_plan, _config())


def test_short_predictions_use_short_executable_returns() -> None:
    features, labels = _tables()
    rows = labels.to_pylist()
    for row in rows:
        for horizon in (15, 30, 60):
            suffix = f"{horizon}m"
            row[f"long_return_{suffix}"], row[f"short_return_{suffix}"] = (
                row[f"short_return_{suffix}"],
                row[f"long_return_{suffix}"],
            )
    inverted = pa.Table.from_pylist(rows, schema=labels.schema)

    predictions = run_walk_forward(features, inverted, _plan(), _config())

    assert set(predictions.column("action").to_pylist()) == {"short"}
    assert min(predictions.column("realized_return").to_pylist()) > 0.0


def test_non_finite_model_predictions_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    features, labels = _tables()

    def non_finite(*args: object, **kwargs: object) -> Any:
        del args, kwargs
        return [[float("nan"), 0.0]] * 6

    monkeypatch.setattr(baseline_module, "_fit_predict", non_finite)
    with pytest.raises(RuntimeError, match="non-finite prediction"):
        run_walk_forward(features, labels, _plan(), _config())


def test_evaluation_uses_executable_action_returns() -> None:
    features, labels = _tables()
    predictions = run_walk_forward(features, labels, _plan(), _config())

    report = evaluate_predictions(predictions)

    assert report["evaluation_set"] == EVALUATION_SET_ID
    assert report["model_set"] == MODEL_SET_ID
    assert len(report["aggregate"]) == 3
    first = report["aggregate"][0]
    assert first["observations"] == 6
    assert first["trades"] == 6
    assert first["trade_rate"] == 1.0
    assert first["hit_rate"] == 1.0
    assert report["always_flat_comparator"]["mean_executable_return_bps"] == 0.0


def test_evaluation_rejects_invalid_prediction_contracts() -> None:
    features, labels = _tables()
    predictions = run_walk_forward(features, labels, _plan(), _config())
    with pytest.raises(ValueError, match="schema is missing"):
        evaluate_predictions(predictions.drop(["action"]))
    with pytest.raises(ValueError, match="metadata"):
        evaluate_predictions(predictions.replace_schema_metadata({}))
    with pytest.raises(ValueError, match="empty predictions"):
        evaluate_predictions(predictions.slice(0, 0))

    rows = predictions.to_pylist()
    rows[0]["action"] = "hold"
    invalid = pa.Table.from_pylist(rows, schema=predictions.schema)
    with pytest.raises(ValueError, match="invalid action"):
        evaluate_predictions(invalid)

    rows[0]["action"] = "long"
    rows[0]["realized_return"] = float("nan")
    non_finite = pa.Table.from_pylist(rows, schema=predictions.schema)
    with pytest.raises(ValueError, match="non-finite returns"):
        evaluate_predictions(non_finite)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "other"}, "model id"),
        ({"feature_set": "future-v2"}, "incompatible"),
        ({"validation_set": "random-v1"}, "purged"),
        ({"horizons_minutes": (30, 15)}, "unique and increasing"),
        ({"training_scope": "pooled"}, "per_symbol"),
        ({"solver": "auto"}, "lsqr"),
        ({"alpha": 0.0}, "finite and positive"),
        ({"imputation": "global_median"}, "training median"),
        ({"action_threshold_bps": float("nan")}, "must be finite"),
        ({"minimum_training_rows": 1}, "at least two"),
        ({"random_seed": -1}, "cannot be negative"),
        ({"locked_test_policy": "allowed"}, "must remain forbidden"),
        ({"features": FEATURE_COLUMNS[:-1]}, "do not match causal-v1"),
    ],
)
def test_baseline_config_rejects_unsafe_values(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(), **change)


def test_baseline_config_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not a file"):
        load_baseline_config(tmp_path / "missing.toml")

    malformed = tmp_path / "malformed.toml"
    malformed.write_text(MODEL_CONFIG.read_text().replace("alpha = 1.0\n", ""))
    with pytest.raises(ValueError, match="invalid baseline config field"):
        load_baseline_config(malformed)


def test_experiment_artifacts_are_published_atomically(tmp_path: Path) -> None:
    features, labels = _tables()
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    validation_path = tmp_path / "validation.toml"
    output = tmp_path / "experiment"
    pq.write_table(features, features_path)
    pq.write_table(labels, labels_path)
    validation_path.write_text(
        VALIDATION_CONFIG.read_text()
        .replace("2018-01-01T00:00:00Z", "2021-12-01T00:00:00Z")
        .replace("2025-01-01T00:00:00Z", "2022-02-01T00:00:00Z")
        .replace("2026-03-11T00:00:00Z", "2022-03-01T00:00:00Z")
    )

    result = run_baseline_experiment(
        features_path,
        labels_path,
        validation_path,
        MODEL_CONFIG,
        output,
    )

    assert result.prediction_rows == 18
    assert result.fold_count == 1
    assert result.symbol == "EURUSD"
    assert pq.read_table(output / "predictions.parquet").num_rows == 18
    assert json.loads((output / "metrics.json").read_text())["format_version"] == 1
    assert not list(tmp_path.glob("*.partial"))
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        run_baseline_experiment(
            features_path,
            labels_path,
            validation_path,
            MODEL_CONFIG,
            output,
        )

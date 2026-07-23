import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from demofml.features.causal import FEATURE_SCHEMA
from demofml.labels.executable import label_schema
from demofml.validation.build import (
    build_validation_manifest,
    validation_manifest,
)
from demofml.validation.splits import (
    INTERVAL_SEMANTICS,
    VALIDATION_SET_ID,
    VALIDATION_STRATEGY,
    TemporalFold,
    ValidationPlan,
    load_validation_plan,
    select_fold_rows,
    select_locked_test_rows,
    validate_feature_label_schemas,
)

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = PROJECT_ROOT / "configs/experiments/purged-walk-forward-v1.toml"


def _plan() -> ValidationPlan:
    return load_validation_plan(CONFIG)


def test_plan_generates_expanding_monthly_folds_with_purge() -> None:
    plan = _plan()
    folds = plan.folds()

    assert plan.id == VALIDATION_SET_ID
    assert plan.strategy == VALIDATION_STRATEGY
    assert plan.interval_semantics == INTERVAL_SEMANTICS
    assert len(folds) == 36
    assert folds[0].id == "wf-2022-01"
    assert folds[0].train_start == datetime(2018, 1, 1, tzinfo=UTC)
    assert folds[0].train_end_exclusive == datetime(2021, 12, 31, 22, 55, tzinfo=UTC)
    assert folds[0].validation_end_exclusive == datetime(2022, 2, 1, tzinfo=UTC)
    assert folds[-1].id == "wf-2024-12"
    assert folds[-1].train_start == folds[0].train_start
    assert folds[-1].validation_end_exclusive == datetime(
        2024, 12, 31, 22, 55, tzinfo=UTC
    )
    assert plan.locked_test_decision_end == datetime(2026, 3, 10, 22, 55, tzinfo=UTC)


def test_fold_selection_excludes_boundaries_and_purge() -> None:
    fold = TemporalFold(
        id="wf-test",
        train_start=datetime(2023, 1, 1, tzinfo=UTC),
        train_end_exclusive=datetime(2023, 1, 2, 22, 55, tzinfo=UTC),
        validation_start=datetime(2023, 1, 3, tzinfo=UTC),
        validation_end_exclusive=datetime(2023, 1, 4, tzinfo=UTC),
        purge_minutes=65,
    )
    times = [
        datetime(2023, 1, 1, tzinfo=UTC),
        datetime(2023, 1, 2, 22, 54, tzinfo=UTC),
        datetime(2023, 1, 2, 22, 55, tzinfo=UTC),
        datetime(2023, 1, 3, tzinfo=UTC),
        datetime(2023, 1, 3, 23, 59, tzinfo=UTC),
        datetime(2023, 1, 4, tzinfo=UTC),
    ]

    selection = select_fold_rows(times, fold)

    assert selection.train == (0, 1)
    assert selection.validation == (3, 4)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"purge_minutes": 0}, "must be positive"),
        (
            {"train_start": datetime(2023, 1, 2, 22, 55, tzinfo=UTC)},
            "training interval must be non-empty",
        ),
        (
            {"train_end_exclusive": datetime(2023, 1, 3, tzinfo=UTC)},
            "require a purge interval",
        ),
        (
            {"validation_end_exclusive": datetime(2023, 1, 3, tzinfo=UTC)},
            "validation interval must be non-empty",
        ),
        ({"purge_minutes": 60}, "does not match purge_minutes"),
    ],
)
def test_fold_rejects_invalid_intervals(change: dict[str, Any], message: str) -> None:
    fold = TemporalFold(
        id="wf-test",
        train_start=datetime(2023, 1, 1, tzinfo=UTC),
        train_end_exclusive=datetime(2023, 1, 2, 22, 55, tzinfo=UTC),
        validation_start=datetime(2023, 1, 3, tzinfo=UTC),
        validation_end_exclusive=datetime(2023, 1, 4, tzinfo=UTC),
        purge_minutes=65,
    )
    with pytest.raises(ValueError, match=message):
        replace(fold, **change)


def test_locked_selection_cannot_require_data_outside_lock() -> None:
    plan = _plan()
    times = [
        plan.locked_test_start - timedelta(minutes=5),
        plan.locked_test_start,
        plan.locked_test_decision_end - timedelta(minutes=5),
        plan.locked_test_decision_end,
    ]

    assert select_locked_test_rows(times, plan) == (1, 2)
    for index in select_locked_test_rows(times, plan):
        assert times[index] + plan.information_window < plan.locked_test_end_exclusive


def test_feature_and_label_metadata_must_match_plan() -> None:
    plan = _plan()
    validate_feature_label_schemas(FEATURE_SCHEMA, label_schema((15, 30, 60)), plan)

    wrong_features = FEATURE_SCHEMA.with_metadata(
        {**(FEATURE_SCHEMA.metadata or {}), b"demofml.feature_set": b"future-v2"}
    )
    with pytest.raises(ValueError, match="feature_set"):
        validate_feature_label_schemas(wrong_features, label_schema((15, 30, 60)), plan)
    with pytest.raises(ValueError, match="horizons"):
        validate_feature_label_schemas(FEATURE_SCHEMA, label_schema((15, 30)), plan)


@pytest.mark.parametrize(
    ("metadata_change", "message"),
    [
        ({b"demofml.label_set": b"future-v2"}, "label_set"),
        ({b"demofml.source_bar_set": b"trades-v1"}, "source bar sets"),
        ({b"demofml.source_bar_interval_minutes": b"15"}, "bar intervals"),
        ({b"demofml.max_quote_latency_minutes": b"4"}, "quote latency"),
        ({b"demofml.horizons_minutes": b"invalid"}, "leakage-window"),
    ],
)
def test_label_metadata_mismatches_are_rejected(
    metadata_change: dict[bytes, bytes], message: str
) -> None:
    labels = label_schema((15, 30, 60))
    changed = labels.with_metadata({**(labels.metadata or {}), **metadata_change})

    with pytest.raises(ValueError, match=message):
        validate_feature_label_schemas(FEATURE_SCHEMA, changed, _plan())


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "random-v1"}, "validation id"),
        ({"purge_minutes": 60}, "65-minute label window"),
        ({"strategy": "random"}, "expanding"),
        ({"interval_semantics": "closed"}, "half-open UTC"),
        ({"feature_set": ""}, "must be identified"),
        ({"step_months": 0}, "months must be positive"),
        ({"step_months": 1, "validation_window_months": 2}, "cannot overlap"),
        ({"max_horizon_minutes": 0}, "horizon must be positive"),
        (
            {"train_start": datetime(2022, 1, 1, tzinfo=UTC)},
            "initial training interval",
        ),
        (
            {"first_validation_start": datetime(2025, 1, 1, tzinfo=UTC)},
            "development validation interval",
        ),
        (
            {"development_end_exclusive": datetime(2024, 12, 1, tzinfo=UTC)},
            "locked test starts",
        ),
        (
            {"locked_test_end_exclusive": datetime(2025, 1, 1, tzinfo=UTC)},
            "locked test interval",
        ),
        (
            {
                "development_end_exclusive": datetime(2022, 1, 1, 0, 30, tzinfo=UTC),
                "locked_test_start": datetime(2022, 1, 1, 0, 30, tzinfo=UTC),
            },
            "no development validation decisions",
        ),
        (
            {"locked_test_end_exclusive": datetime(2025, 1, 1, 0, 30, tzinfo=UTC)},
            "no locked test decisions",
        ),
    ],
)
def test_plan_rejects_unsafe_parameters(change: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_plan(), **change)


def test_decision_times_must_be_ordered_utc() -> None:
    fold = _plan().folds()[0]
    timestamp = datetime(2022, 1, 1, tzinfo=UTC)
    with pytest.raises(ValueError, match="strictly ordered"):
        select_fold_rows([timestamp, timestamp], fold)
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        select_fold_rows([timestamp.replace(tzinfo=None)], fold)


def test_monthly_boundaries_reject_missing_calendar_day() -> None:
    plan = replace(
        _plan(),
        first_validation_start=datetime(2022, 1, 31, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="preserve the day"):
        plan.folds()


def test_config_loader_rejects_missing_and_malformed_values(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not a file"):
        load_validation_plan(tmp_path / "missing.toml")

    original = CONFIG.read_text()
    missing = tmp_path / "missing-field.toml"
    missing.write_text(original.replace('id = "purged-walk-forward-v1"\n', ""))
    with pytest.raises(ValueError, match="invalid validation config field"):
        load_validation_plan(missing)

    wrong_type = tmp_path / "wrong-type.toml"
    wrong_type.write_text(
        original.replace(
            'train_start = "2018-01-01T00:00:00Z"',
            "train_start = 2018-01-01T00:00:00Z",
        )
    )
    with pytest.raises(ValueError, match="ISO-8601 string"):
        load_validation_plan(wrong_type)

    malformed = tmp_path / "malformed.toml"
    malformed.write_text(original.replace("2018-01-01T00:00:00Z", "not-a-date"))
    with pytest.raises(ValueError, match="valid ISO-8601"):
        load_validation_plan(malformed)


def test_manifest_is_deterministic_and_built_atomically(tmp_path: Path) -> None:
    output = tmp_path / "validation.json"

    expected = validation_manifest(_plan())
    result = build_validation_manifest(CONFIG, output)

    assert result.fold_count == 36
    assert result.validation_set == VALIDATION_SET_ID
    assert json.loads(output.read_text()) == expected
    assert not list(tmp_path.glob("*.partial"))
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        build_validation_manifest(CONFIG, output)


def test_schema_validation_rejects_missing_leakage_metadata() -> None:
    metadata = dict(label_schema((15, 30, 60)).metadata or {})
    del metadata[b"demofml.max_quote_latency_minutes"]
    incomplete_labels = pa.schema(label_schema((15, 30, 60)), metadata=metadata)

    with pytest.raises(ValueError, match="leakage-window metadata"):
        validate_feature_label_schemas(FEATURE_SCHEMA, incomplete_labels, _plan())

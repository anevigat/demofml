from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from demofml.evaluation.portfolio import PORTFOLIO_HORIZONS, PORTFOLIO_SYMBOLS
from demofml.models.baseline import load_baseline_config, prediction_schema
from demofml.reporting.screen import (
    evaluate_microstructure_screen,
    load_screen_acceptance_config,
)
from demofml.validation.splits import load_validation_plan

PROJECT_ROOT = Path(__file__).parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs/experiments/baseline-ridge-v3.toml"
VALIDATION_CONFIG = (
    PROJECT_ROOT / "configs/experiments/causal-v2-screen-2021-v1.toml"
)
ACCEPTANCE_CONFIG = (
    PROJECT_ROOT / "configs/experiments/microstructure-screen-acceptance-v1.toml"
)


def _prediction_tables() -> list[pa.Table]:
    schema = prediction_schema(load_baseline_config(MODEL_CONFIG))
    start = datetime(2021, 1, 4, tzinfo=UTC)
    tables: list[pa.Table] = []
    for symbol in PORTFOLIO_SYMBOLS:
        rows: list[dict[str, object]] = []
        for decision_index in range(100):
            decision_time = start + timedelta(minutes=5 * decision_index)
            for horizon in PORTFOLIO_HORIZONS:
                rows.append(
                    {
                        "model_set": "baseline-ridge-v3",
                        "validation_set": "causal-v2-screen-2021-v1",
                        "fold_id": "wf-2021-01",
                        "symbol": symbol,
                        "decision_time": decision_time,
                        "entry_time": decision_time + timedelta(seconds=1),
                        "exit_time": decision_time
                        + timedelta(minutes=horizon, seconds=1),
                        "horizon_minutes": horizon,
                        "predicted_long_return": 0.001,
                        "predicted_short_return": -0.001,
                        "action": "long",
                        "realized_return": 0.001,
                    }
                )
        tables.append(pa.Table.from_pylist(rows, schema=schema))
    return tables


def _evaluate(tables: list[pa.Table]) -> dict[str, Any]:
    return evaluate_microstructure_screen(
        tables,
        load_screen_acceptance_config(ACCEPTANCE_CONFIG),
        load_validation_plan(VALIDATION_CONFIG),
    )


def test_screen_accepts_only_when_all_three_gates_pass() -> None:
    report = _evaluate(_prediction_tables())

    assert report["accepted"] is True
    assert report["promotion_authorized"] is False
    assert report["report_scope"] == "scientific_result_only"
    assert report["locked_test_policy"] == "forbidden"
    assert report["next_action"] == (
        "freeze_causal_v2_before_one_2022_2024_walk_forward"
    )


def test_screen_rejects_five_positive_symbols_and_one_short_cell() -> None:
    tables = _prediction_tables()
    for index in range(3):
        rows = tables[index].to_pylist()
        for row in rows:
            if row["horizon_minutes"] == 15:
                row["realized_return"] = -0.001
        tables[index] = pa.Table.from_pylist(rows, schema=tables[index].schema)
    report = _evaluate(tables)

    assert report["accepted"] is False
    assert report["promotion_authorized"] is False
    assert report["gates"]["positive_mean_all_horizons"] is True
    assert report["gates"]["minimum_positive_symbols_all_horizons"] is False

    tables = _prediction_tables()
    rows = tables[0].to_pylist()
    rows.pop(0)
    tables[0] = pa.Table.from_pylist(rows, schema=tables[0].schema)
    report = _evaluate(tables)
    assert report["accepted"] is False
    assert report["gates"]["minimum_trades_all_cells"] is False


def test_screen_rejects_duplicate_symbol_artifacts_and_keys() -> None:
    tables = _prediction_tables()
    tables[-1] = tables[0]
    with pytest.raises(ValueError, match="canonical eight symbols"):
        _evaluate(tables)

    tables = _prediction_tables()
    rows = tables[0].to_pylist()
    rows.insert(1, dict(rows[0]))
    tables[0] = pa.Table.from_pylist(rows, schema=tables[0].schema)
    with pytest.raises(ValueError, match="unique and ordered"):
        _evaluate(tables)

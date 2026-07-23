import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from demofml.evaluation.portfolio import (
    PORTFOLIO_HORIZONS,
    PORTFOLIO_SET_ID,
    PORTFOLIO_SYMBOLS,
    PortfolioConfig,
    load_portfolio_config,
    simulate_portfolio,
)
from demofml.models.baseline import load_baseline_config, prediction_schema
from demofml.reporting.portfolio import (
    portfolio_report,
    run_portfolio_evaluation,
)

PROJECT_ROOT = Path(__file__).parents[2]
PORTFOLIO_CONFIG = PROJECT_ROOT / "configs/experiments/portfolio-v1.toml"
MODEL_CONFIG = PROJECT_ROOT / "configs/experiments/baseline-ridge-v1.toml"
VALIDATION_CONFIG = PROJECT_ROOT / "configs/experiments/purged-walk-forward-v1.toml"
LOCKED_TEST_START = datetime(2025, 1, 1, tzinfo=UTC)


def _config() -> PortfolioConfig:
    return replace(
        load_portfolio_config(PORTFOLIO_CONFIG),
        annualization_periods=1,
        volatility_lookback_periods=4,
        volatility_min_observations=2,
    )


def _prediction_tables(
    decision_count: int = 3,
    realized_return: float = 0.01,
) -> list[pa.Table]:
    schema = prediction_schema(load_baseline_config(MODEL_CONFIG))
    start = datetime(2022, 1, 3, tzinfo=UTC)
    tables: list[pa.Table] = []
    for symbol in PORTFOLIO_SYMBOLS:
        rows: list[dict[str, object]] = []
        for decision_index in range(decision_count):
            decision_time = start + timedelta(minutes=5 * decision_index)
            for horizon in PORTFOLIO_HORIZONS:
                rows.append(
                    {
                        "model_set": "baseline-ridge-v1",
                        "validation_set": "purged-walk-forward-v1",
                        "fold_id": "wf-2022-01",
                        "symbol": symbol,
                        "decision_time": decision_time,
                        "entry_time": decision_time + timedelta(seconds=1),
                        "exit_time": decision_time
                        + timedelta(minutes=horizon, seconds=1),
                        "horizon_minutes": horizon,
                        "predicted_long_return": realized_return,
                        "predicted_short_return": -realized_return,
                        "action": "long",
                        "realized_return": realized_return,
                    }
                )
        tables.append(pa.Table.from_pylist(rows, schema=schema))
    return tables


def _replace_rows(
    tables: list[pa.Table],
    transform: Any,
) -> list[pa.Table]:
    changed: list[pa.Table] = []
    for table in tables:
        rows = table.to_pylist()
        transform(rows)
        changed.append(pa.Table.from_pylist(rows, schema=table.schema))
    return changed


def test_overlap_adjusted_weights_sum_to_one_at_steady_state() -> None:
    config = _config()

    scheduled_gross = sum(
        config.lot_weight(horizon) * (horizon // config.decision_interval_minutes)
        for _symbol in config.symbols
        for horizon in config.horizons_minutes
    )

    assert scheduled_gross == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unsupported portfolio horizon"):
        config.lot_weight(45)


def test_independent_overlapping_lots_settle_on_actual_exit_time() -> None:
    simulation = simulate_portfolio(_prediction_tables(), _config(), LOCKED_TEST_START)

    assert simulation.ledger.num_rows == 72
    assert simulation.maximum_active_positions == 72
    assert simulation.maximum_gross_leverage < 1.0
    assert simulation.equity.column("equity_usd")[-1].as_py() > 100_000.0
    assert min(simulation.ledger.column("entry_time").to_pylist()) == datetime(
        2022, 1, 3, 0, 0, 1, tzinfo=UTC
    )
    assert min(simulation.ledger.column("exit_time").to_pylist()) == datetime(
        2022, 1, 3, 0, 15, 1, tzinfo=UTC
    )


def test_future_return_cannot_change_earlier_position_size() -> None:
    original_tables = _prediction_tables(decision_count=8, realized_return=0.001)
    changed_tables = _replace_rows(
        original_tables,
        lambda rows: rows[-1].update({"realized_return": 0.5}),
    )

    original = simulate_portfolio(original_tables, _config(), LOCKED_TEST_START)
    changed = simulate_portfolio(changed_tables, _config(), LOCKED_TEST_START)
    cutoff = datetime(2022, 1, 3, 0, 35, tzinfo=UTC)
    original_rows = [
        row for row in original.ledger.to_pylist() if row["decision_time"] < cutoff
    ]
    changed_rows = [
        row for row in changed.ledger.to_pylist() if row["decision_time"] < cutoff
    ]

    assert [row["notional_usd"] for row in changed_rows] == [
        row["notional_usd"] for row in original_rows
    ]


def test_market_gap_does_not_create_wall_clock_zero_returns() -> None:
    tables = _prediction_tables(decision_count=2, realized_return=0.001)

    def add_weekend_gap(rows: list[dict[str, object]]) -> None:
        for row in rows:
            if row["decision_time"] == datetime(2022, 1, 3, 0, 5, tzinfo=UTC):
                shift = timedelta(days=3)
                for name in ("decision_time", "entry_time", "exit_time"):
                    value = row[name]
                    assert isinstance(value, datetime)
                    row[name] = value + shift

    changed = _replace_rows(tables, add_weekend_gap)
    simulation = simulate_portfolio(changed, _config(), LOCKED_TEST_START)

    assert len(simulation.period_returns) < 10


def test_zero_realized_volatility_uses_maximum_leverage_after_warmup() -> None:
    simulation = simulate_portfolio(
        _prediction_tables(decision_count=8, realized_return=0.0),
        _config(),
        LOCKED_TEST_START,
    )

    leverages = set(simulation.ledger.column("risk_leverage").to_pylist())
    assert 1.0 in leverages
    assert 2.0 in leverages


def test_drawdown_halts_new_positions_but_settles_open_lots() -> None:
    simulation = simulate_portfolio(
        _prediction_tables(decision_count=8, realized_return=-1.0),
        _config(),
        LOCKED_TEST_START,
    )
    report = portfolio_report(simulation, _config())

    assert report["drawdown_halt_triggered"] is True
    assert simulation.suppressed_signals > 0
    assert simulation.ledger.num_rows > 0
    assert max(simulation.ledger.column("decision_time").to_pylist()) < datetime(
        2022, 1, 3, 0, 20, tzinfo=UTC
    )
    assert simulation.equity.column("active_positions")[-1].as_py() == 0


def test_loss_strictly_before_delayed_entry_cancels_pending_lot() -> None:
    tables = _prediction_tables(decision_count=8, realized_return=-1.0)

    def delay_entry(rows: list[dict[str, object]]) -> None:
        target = datetime(2022, 1, 3, 0, 15, tzinfo=UTC)
        for row in rows:
            if row["decision_time"] == target:
                row["entry_time"] = target + timedelta(seconds=2)

    changed = _replace_rows(tables, delay_entry)
    simulation = simulate_portfolio(changed, _config(), LOCKED_TEST_START)

    assert max(simulation.ledger.column("decision_time").to_pylist()) < datetime(
        2022, 1, 3, 0, 15, tzinfo=UTC
    )
    assert simulation.suppressed_signals > 0


def test_maximum_gross_leverage_includes_post_loss_spikes() -> None:
    config = replace(
        _config(),
        warmup_leverage=2.0,
        maximum_drawdown=0.99,
    )
    simulation = simulate_portfolio(
        _prediction_tables(decision_count=4, realized_return=-1.0),
        config,
        LOCKED_TEST_START,
    )
    equity_rows = simulation.equity.to_pylist()
    observed = max(
        row["gross_notional_usd"] / row["equity_usd"]
        for row in equity_rows
        if row["equity_usd"] > 0.0
    )

    assert simulation.maximum_gross_leverage == pytest.approx(observed)
    assert simulation.maximum_gross_leverage > 1.5


def test_report_contains_risk_metrics_and_attribution() -> None:
    config = _config()
    simulation = simulate_portfolio(
        _prediction_tables(decision_count=5), config, LOCKED_TEST_START
    )

    report = portfolio_report(simulation, config)

    assert report["portfolio_set"] == PORTFOLIO_SET_ID
    assert report["development_only"] is True
    assert report["trades"] == 120
    assert len(report["attribution"]["symbols"]) == 8
    assert len(report["attribution"]["horizons"]) == 3
    assert len(report["attribution"]["folds"]) == 1
    assert report["always_flat_comparator"]["total_return"] == 0.0
    assert report["accounting"] == "normalized_executable_return"


def test_predictions_require_exact_provenance_universe_and_keys() -> None:
    tables = _prediction_tables()
    with pytest.raises(ValueError, match="at least one"):
        simulate_portfolio([], _config(), LOCKED_TEST_START)
    with pytest.raises(ValueError, match="canonical symbol universe"):
        simulate_portfolio(tables[:-1], _config(), LOCKED_TEST_START)

    duplicate = pa.concat_tables([tables[0], tables[0]])
    with pytest.raises(ValueError, match="duplicate"):
        simulate_portfolio([duplicate, *tables[1:]], _config(), LOCKED_TEST_START)

    wrong_metadata = tables[0].replace_schema_metadata({})
    with pytest.raises(ValueError, match="metadata mismatch"):
        simulate_portfolio([wrong_metadata, *tables[1:]], _config(), LOCKED_TEST_START)

    def wrong_row_provenance(rows: list[dict[str, object]]) -> None:
        rows[0]["model_set"] = "other-model"

    changed = _replace_rows(tables, wrong_row_provenance)
    with pytest.raises(ValueError, match="row provenance"):
        simulate_portfolio(changed, _config(), LOCKED_TEST_START)


def test_prelock_decision_cannot_exit_inside_locked_test() -> None:
    tables = _prediction_tables()

    def cross_lock(rows: list[dict[str, object]]) -> None:
        row = rows[0]
        row["decision_time"] = LOCKED_TEST_START - timedelta(minutes=15)
        row["entry_time"] = LOCKED_TEST_START - timedelta(minutes=15)
        row["exit_time"] = LOCKED_TEST_START

    changed = _replace_rows(tables, cross_lock)
    with pytest.raises(ValueError, match="locked-test"):
        simulate_portfolio(changed, _config(), LOCKED_TEST_START)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("decision_time", LOCKED_TEST_START, "locked-test"),
        ("entry_time", datetime(2022, 1, 3, 0, 6, tzinfo=UTC), "entry_time"),
        ("exit_time", datetime(2022, 1, 3, 0, 10, tzinfo=UTC), "exit_time"),
        ("action", "hold", "horizon or action"),
        ("realized_return", float("inf"), "must be finite"),
    ],
)
def test_predictions_reject_invalid_execution_rows(
    field: str, value: object, message: str
) -> None:
    tables = _prediction_tables()

    def change_first(rows: list[dict[str, object]]) -> None:
        rows[0][field] = value

    changed = _replace_rows(tables, change_first)
    with pytest.raises(ValueError, match=message):
        simulate_portfolio(changed, _config(), LOCKED_TEST_START)


def test_flat_predictions_require_zero_return() -> None:
    tables = _prediction_tables()

    def invalid_flat(rows: list[dict[str, object]]) -> None:
        rows[0]["action"] = "flat"

    changed = _replace_rows(tables, invalid_flat)
    with pytest.raises(ValueError, match="flat predictions"):
        simulate_portfolio(changed, _config(), LOCKED_TEST_START)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"id": "other"}, "portfolio id"),
        ({"prediction_set": "v1"}, "provenance"),
        ({"symbols": PORTFOLIO_SYMBOLS[:-1]}, "eight symbols"),
        ({"horizons_minutes": (15, 30)}, "15, 30, and 60"),
        ({"initial_capital_usd": 0.0}, "capital"),
        ({"decision_interval_minutes": 15}, "five-minute"),
        ({"overlap_policy": "net"}, "policy is not supported"),
        ({"annualization_periods": 0}, "must be positive"),
        ({"volatility_min_observations": 1}, "window is invalid"),
        ({"target_annual_volatility": 1.0}, "below one"),
        ({"warmup_leverage": 3.0}, "cannot exceed"),
        ({"maximum_drawdown": 1.0}, "below one"),
    ],
)
def test_portfolio_config_rejects_unsafe_values(
    change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_config(), **change)


def test_portfolio_config_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not a file"):
        load_portfolio_config(tmp_path / "missing.toml")

    malformed = tmp_path / "malformed.toml"
    malformed.write_text(
        PORTFOLIO_CONFIG.read_text().replace("initial_capital_usd = 100000.0\n", "")
    )
    with pytest.raises(ValueError, match="invalid portfolio config field"):
        load_portfolio_config(malformed)


def test_portfolio_artifacts_are_published_atomically(tmp_path: Path) -> None:
    paths: list[Path] = []
    for symbol, table in zip(PORTFOLIO_SYMBOLS, _prediction_tables(), strict=True):
        path = tmp_path / f"{symbol}.parquet"
        pq.write_table(table, path)
        paths.append(path)
    output = tmp_path / "portfolio"

    result = run_portfolio_evaluation(
        paths,
        PORTFOLIO_CONFIG,
        VALIDATION_CONFIG,
        output,
    )

    assert result.trades == 72
    assert result.equity_events > 1
    assert result.final_equity_usd > 100_000.0
    assert pq.read_table(output / "ledger.parquet").num_rows == 72
    assert pq.read_table(output / "equity.parquet").num_rows > 1
    assert json.loads((output / "metrics.json").read_text())["format_version"] == 1
    assert not list(tmp_path.glob("*.partial"))
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        run_portfolio_evaluation(
            paths,
            PORTFOLIO_CONFIG,
            VALIDATION_CONFIG,
            output,
        )

from datetime import UTC, datetime, timedelta

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from demofml.bars.prospective import PROSPECTIVE_BAR_SCHEMA
from demofml.calendars.prospective_fx import expected_decision_boundaries
from demofml.features.cross_pair import (
    CANDIDATE_FEATURE_SCHEMA,
    CROSS_PAIR_COLUMNS,
    PAIRS,
    EightPairSynchronizer,
    PairedCrossPairFeatureBuilder,
    solve_cross_pair_factor,
)


def _bar(
    symbol: str,
    end: datetime,
    close: float,
    *,
    available_delay_seconds: int = 1,
) -> pa.Table:
    start = end - timedelta(minutes=5)
    finalized = end + timedelta(milliseconds=20)
    row = {
        "symbol": symbol,
        "bar_start": start,
        "bar_end": end,
        "first_tick": start + timedelta(seconds=1),
        "last_tick": end - timedelta(seconds=1),
        "bid_open": close - 0.0001,
        "bid_high": close - 0.0001,
        "bid_low": close - 0.0001,
        "bid_close": close - 0.0001,
        "ask_open": close + 0.0001,
        "ask_high": close + 0.0001,
        "ask_low": close + 0.0001,
        "ask_close": close + 0.0001,
        "mid_open": close,
        "mid_high": close,
        "mid_low": close,
        "mid_close": close,
        "spread_open": 0.0002,
        "spread_high": 0.0002,
        "spread_low": 0.0002,
        "spread_close": 0.0002,
        "spread_mean": 0.0002,
        "quote_count": 2,
        "staleness_ns": 1_000_000_000,
        "first_received_at": start + timedelta(seconds=1, milliseconds=10),
        "last_received_at": end - timedelta(seconds=1, milliseconds=-10),
        "watermark_provider_timestamp": end,
        "watermark_ingest_sequence": 3,
        "bar_finalized_at": finalized,
        "feature_available_at": finalized
        + timedelta(seconds=available_delay_seconds),
        "first_ingest_sequence": 1,
        "last_ingest_sequence": 2,
    }
    return pa.Table.from_pylist([row], schema=PROSPECTIVE_BAR_SCHEMA)


def _push_section(
    synchronizer: EightPairSynchronizer,
    decision_time: datetime,
    closes: dict[str, float],
    omitted: str | None = None,
) -> None:
    for symbol in PAIRS:
        if symbol != omitted:
            synchronizer.push(_bar(symbol, decision_time, closes[symbol]))


def test_calendar_freezes_weekly_cutoff_and_dst_conversion() -> None:
    start = datetime(2026, 9, 6, tzinfo=UTC)
    end = start + timedelta(days=6)
    boundaries = expected_decision_boundaries(start, end)

    assert len(boundaries) == 1_427
    assert boundaries[0] == datetime(2026, 9, 6, 21, 5, tzinfo=UTC)
    assert boundaries[-1] == datetime(2026, 9, 11, 19, 55, tzinfo=UTC)

    dst_end = expected_decision_boundaries(
        datetime(2026, 11, 1, 21, 55, tzinfo=UTC),
        datetime(2026, 11, 1, 22, 10, tzinfo=UTC),
    )
    assert dst_end == (datetime(2026, 11, 1, 22, 5, tzinfo=UTC),)

    dst_start = expected_decision_boundaries(
        datetime(2026, 3, 8, 20, 55, tzinfo=UTC),
        datetime(2026, 3, 8, 21, 10, tzinfo=UTC),
    )
    assert dst_start == (datetime(2026, 3, 8, 21, 5, tzinfo=UTC),)

    winter_friday = expected_decision_boundaries(
        datetime(2026, 1, 9, 20, 55, tzinfo=UTC),
        datetime(2026, 1, 9, 21, 5, tzinfo=UTC),
    )
    assert winter_friday == (datetime(2026, 1, 9, 20, 55, tzinfo=UTC),)


def test_closed_form_factor_recovers_consistent_currency_strengths() -> None:
    pair_returns = {
        "AUDUSD": 0.01,
        "EURCHF": 0.01,
        "EURJPY": -0.02,
        "EURUSD": 0.04,
        "GBPJPY": -0.11,
        "GBPUSD": -0.05,
        "USDCAD": 0.02,
        "USDJPY": -0.06,
    }
    solution = solve_cross_pair_factor(dict(reversed(tuple(pair_returns.items()))))

    assert solution.strength("AUD") == pytest.approx(0.01)
    assert solution.strength("CAD") == pytest.approx(-0.02)
    assert solution.strength("CHF") == pytest.approx(0.03)
    assert solution.strength("EUR") == pytest.approx(0.04)
    assert solution.strength("GBP") == pytest.approx(-0.05)
    assert solution.strength("JPY") == pytest.approx(0.06)
    assert solution.strength("USD") == 0.0
    assert solution.residuals == pytest.approx((0.0,) * 8, abs=1e-15)
    assert solution.residual_dispersion == pytest.approx(0.0, abs=1e-15)


def test_closed_form_factor_golden_inconsistent_vector() -> None:
    values = (0.1, 0.2, -0.1, 0.3, 0.4, -0.2, 0.05, 0.15)
    solution = solve_cross_pair_factor(dict(zip(PAIRS, values, strict=True)))

    assert solution.strengths == pytest.approx(
        (0.1, -0.05, -0.1625, 0.0375, 0.0375, -0.125)
    )
    assert solution.residuals == pytest.approx(
        (0.0, 0.0, -0.2625, 0.2625, 0.2375, -0.2375, 0.0, 0.025)
    )


def test_synchronizer_and_paired_builder_reset_after_missing_section() -> None:
    decision = datetime(2026, 9, 1, 0, 5, tzinfo=UTC)
    closes = {symbol: 1.0 + index / 10.0 for index, symbol in enumerate(PAIRS)}
    synchronizer = EightPairSynchronizer()
    builder = PairedCrossPairFeatureBuilder()

    _push_section(synchronizer, decision, closes)
    first = builder.push(synchronizer.close(decision))
    assert not first.ready
    assert first.control.num_rows == 8
    assert first.candidate.schema == CANDIDATE_FEATURE_SCHEMA

    current_closes = closes
    latest = first
    for step in range(1, 4):
        boundary = decision + timedelta(minutes=5 * step)
        current_closes = {
            symbol: value * (1.0 + (index + 1) * 1e-5)
            for index, (symbol, value) in enumerate(current_closes.items())
        }
        _push_section(synchronizer, boundary, current_closes)
        latest = builder.push(synchronizer.close(boundary))

    assert latest.ready
    assert latest.candidate.num_rows == 8
    assert tuple(latest.candidate.column_names[-12:]) == CROSS_PAIR_COLUMNS
    assert all(
        value is not None
        for value in latest.candidate.column("pair_factor_residual_sum_3").to_pylist()
    )
    assert latest.candidate.column("pair_factor_residual_sum_12").null_count == 8

    missing_boundary = decision + timedelta(minutes=20)
    _push_section(
        synchronizer, missing_boundary, current_closes, omitted="USDJPY"
    )
    missing = builder.push(synchronizer.close(missing_boundary))
    assert missing.missing_symbols == ("USDJPY",)
    assert missing.control.num_rows == 0

    after_gap_boundary = decision + timedelta(minutes=25)
    _push_section(synchronizer, after_gap_boundary, current_closes)
    after_gap = builder.push(synchronizer.close(after_gap_boundary))
    assert not after_gap.ready
    assert after_gap.candidate.column("pair_factor_residual_1").null_count == 8
    control_rows = {
        row["symbol"]: row for row in after_gap.control.to_pylist()
    }
    assert control_rows["AUDUSD"]["elapsed_seconds"] == 300.0
    assert control_rows["AUDUSD"]["mid_return_1"] is not None
    assert control_rows["USDJPY"]["elapsed_seconds"] == 600.0
    assert control_rows["USDJPY"]["mid_return_1"] is None

    with pytest.raises(ValueError, match="closed boundary"):
        synchronizer.push(_bar("USDJPY", missing_boundary, 1.0))

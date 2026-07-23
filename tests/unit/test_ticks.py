from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from demofml.data.audit import audit_dataset
from demofml.data.ticks import (
    TickContractError,
    audit_parquet_file,
    audit_tick_table,
    canonicalize_ticks,
    validate_tick_schema,
)


def _ticks(
    timestamps: list[datetime],
    bids: list[float],
    asks: list[float],
    mids: list[float] | None = None,
    spreads: list[float] | None = None,
) -> pa.Table:
    return pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.timestamp("us", tz="UTC")),
            "bid": bids,
            "ask": asks,
            "mid": mids
            or [
                (bid + ask) / 2
                for bid, ask in zip(bids, asks, strict=True)
            ],
            "spread": spreads
            or [ask - bid for bid, ask in zip(bids, asks, strict=True)],
        }
    )


def test_tick_quality_detects_order_duplicates_and_quote_errors() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [
        start,
        start + timedelta(seconds=1),
        start + timedelta(seconds=1),
        start,
    ]
    table = _ticks(
        timestamps,
        bids=[1.0, 1.0, 1.0, 2.0],
        asks=[1.1, 1.1, 1.1, 1.0],
        mids=[1.05, 1.05, 1.05, 0.0],
        spreads=[0.1, 0.1, 0.1, 0.0],
    )

    report = audit_tick_table(table)

    assert report.rows == 4
    assert report.crossed_quotes == 1
    assert report.inconsistent_mid == 1
    assert report.inconsistent_spread == 1
    assert report.out_of_order == 1
    assert report.exact_duplicates == 1
    assert report.critical_violations == 5


def test_tick_contract_rejects_timestamp_without_utc() -> None:
    schema = pa.schema(
        [
            ("timestamp", pa.timestamp("us")),
            ("bid", pa.float64()),
            ("ask", pa.float64()),
            ("mid", pa.float64()),
            ("spread", pa.float64()),
        ]
    )

    with pytest.raises(TickContractError, match="UTC"):
        validate_tick_schema(schema)


def test_tick_quality_detects_non_adjacent_duplicate_at_same_timestamp() -> None:
    timestamp = datetime(2026, 1, 1, tzinfo=UTC)
    table = _ticks(
        [timestamp, timestamp, timestamp],
        bids=[1.0, 1.1, 1.0],
        asks=[1.2, 1.3, 1.2],
    )

    report = audit_tick_table(table)

    assert report.exact_duplicates == 1


def test_dataset_audit_preserves_order_state_between_files(tmp_path: Path) -> None:
    source = tmp_path / "cleaned_ticks"
    first = source / "EURUSD" / "2020" / "a.parquet"
    second = source / "EURUSD" / "2021" / "b.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    start = datetime(2026, 1, 1, tzinfo=UTC)
    pq.write_table(_ticks([start + timedelta(seconds=1)], [1.0], [1.1]), first)
    pq.write_table(_ticks([start], [1.0], [1.1]), second)

    report = audit_dataset(source, None)

    assert report["critical_violations"] == 1
    assert report["streams"]["EURUSD"]["out_of_order"] == 1


def test_tick_contract_rejects_columns_timestamp_and_price_types() -> None:
    valid = _ticks([datetime(2026, 1, 1, tzinfo=UTC)], [1.0], [1.1])
    with pytest.raises(TickContractError, match="Expected columns"):
        validate_tick_schema(valid.drop(["spread"]).schema)

    wrong_timestamp = valid.set_column(0, "timestamp", pa.array([1], pa.int64()))
    with pytest.raises(TickContractError, match="Arrow timestamp"):
        validate_tick_schema(wrong_timestamp.schema)

    wrong_bid = valid.set_column(1, "bid", pa.array([1.0], pa.float32()))
    with pytest.raises(TickContractError, match="bid must be float64"):
        validate_tick_schema(wrong_bid.schema)


def test_tick_audit_handles_empty_null_and_non_finite_values() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    valid = _ticks([start], [1.0], [1.1])
    assert audit_tick_table(valid.slice(0, 0)).rows == 0
    assert canonicalize_ticks(valid).schema.field("timestamp").type == pa.timestamp(
        "ns", tz="UTC"
    )

    invalid = pa.table(
        {
            "timestamp": pa.array([start, None], pa.timestamp("us", tz="UTC")),
            "bid": [0.0, float("nan")],
            "ask": [float("inf"), -1.0],
            "mid": [0.0, float("nan")],
            "spread": [float("inf"), -1.0],
        }
    )
    report = audit_tick_table(invalid)
    assert report.null_values == 1
    assert report.non_finite_values == 4
    assert report.non_positive_bid == 1
    assert report.non_positive_ask == 1


def test_parquet_audit_limit_and_cross_batch_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "ticks.parquet"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ticks = _ticks([start, start], [1.0, 1.0], [1.1, 1.1])
    pq.write_table(ticks, path, row_group_size=1)

    limited = audit_parquet_file(path, max_row_groups=1)
    complete = audit_parquet_file(path)

    assert limited.rows == 1
    assert complete.rows == 2
    assert complete.exact_duplicates == 1

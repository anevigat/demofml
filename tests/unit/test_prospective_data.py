from datetime import UTC, datetime, timedelta

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from demofml.bars.prospective import (
    PROSPECTIVE_BAR_SCHEMA,
    ProspectiveQuoteBarBuilder,
    project_quote_bars_v1,
    validate_prospective_bar_table,
)
from demofml.bars.quotes import QUOTE_BAR_SCHEMA
from demofml.data.prospective_ticks import (
    PROSPECTIVE_TICK_SCHEMA,
    audit_prospective_tick_table,
)


def _ticks(
    values: list[tuple[datetime, datetime, int]],
    symbol: str = "EURUSD",
) -> pa.Table:
    rows = [
        {
            "symbol": symbol,
            "provider_timestamp": provider,
            "received_at": received,
            "ingest_sequence": sequence,
            "bid": 1.0999,
            "ask": 1.1001,
            "mid": 1.1,
            "spread": 0.0002,
        }
        for provider, received, sequence in values
    ]
    return pa.Table.from_pylist(rows, schema=PROSPECTIVE_TICK_SCHEMA)


def test_prospective_tick_audit_preserves_receipt_and_provider_order() -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    valid = _ticks(
        [
            (start, start + timedelta(milliseconds=10), 1),
            (
                start + timedelta(seconds=1),
                start + timedelta(seconds=1, milliseconds=10),
                2,
            ),
        ]
    )
    report = audit_prospective_tick_table(valid)

    assert report.rows == 2
    assert report.critical_violations == 0
    assert report.max_delivery_latency_ns == 10_000_000

    reordered = _ticks(
        [
            (
                start + timedelta(seconds=2),
                start + timedelta(seconds=2, milliseconds=10),
                3,
            ),
            (
                start + timedelta(seconds=1),
                start + timedelta(seconds=2, milliseconds=20),
                4,
            ),
        ]
    )
    bad = audit_prospective_tick_table(reordered)
    assert bad.provider_out_of_order == 1
    assert bad.critical_violations == 1

    null_timestamp = valid.slice(0, 1).set_column(
        1,
        valid.schema.field("provider_timestamp"),
        pa.array([None], type=pa.timestamp("ns", tz="UTC")),
    )
    null_report = audit_prospective_tick_table(null_timestamp)
    assert null_report.null_values == 1
    assert null_report.critical_violations == 1


def test_bar_waits_for_received_watermark_and_preserves_availability() -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    watermark_received = start + timedelta(minutes=5, milliseconds=200)
    builder = ProspectiveQuoteBarBuilder("EURUSD")
    emitted = builder.push(
        _ticks(
            [
                (
                    start + timedelta(seconds=10),
                    start + timedelta(seconds=10, milliseconds=10),
                    1,
                ),
                (
                    start + timedelta(minutes=4, seconds=50),
                    start + timedelta(minutes=4, seconds=50, milliseconds=10),
                    2,
                ),
                (start + timedelta(minutes=5), watermark_received, 3),
            ]
        )
    )

    assert emitted.schema == PROSPECTIVE_BAR_SCHEMA
    assert emitted.num_rows == 1
    assert emitted.column("bar_end")[0].as_py() == start + timedelta(minutes=5)
    assert emitted.column("bar_finalized_at")[0].as_py() == watermark_received
    assert emitted.column("feature_available_at")[0].as_py() == (
        watermark_received + timedelta(seconds=1)
    )
    assert emitted.column("quote_count")[0].as_py() == 2
    assert project_quote_bars_v1(emitted).schema == QUOTE_BAR_SCHEMA

    invalid = emitted.set_column(
        emitted.schema.get_field_index("feature_available_at"),
        emitted.schema.field("feature_available_at"),
        pa.array(
            [watermark_received + timedelta(seconds=2)],
            type=pa.timestamp("ns", tz="UTC"),
        ),
    )
    with pytest.raises(ValueError, match="exactly one second"):
        validate_prospective_bar_table(invalid)


def test_late_tick_terminally_invalidates_its_provider_boundary() -> None:
    start = datetime(2026, 9, 1, tzinfo=UTC)
    builder = ProspectiveQuoteBarBuilder("EURUSD")
    finalized = builder.push(
        _ticks(
            [
                (
                    start + timedelta(seconds=10),
                    start + timedelta(seconds=10, milliseconds=10),
                    1,
                ),
                (
                    start + timedelta(minutes=5),
                    start + timedelta(minutes=5, milliseconds=20),
                    2,
                ),
            ]
        )
    )
    assert finalized.num_rows == 1

    with pytest.raises(ValueError, match="regressed behind the watermark"):
        builder.push(
            _ticks(
                [
                    (
                        start + timedelta(minutes=4, seconds=59),
                        start + timedelta(minutes=5, milliseconds=30),
                        3,
                    )
                ]
            )
        )
    assert builder.invalidated_boundaries == (start + timedelta(minutes=5),)
    assert builder.quality.provider_out_of_order == 1
    assert builder.quality.critical_violations == 1

    with pytest.raises(RuntimeError, match="terminally invalid"):
        builder.push(
            _ticks(
                [
                    (
                        start + timedelta(minutes=6),
                        start + timedelta(minutes=6, milliseconds=10),
                        4,
                    )
                ]
            )
        )

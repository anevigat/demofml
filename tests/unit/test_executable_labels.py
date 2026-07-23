from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from demofml.bars.quotes import aggregate_quote_bars
from demofml.labels.build import build_labels
from demofml.labels.executable import (
    LABEL_SET_ID,
    ExecutableLabelBuilder,
    generate_executable_labels,
)


def _bars(offsets_minutes: list[int], bids: list[float]) -> pa.Table:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [
        start + timedelta(minutes=offset, seconds=1) for offset in offsets_minutes
    ]
    asks = [bid + 0.01 for bid in bids]
    ticks = pa.table(
        {
            "timestamp": pa.array(timestamps, type=pa.timestamp("ns", tz="UTC")),
            "bid": bids,
            "ask": asks,
            "mid": [
                (bid + ask) / 2
                for bid, ask in zip(bids, asks, strict=True)
            ],
            "spread": [
                ask - bid for bid, ask in zip(bids, asks, strict=True)
            ],
        }
    )
    return aggregate_quote_bars(ticks, "EURUSD")


def test_labels_use_next_executable_bid_and_ask() -> None:
    bars = _bars(list(range(0, 70, 5)), [1.0 + index * 0.01 for index in range(14)])

    labels = generate_executable_labels(bars, horizons_minutes=(15, 30, 60))

    first = labels.slice(0, 1).to_pylist()[0]
    expected_long = 1.04 / 1.02 - 1.0
    expected_short = 1.0 - 1.05 / 1.01
    assert first["entry_bid"] == 1.01
    assert first["entry_ask"] == 1.02
    assert first["long_return_15m"] == expected_long
    assert first["short_return_15m"] == expected_short
    assert first["action_15m"] == "long"
    assert labels.column("long_return_60m")[-1].as_py() is None


def test_label_is_null_when_next_entry_is_after_horizon() -> None:
    bars = _bars([0, 120], [1.0, 1.1])

    labels = generate_executable_labels(bars, horizons_minutes=(15,))
    first = labels.slice(0, 1).to_pylist()[0]

    assert first["entry_time"] is None
    assert first["long_return_15m"] is None
    assert first["short_return_15m"] is None
    assert first["action_15m"] is None


def test_label_build_preserves_version_metadata(tmp_path: Path) -> None:
    source = tmp_path / "bars.parquet"
    output = tmp_path / "labels.parquet"
    pq.write_table(_bars(list(range(0, 70, 5)), [1.0] * 14), source)

    result = build_labels(source, output)
    metadata = pq.read_schema(output).metadata

    assert result.input_bars == 14
    assert result.output_rows == 14
    assert metadata is not None
    assert metadata[b"demofml.label_set"] == LABEL_SET_ID.encode()


def test_label_builder_is_partition_invariant() -> None:
    bars = _bars(list(range(0, 100, 5)), [1.0] * 20)
    expected = generate_executable_labels(bars)
    builder = ExecutableLabelBuilder()

    actual = pa.concat_tables(
        [builder.push(bars.slice(0, 7)), builder.push(bars.slice(7)), builder.finish()]
    )

    assert actual.equals(expected)


def test_labels_reject_horizons_inside_a_five_minute_bar() -> None:
    bars = _bars(list(range(0, 30, 5)), [1.0] * 6)

    with pytest.raises(ValueError, match="multiples of five"):
        generate_executable_labels(bars, horizons_minutes=(7,))


def test_label_metadata_records_action_threshold() -> None:
    bars = _bars(list(range(0, 30, 5)), [1.0] * 6)

    labels = generate_executable_labels(
        bars, horizons_minutes=(15,), minimum_return_bps=2.5
    )

    assert labels.schema.metadata is not None
    assert labels.schema.metadata[b"demofml.minimum_return_bps"] == b"2.5"

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from demofml.bars.build import build_quote_bars
from demofml.bars.quotes import QuoteBarBuilder, aggregate_quote_bars


def _ticks(timestamps: list[datetime], bids: list[float]) -> pa.Table:
    asks = [bid + 0.0002 for bid in bids]
    return pa.table(
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


def test_bar_boundary_is_half_open_and_has_no_lookahead() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ticks = _ticks(
        [
            start + timedelta(seconds=1),
            start + timedelta(minutes=4, seconds=59),
            start + timedelta(minutes=5),
        ],
        [1.0, 1.2, 2.0],
    )

    bars = aggregate_quote_bars(ticks, "EURUSD")

    assert bars.num_rows == 2
    assert bars.column("bar_end")[0].as_py() == start + timedelta(minutes=5)
    assert bars.column("bid_close")[0].as_py() == 1.2
    assert bars.column("quote_count")[0].as_py() == 2
    assert bars.column("staleness_ns")[0].as_py() == 1_000_000_000
    assert bars.column("bid_open")[1].as_py() == 2.0
    assert bars.column("quote_count")[1].as_py() == 1


def test_streaming_builder_matches_single_table_aggregation() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    first = _ticks(
        [start + timedelta(seconds=1), start + timedelta(minutes=2)],
        [1.0, 1.1],
    )
    second = _ticks(
        [start + timedelta(minutes=4), start + timedelta(minutes=5)],
        [1.2, 2.0],
    )
    expected = aggregate_quote_bars(pa.concat_tables([first, second]), "EURUSD")

    builder = QuoteBarBuilder("EURUSD")
    initial = builder.push(first)
    completed = builder.push(second)
    final = builder.finish()
    actual = pa.concat_tables([completed, final])

    assert initial.num_rows == 0
    assert actual.equals(expected)


def test_aggregation_rejects_null_ticks() -> None:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    ticks = pa.table(
        {
            "timestamp": pa.array(
                [start, None], type=pa.timestamp("ns", tz="UTC")
            ),
            "bid": [1.0, 1.0],
            "ask": [1.1, 1.1],
            "mid": [1.05, 1.05],
            "spread": [0.1, 0.1],
        }
    )

    with pytest.raises(ValueError, match="critical violations"):
        aggregate_quote_bars(ticks, "EURUSD")


def test_parquet_bar_build_is_streaming_and_atomic(tmp_path: Path) -> None:
    source = tmp_path / "ticks"
    source.mkdir()
    output = tmp_path / "bars.parquet"
    start = datetime(2026, 1, 1, tzinfo=UTC)
    pq.write_table(
        _ticks([start + timedelta(seconds=1)], [1.0]),
        source / "a.parquet",
    )
    pq.write_table(
        _ticks([start + timedelta(minutes=5)], [2.0]),
        source / "b.parquet",
    )

    result = build_quote_bars(source, output, "EURUSD")
    bars = pq.read_table(output)

    assert result.input_files == 2
    assert result.input_rows == 2
    assert result.output_bars == 2
    assert bars.num_rows == 2
    assert pq.read_metadata(output).num_row_groups == 1
    assert not list(tmp_path.glob(".*.partial"))

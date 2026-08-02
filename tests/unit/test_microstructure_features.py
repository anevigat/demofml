from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from demofml.bars.build import build_quote_bars
from demofml.bars.quotes import QUOTE_BAR_SCHEMA, aggregate_quote_bars
from demofml.bars.quotes_v2 import (
    QUOTE_BAR_V2_SCHEMA,
    QuoteBarV2Builder,
    aggregate_quote_bars_v2,
    validate_quote_bar_v2_schema,
)
from demofml.features.build import build_features
from demofml.features.causal import CausalFeatureBuilder
from demofml.features.causal_v2 import (
    FEATURE_SET_V2_ID,
    CausalV2FeatureBuilder,
)


def _ticks(
    timestamps: list[datetime], bids: list[float], asks: list[float]
) -> pa.Table:
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


def _bars(count: int = 15, gap_at: int | None = None) -> pa.Table:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    timestamps: list[datetime] = []
    bids: list[float] = []
    asks: list[float] = []
    for index in range(count):
        offset = 120 if gap_at is not None and index >= gap_at else 0
        bar_start = start + timedelta(minutes=5 * index + offset)
        timestamps.extend(
            [
                bar_start + timedelta(seconds=1),
                bar_start + timedelta(seconds=2),
                bar_start + timedelta(seconds=4),
            ]
        )
        base = 1.0 + index / 10_000
        bids.extend([base, base + 0.0001, base + 0.0001])
        asks.extend([base + 0.0002, base + 0.0002, base + 0.0003])
    return aggregate_quote_bars_v2(_ticks(timestamps, bids, asks), "EURUSD")


def test_quote_bars_v2_freeze_intrabar_transition_definitions() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    ticks = _ticks(
        [start + timedelta(seconds=value) for value in (1, 2, 4, 7)],
        [1.0, 1.1, 1.1, 1.2],
        [1.2, 1.2, 1.3, 1.3],
    )

    bars = aggregate_quote_bars_v2(ticks, "EURUSD")
    row = bars.to_pylist()[0]

    assert row["bid_update_count"] == 2
    assert row["ask_update_count"] == 1
    assert row["bid_ask_update_imbalance"] == pytest.approx(1 / 3)
    assert row["mid_uptick_count"] == 3
    assert row["mid_downtick_count"] == 0
    assert row["mid_tick_imbalance"] == 1.0
    assert row["spread_widening_count"] == 1
    assert row["spread_narrowing_count"] == 2
    assert row["spread_change_imbalance"] == pytest.approx(-1 / 3)
    assert row["interarrival_dispersion_seconds"] == pytest.approx(
        (2 / 3) ** 0.5
    )
    v1 = aggregate_quote_bars(ticks, "EURUSD")
    assert bars.select(QUOTE_BAR_SCHEMA.names).replace_schema_metadata(
        QUOTE_BAR_SCHEMA.metadata
    ).equals(v1)


def test_quote_bars_v2_are_partition_invariant_and_do_not_cross_boundaries() -> None:
    start = datetime(2021, 1, 1, tzinfo=UTC)
    ticks = _ticks(
        [
            start + timedelta(seconds=1),
            start + timedelta(minutes=4),
            start + timedelta(minutes=5),
            start + timedelta(minutes=6),
        ],
        [1.0, 1.1, 2.0, 2.1],
        [1.1, 1.2, 2.1, 2.2],
    )
    expected = aggregate_quote_bars_v2(ticks, "EURUSD")
    builder = QuoteBarV2Builder("EURUSD")
    actual = pa.concat_tables(
        [
            builder.push(ticks.slice(0, 3)),
            builder.push(ticks.slice(3)),
            builder.finish(),
        ]
    )

    assert actual.equals(expected)
    assert expected.column("mid_uptick_count").to_pylist() == [1, 1]
    with pytest.raises(ValueError, match="quote-bars-v2"):
        validate_quote_bar_v2_schema(QUOTE_BAR_SCHEMA)


def test_quote_bar_build_v2_applies_research_cutoff(tmp_path: Path) -> None:
    source = tmp_path / "ticks.parquet"
    output = tmp_path / "bars.parquet"
    cutoff = datetime(2022, 1, 1, tzinfo=UTC)
    pq.write_table(
        _ticks(
            [cutoff - timedelta(seconds=1), cutoff],
            [1.0, 2.0],
            [1.1, 2.1],
        ),
        source,
    )

    result = build_quote_bars(
        source,
        output,
        "EURUSD",
        bar_set="quote-bars-v2",
        end_exclusive=cutoff,
    )
    bars = pq.read_table(output)

    assert result.input_rows == 1
    assert bars.num_rows == 1
    assert bars.schema == QUOTE_BAR_V2_SCHEMA
    assert bars.column("last_tick")[0].as_py() < cutoff


def test_causal_v2_preserves_v1_and_has_fixed_warmups() -> None:
    bars = _bars()
    v2 = CausalV2FeatureBuilder("EURUSD").push(bars)
    projected = bars.select(QUOTE_BAR_SCHEMA.names).replace_schema_metadata(
        QUOTE_BAR_SCHEMA.metadata
    )
    v1 = CausalFeatureBuilder("EURUSD").push(projected)
    streaming_builder = CausalV2FeatureBuilder("EURUSD")
    streaming = pa.concat_tables(
        [
            streaming_builder.push(bars.slice(0, 7)),
            streaming_builder.push(bars.slice(7)),
        ]
    )

    assert streaming.equals(v2)
    assert v2.select(v1.column_names).replace_schema_metadata(
        v1.schema.metadata
    ).equals(v1)
    assert v2.column("bid_ask_update_imbalance_15m")[1].as_py() is None
    assert v2.column("bid_ask_update_imbalance_15m")[2].as_py() is not None
    assert v2.column("bid_ask_update_imbalance_60m")[10].as_py() is None
    assert v2.column("bid_ask_update_imbalance_60m")[11].as_py() is not None
    assert v2.schema.metadata is not None
    assert v2.schema.metadata[b"demofml.feature_set"] == FEATURE_SET_V2_ID.encode()


def test_causal_v2_rolling_imbalance_uses_summed_counts() -> None:
    bars = _bars(3)
    rows = bars.to_pylist()
    for row, bid, ask in zip(rows, (9, 0, 0), (1, 1, 1), strict=True):
        row["bid_update_count"] = bid
        row["ask_update_count"] = ask
        row["bid_ask_update_imbalance"] = (
            (bid - ask) / (bid + ask) if bid + ask else 0.0
        )
    changed = pa.Table.from_pylist(rows, schema=bars.schema)

    features = CausalV2FeatureBuilder("EURUSD").push(changed)

    assert features.column("bid_ask_update_imbalance_15m")[2].as_py() == 0.5


def test_feature_build_dispatches_causal_v2(tmp_path: Path) -> None:
    source = tmp_path / "bars.parquet"
    output = tmp_path / "features.parquet"
    pq.write_table(_bars(), source, row_group_size=4)

    result = build_features(
        source,
        output,
        "EURUSD",
        feature_set="causal-v2",
    )
    metadata = pq.read_schema(output).metadata

    assert result.input_bars == 15
    assert result.output_rows == 15
    assert metadata is not None
    assert metadata[b"demofml.feature_set"] == FEATURE_SET_V2_ID.encode()


def test_causal_v2_resets_microstructure_windows_after_gap() -> None:
    features = CausalV2FeatureBuilder("EURUSD").push(_bars(15, gap_at=12))

    assert features.column("bid_ask_update_imbalance_15m")[12].as_py() is None
    assert features.column("bid_ask_update_imbalance_60m")[12].as_py() is None

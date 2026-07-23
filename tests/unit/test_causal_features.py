from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.bars.quotes import aggregate_quote_bars
from demofml.features.build import build_features
from demofml.features.causal import FEATURE_SET_ID, CausalFeatureBuilder


def _bars(
    count: int,
    future_jump_at: int | None = None,
    gap_at: int | None = None,
) -> pa.Table:
    start = datetime(2026, 1, 1, tzinfo=UTC)
    timestamps = [
        start
        + timedelta(
            minutes=5 * index + (120 if gap_at is not None and index >= gap_at else 0),
            seconds=1,
        )
        for index in range(count)
    ]
    bids = [
        1.0
        + index / 10_000
        + (
            0.2
            if future_jump_at is not None and index >= future_jump_at
            else 0.0
        )
        for index in range(count)
    ]
    asks = [bid + 0.0002 for bid in bids]
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


def test_features_are_causal_and_stream_partition_invariant() -> None:
    bars = _bars(100)
    changed_future = _bars(100, future_jump_at=60)

    complete = CausalFeatureBuilder("EURUSD").push(bars)
    changed = CausalFeatureBuilder("EURUSD").push(changed_future)
    streaming_builder = CausalFeatureBuilder("EURUSD")
    streaming = pa.concat_tables(
        [
            streaming_builder.push(bars.slice(0, 37)),
            streaming_builder.push(bars.slice(37)),
        ]
    )

    assert complete.slice(0, 60).equals(changed.slice(0, 60))
    assert streaming.equals(complete)
    assert complete.column("mid_return_1")[0].as_py() is None
    assert complete.column("mid_return_1")[1].as_py() is not None
    assert complete.column("mid_return_12")[11].as_py() is None
    assert complete.column("mid_return_12")[12].as_py() is not None
    assert complete.column("realized_volatility_12")[12].as_py() is not None
    assert complete.column("spread_zscore_72")[70].as_py() is None
    assert complete.column("spread_zscore_72")[71].as_py() is not None


def test_feature_build_preserves_version_metadata(tmp_path: Path) -> None:
    source = tmp_path / "bars.parquet"
    output = tmp_path / "features.parquet"
    pq.write_table(_bars(20), source, row_group_size=7)

    result = build_features(source, output, "EURUSD")
    metadata = pq.read_schema(output).metadata

    assert result.input_bars == 20
    assert result.output_rows == 20
    assert metadata is not None
    assert metadata[b"demofml.feature_set"] == FEATURE_SET_ID.encode()


def test_feature_windows_reset_after_missing_bars() -> None:
    bars = _bars(20, gap_at=15)

    features = CausalFeatureBuilder("EURUSD").push(bars)

    assert features.column("elapsed_seconds")[15].as_py() == 7_500.0
    assert features.column("mid_return_1")[15].as_py() is None
    assert features.column("mid_return_12")[15].as_py() is None
    assert features.column("realized_volatility_12")[15].as_py() is None

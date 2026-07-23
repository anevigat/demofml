"""Causal streaming aggregation of bid/ask ticks into quote bars."""

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]

from demofml.data.ticks import (
    TickQualityReport,
    audit_canonical_tick_table,
    canonicalize_ticks,
)

BAR_TIMESTAMP = pa.timestamp("ns", tz="UTC")
QUOTE_BAR_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("bar_start", BAR_TIMESTAMP, nullable=False),
        pa.field("bar_end", BAR_TIMESTAMP, nullable=False),
        pa.field("first_tick", BAR_TIMESTAMP, nullable=False),
        pa.field("last_tick", BAR_TIMESTAMP, nullable=False),
        *[
            pa.field(f"{price}_{statistic}", pa.float64(), nullable=False)
            for price in ("bid", "ask", "mid")
            for statistic in ("open", "high", "low", "close")
        ],
        pa.field("spread_open", pa.float64(), nullable=False),
        pa.field("spread_high", pa.float64(), nullable=False),
        pa.field("spread_low", pa.float64(), nullable=False),
        pa.field("spread_close", pa.float64(), nullable=False),
        pa.field("spread_mean", pa.float64(), nullable=False),
        pa.field("quote_count", pa.int64(), nullable=False),
        pa.field("staleness_ns", pa.int64(), nullable=False),
    ]
)


def empty_quote_bars() -> pa.Table:
    """Return an empty table with the canonical quote-bar schema."""
    return pa.Table.from_batches([], schema=QUOTE_BAR_SCHEMA)


def _bucket_start(timestamp: pa.Array | pa.ChunkedArray, minutes: int) -> pa.Array:
    return pc.floor_temporal(timestamp, multiple=minutes, unit="minute")


def aggregate_quote_bars(
    ticks: pa.Table,
    symbol: str,
    interval_minutes: int = 5,
) -> pa.Table:
    """Aggregate sorted ticks into half-open bars labelled by their end time."""
    if interval_minutes <= 0:
        raise ValueError("interval_minutes must be positive")
    canonical = canonicalize_ticks(ticks)
    if canonical.num_rows == 0:
        return empty_quote_bars()
    quality = audit_canonical_tick_table(canonical)
    if quality.critical_violations:
        raise ValueError(
            f"ticks contain {quality.critical_violations} critical violations"
        )
    return _aggregate_canonical(canonical, symbol, interval_minutes)


def _aggregate_canonical(
    canonical: pa.Table,
    symbol: str,
    interval_minutes: int,
) -> pa.Table:
    """Aggregate canonical ticks that have already passed quality checks."""

    timestamp = canonical.column("timestamp")
    with_bucket = canonical.append_column(
        "bar_start", _bucket_start(timestamp, interval_minutes)
    )
    aggregations = [
        ("timestamp", "first"),
        ("timestamp", "last"),
        *[
            (price, statistic)
            for price in ("bid", "ask", "mid", "spread")
            for statistic in ("first", "max", "min", "last")
        ],
        ("spread", "mean"),
        ("timestamp", "count"),
    ]
    grouped = (
        with_bucket.group_by("bar_start", use_threads=False)
        .aggregate(aggregations)
        .sort_by("bar_start")
    )
    bar_start = grouped.column("bar_start")
    interval_ns = interval_minutes * 60 * 1_000_000_000
    bar_end = pc.add(bar_start, pa.scalar(interval_ns, type=pa.duration("ns")))
    last_tick = grouped.column("timestamp_last")
    staleness = pc.cast(pc.subtract(bar_end, last_tick), pa.int64())
    symbols = pa.array([symbol] * grouped.num_rows, type=pa.string())
    columns: list[pa.Array | pa.ChunkedArray] = [
        symbols,
        bar_start,
        bar_end,
        grouped.column("timestamp_first"),
        last_tick,
    ]
    for price in ("bid", "ask", "mid"):
        columns.extend(
            grouped.column(f"{price}_{statistic}")
            for statistic in ("first", "max", "min", "last")
        )
    columns.extend(
        [
            grouped.column("spread_first"),
            grouped.column("spread_max"),
            grouped.column("spread_min"),
            grouped.column("spread_last"),
            grouped.column("spread_mean"),
            grouped.column("timestamp_count"),
            staleness,
        ]
    )
    return pa.table(columns, schema=QUOTE_BAR_SCHEMA)


class QuoteBarBuilder:
    """Retain only the open final bar while consuming ordered tick batches."""

    def __init__(self, symbol: str, interval_minutes: int = 5) -> None:
        if interval_minutes <= 0:
            raise ValueError("interval_minutes must be positive")
        self._symbol = symbol
        self._interval_minutes = interval_minutes
        self._carry: pa.Table | None = None
        self._quality = TickQualityReport()

    def push(self, ticks: pa.Table) -> pa.Table:
        """Consume ticks and return bars that can no longer receive future ticks."""
        canonical = canonicalize_ticks(ticks)
        if canonical.num_rows == 0:
            return empty_quote_bars()
        previous_violations = self._quality.critical_violations
        audit_canonical_tick_table(canonical, self._quality)
        if self._quality.critical_violations != previous_violations:
            raise ValueError("tick batches contain critical quality violations")
        if self._carry is not None:
            canonical = pa.concat_tables([self._carry, canonical])
        timestamp = canonical.column("timestamp")

        buckets = _bucket_start(timestamp, self._interval_minutes)
        last_bucket = buckets[canonical.num_rows - 1]
        complete_mask = pc.less(buckets, last_bucket)
        complete = canonical.filter(complete_mask)
        self._carry = canonical.filter(pc.invert(complete_mask))
        if complete.num_rows == 0:
            return empty_quote_bars()
        return _aggregate_canonical(
            complete, self._symbol, self._interval_minutes
        )

    def finish(self) -> pa.Table:
        """Emit the final open bar after the input stream ends."""
        if self._carry is None or self._carry.num_rows == 0:
            return empty_quote_bars()
        final = _aggregate_canonical(
            self._carry, self._symbol, self._interval_minutes
        )
        self._carry = None
        return final

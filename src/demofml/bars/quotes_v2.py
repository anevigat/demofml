"""Causal quote bars with frozen intrabar microstructure aggregates."""

import math

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]

from demofml.bars.quotes import QUOTE_BAR_SCHEMA, _bucket_start
from demofml.data.ticks import (
    TickQualityReport,
    audit_canonical_tick_table,
    canonicalize_ticks,
)

QUOTE_BAR_V2_SET_ID = "quote-bars-v2"
_MICROSTRUCTURE_FIELDS = (
    pa.field("bid_update_count", pa.int64(), nullable=False),
    pa.field("ask_update_count", pa.int64(), nullable=False),
    pa.field("bid_ask_update_imbalance", pa.float64(), nullable=False),
    pa.field("mid_uptick_count", pa.int64(), nullable=False),
    pa.field("mid_downtick_count", pa.int64(), nullable=False),
    pa.field("mid_tick_imbalance", pa.float64(), nullable=False),
    pa.field("spread_widening_count", pa.int64(), nullable=False),
    pa.field("spread_narrowing_count", pa.int64(), nullable=False),
    pa.field("spread_change_imbalance", pa.float64(), nullable=False),
    pa.field("interarrival_dispersion_seconds", pa.float64(), nullable=False),
)
QUOTE_BAR_V2_SCHEMA = pa.schema(
    [*QUOTE_BAR_SCHEMA, *_MICROSTRUCTURE_FIELDS],
    metadata={
        b"demofml.bar_set": QUOTE_BAR_V2_SET_ID.encode(),
        b"demofml.interval_minutes": b"5",
        b"demofml.transition_scope": b"within_half_open_bar",
        b"demofml.zero_denominator_policy": b"zero",
        b"demofml.interarrival_dispersion": b"population_std_seconds",
        b"demofml.equal_timestamp_order": b"canonical_physical_order",
    },
)


def empty_quote_bars_v2() -> pa.Table:
    """Return an empty table with the quote-bars-v2 schema."""
    return pa.Table.from_batches([], schema=QUOTE_BAR_V2_SCHEMA)


def validate_quote_bar_v2_schema(schema: pa.Schema) -> None:
    """Require the exact immutable quote-bars-v2 physical contract."""
    if schema.names != QUOTE_BAR_V2_SCHEMA.names:
        raise ValueError("quote-bar columns do not match quote-bars-v2")
    for expected in QUOTE_BAR_V2_SCHEMA:
        actual = schema.field(expected.name)
        if actual.type != expected.type or actual.nullable != expected.nullable:
            raise ValueError(f"invalid quote-bars-v2 type for {expected.name}")
    if schema.metadata != QUOTE_BAR_V2_SCHEMA.metadata:
        raise ValueError("quote-bar metadata does not identify quote-bars-v2")


def _as_array(value: pa.Array | pa.ChunkedArray) -> pa.Array:
    return value.combine_chunks() if isinstance(value, pa.ChunkedArray) else value


def _transition_flag(
    same_bar: pa.Array,
    current: pa.Array,
    previous: pa.Array,
    operation: str,
) -> pa.Array:
    compared = getattr(pc, operation)(current, previous)
    return _as_array(pc.cast(pc.and_(same_bar, compared), pa.int64()))


def _imbalance(positive: int, negative: int) -> float:
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def _aggregate_canonical_v2(
    canonical: pa.Table,
    symbol: str,
    interval_minutes: int,
) -> pa.Table:
    canonical = canonical.combine_chunks()
    timestamp = _as_array(canonical.column("timestamp"))
    buckets = _as_array(_bucket_start(timestamp, interval_minutes))
    same_bar = pa.concat_arrays(
        [
            pa.array([False], type=pa.bool_()),
            _as_array(
                pc.equal(buckets.slice(1), buckets.slice(0, len(buckets) - 1))
            ),
        ]
    )

    transition_columns: dict[str, pa.Array] = {}
    for name, source, operation in (
        ("_bid_update", "bid", "not_equal"),
        ("_ask_update", "ask", "not_equal"),
        ("_mid_uptick", "mid", "greater"),
        ("_mid_downtick", "mid", "less"),
        ("_spread_widening", "spread", "greater"),
        ("_spread_narrowing", "spread", "less"),
    ):
        values = _as_array(canonical.column(source))
        previous = pa.concat_arrays(
            [values.slice(0, 1), values.slice(0, len(values) - 1)]
        )
        transition_columns[name] = _transition_flag(
            same_bar,
            values,
            previous,
            operation,
        )

    delta_ns = _as_array(
        pc.cast(
            pc.subtract(timestamp.slice(1), timestamp.slice(0, len(timestamp) - 1)),
            pa.int64(),
        )
    )
    interarrival = pa.concat_arrays(
        [
            pa.array([None], type=pa.float64()),
            _as_array(pc.divide(pc.cast(delta_ns, pa.float64()), 1_000_000_000.0)),
        ]
    )
    interarrival = _as_array(
        pc.if_else(same_bar, interarrival, pa.scalar(None, pa.float64()))
    )
    interarrival_squared = _as_array(pc.multiply(interarrival, interarrival))

    with_transitions = canonical.append_column("bar_start", buckets)
    for name, values in transition_columns.items():
        with_transitions = with_transitions.append_column(name, values)
    with_transitions = with_transitions.append_column("_interarrival", interarrival)
    with_transitions = with_transitions.append_column(
        "_interarrival_squared", interarrival_squared
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
        *((name, "sum") for name in transition_columns),
        ("_interarrival", "count"),
        ("_interarrival", "sum"),
        ("_interarrival_squared", "sum"),
    ]
    grouped = (
        with_transitions.group_by("bar_start", use_threads=False)
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

    metric_rows: list[
        tuple[int, int, float, int, int, float, int, int, float, float]
    ] = []
    metric_names = [
        *(f"{name}_sum" for name in transition_columns),
        "_interarrival_count",
        "_interarrival_sum",
        "_interarrival_squared_sum",
    ]
    for row in grouped.select(metric_names).to_pylist():
        bid_updates = int(row["_bid_update_sum"])
        ask_updates = int(row["_ask_update_sum"])
        upticks = int(row["_mid_uptick_sum"])
        downticks = int(row["_mid_downtick_sum"])
        widening = int(row["_spread_widening_sum"])
        narrowing = int(row["_spread_narrowing_sum"])
        count = int(row["_interarrival_count"])
        total = float(row["_interarrival_sum"] or 0.0)
        squared = float(row["_interarrival_squared_sum"] or 0.0)
        variance = max(squared / count - (total / count) ** 2, 0.0) if count else 0.0
        metric_rows.append(
            (
                bid_updates,
                ask_updates,
                _imbalance(bid_updates, ask_updates),
                upticks,
                downticks,
                _imbalance(upticks, downticks),
                widening,
                narrowing,
                _imbalance(widening, narrowing),
                math.sqrt(variance),
            )
        )
    for index, field in enumerate(_MICROSTRUCTURE_FIELDS):
        columns.append(
            pa.array([row[index] for row in metric_rows], type=field.type)
        )
    return pa.table(columns, schema=QUOTE_BAR_V2_SCHEMA)


def aggregate_quote_bars_v2(
    ticks: pa.Table,
    symbol: str,
    interval_minutes: int = 5,
) -> pa.Table:
    """Aggregate ticks into quote-bars-v2 without crossing bar boundaries."""
    if interval_minutes != 5:
        raise ValueError("quote-bars-v2 requires interval_minutes=5")
    canonical = canonicalize_ticks(ticks)
    if canonical.num_rows == 0:
        return empty_quote_bars_v2()
    quality = audit_canonical_tick_table(canonical)
    if quality.critical_violations:
        raise ValueError(
            f"ticks contain {quality.critical_violations} critical violations"
        )
    return _aggregate_canonical_v2(canonical, symbol, interval_minutes)


class QuoteBarV2Builder:
    """Stream quote-bars-v2 while retaining only the final open bar."""

    def __init__(self, symbol: str, interval_minutes: int = 5) -> None:
        if interval_minutes != 5:
            raise ValueError("quote-bars-v2 requires interval_minutes=5")
        self._symbol = symbol
        self._interval_minutes = interval_minutes
        self._carry: pa.Table | None = None
        self._quality = TickQualityReport()

    def push(self, ticks: pa.Table) -> pa.Table:
        """Consume ordered ticks and emit bars closed by a later tick."""
        canonical = canonicalize_ticks(ticks)
        if canonical.num_rows == 0:
            return empty_quote_bars_v2()
        previous_violations = self._quality.critical_violations
        audit_canonical_tick_table(canonical, self._quality)
        if self._quality.critical_violations != previous_violations:
            raise ValueError("tick batches contain critical quality violations")
        if self._carry is not None:
            canonical = pa.concat_tables([self._carry, canonical])
        buckets = _bucket_start(canonical.column("timestamp"), self._interval_minutes)
        last_bucket = buckets[canonical.num_rows - 1]
        complete_mask = pc.less(buckets, last_bucket)
        complete = canonical.filter(complete_mask)
        self._carry = canonical.filter(pc.invert(complete_mask))
        if complete.num_rows == 0:
            return empty_quote_bars_v2()
        return _aggregate_canonical_v2(
            complete, self._symbol, self._interval_minutes
        )

    def finish(self) -> pa.Table:
        """Emit the final open bar after all ticks have been consumed."""
        if self._carry is None or self._carry.num_rows == 0:
            return empty_quote_bars_v2()
        final = _aggregate_canonical_v2(
            self._carry, self._symbol, self._interval_minutes
        )
        self._carry = None
        return final

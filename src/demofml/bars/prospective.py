"""Watermark-finalized quote bars preserving prospective receipt causality."""

from datetime import datetime

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]

from demofml.bars.quotes import QUOTE_BAR_SCHEMA, _aggregate_canonical
from demofml.data.prospective_ticks import (
    ProspectiveTickQualityReport,
    audit_prospective_tick_table,
    canonicalize_prospective_ticks,
)
from demofml.data.ticks import CANONICAL_TIMESTAMP, PRICE_COLUMNS, TICK_COLUMNS

PROSPECTIVE_BAR_SET_ID = "prospective-quote-bars-v1"
PROSPECTIVE_BAR_SCHEMA = pa.schema(
    [
        *QUOTE_BAR_SCHEMA,
        pa.field("first_received_at", CANONICAL_TIMESTAMP, nullable=False),
        pa.field("last_received_at", CANONICAL_TIMESTAMP, nullable=False),
        pa.field("watermark_provider_timestamp", CANONICAL_TIMESTAMP, nullable=False),
        pa.field("watermark_ingest_sequence", pa.uint64(), nullable=False),
        pa.field("bar_finalized_at", CANONICAL_TIMESTAMP, nullable=False),
        pa.field("feature_available_at", CANONICAL_TIMESTAMP, nullable=False),
        pa.field("first_ingest_sequence", pa.uint64(), nullable=False),
        pa.field("last_ingest_sequence", pa.uint64(), nullable=False),
    ],
    metadata={
        b"demofml.bar_set": PROSPECTIVE_BAR_SET_ID.encode(),
        b"demofml.source_tick_set": b"prospective-ticks-v1",
        b"demofml.interval_minutes": b"5",
        b"demofml.finalization": b"next_received_quote_provider_time_watermark",
        b"demofml.compute_allowance_seconds": b"1",
    },
)
_INTERVAL_NS = 5 * 60 * 1_000_000_000
_COMPUTE_ALLOWANCE_NS = 1_000_000_000


def empty_prospective_bars() -> pa.Table:
    """Return an empty receipt-aware bar table."""
    return pa.Table.from_batches([], schema=PROSPECTIVE_BAR_SCHEMA)


def validate_prospective_bar_schema(schema: pa.Schema) -> None:
    """Validate the complete immutable prospective bar contract."""
    if schema.names != PROSPECTIVE_BAR_SCHEMA.names:
        raise ValueError("bar columns do not match prospective-quote-bars-v1")
    for expected in PROSPECTIVE_BAR_SCHEMA:
        actual = schema.field(expected.name)
        if actual.type != expected.type or actual.nullable != expected.nullable:
            raise ValueError(f"invalid prospective bar field {expected.name}")
    metadata = schema.metadata or {}
    for key, value in (PROSPECTIVE_BAR_SCHEMA.metadata or {}).items():
        if metadata.get(key) != value:
            raise ValueError(f"prospective bar metadata mismatch for {key.decode()}")


def validate_prospective_bar_table(bars: pa.Table) -> None:
    """Validate timestamp, sequence, and watermark relationships."""
    validate_prospective_bar_schema(bars.schema)
    timestamp_names = (
        "bar_start",
        "bar_end",
        "first_tick",
        "last_tick",
        "first_received_at",
        "last_received_at",
        "watermark_provider_timestamp",
        "bar_finalized_at",
        "feature_available_at",
    )
    timestamps = {
        name: pc.cast(bars.column(name), pa.int64()).to_pylist()
        for name in timestamp_names
    }
    first_sequence = bars.column("first_ingest_sequence").to_pylist()
    last_sequence = bars.column("last_ingest_sequence").to_pylist()
    watermark_sequence = bars.column("watermark_ingest_sequence").to_pylist()
    quote_count = bars.column("quote_count").to_pylist()
    staleness = bars.column("staleness_ns").to_pylist()
    for index in range(bars.num_rows):
        start = int(timestamps["bar_start"][index])
        end = int(timestamps["bar_end"][index])
        first_tick = int(timestamps["first_tick"][index])
        last_tick = int(timestamps["last_tick"][index])
        first_received = int(timestamps["first_received_at"][index])
        last_received = int(timestamps["last_received_at"][index])
        watermark_provider = int(timestamps["watermark_provider_timestamp"][index])
        finalized = int(timestamps["bar_finalized_at"][index])
        available = int(timestamps["feature_available_at"][index])
        if end - start != _INTERVAL_NS or end % _INTERVAL_NS:
            raise ValueError("prospective bars must span five minutes")
        if not (start <= first_tick <= last_tick < end):
            raise ValueError("provider ticks must remain inside the half-open bar")
        if not (first_received <= last_received <= finalized):
            raise ValueError("receipt timestamps must precede bar finalization")
        if watermark_provider < end:
            raise ValueError("watermark provider time must reach bar_end")
        if available < end:
            raise ValueError("features cannot be available before bar_end")
        if available - finalized != _COMPUTE_ALLOWANCE_NS:
            raise ValueError("feature availability must include exactly one second")
        if int(first_sequence[index]) > int(last_sequence[index]):
            raise ValueError("bar ingest sequences must be ordered")
        if int(watermark_sequence[index]) <= int(last_sequence[index]):
            raise ValueError("watermark sequence must follow all bar ticks")
        if int(quote_count[index]) <= 0:
            raise ValueError("prospective bars cannot be empty")
        if int(staleness[index]) != end - last_tick:
            raise ValueError("staleness must equal bar_end minus last_tick")


def project_quote_bars_v1(bars: pa.Table) -> pa.Table:
    """Project receipt-aware bars to the unchanged causal-v1 source schema."""
    validate_prospective_bar_table(bars)
    return pa.Table.from_arrays(
        [bars.column(field.name) for field in QUOTE_BAR_SCHEMA],
        schema=QUOTE_BAR_SCHEMA,
    )


def _provider_bucket_ns(provider_ns: int) -> int:
    return provider_ns // _INTERVAL_NS * _INTERVAL_NS


class ProspectiveQuoteBarBuilder:
    """Emit bars only after a causally received provider-time watermark."""

    def __init__(self, symbol: str) -> None:
        if not symbol:
            raise ValueError("symbol cannot be empty")
        self._symbol = symbol
        self._quality = ProspectiveTickQualityReport()
        self._carry: pa.Table | None = None
        self._last_provider_ns: int | None = None
        self._last_received_ns: int | None = None
        self._last_sequence: int | None = None
        self._invalidated_boundaries_ns: set[int] = set()
        self._failed = False

    @property
    def quality(self) -> ProspectiveTickQualityReport:
        """Expose monotone quality counters for custody manifests."""
        return self._quality

    @property
    def invalidated_boundaries(self) -> tuple[datetime, ...]:
        """Return provider-time boundaries invalidated by quarantined messages."""
        return tuple(
            pa.scalar(value, type=CANONICAL_TIMESTAMP).as_py()
            for value in sorted(self._invalidated_boundaries_ns)
        )

    def push(self, ticks: pa.Table) -> pa.Table:
        """Consume receipt-ordered ticks and emit only watermark-finalized bars."""
        if self._failed:
            raise RuntimeError("prospective bar builder is terminally invalid")
        canonical = canonicalize_prospective_ticks(ticks)
        if canonical.num_rows == 0:
            return empty_prospective_bars()
        symbols = set(canonical.column("symbol").to_pylist())
        if symbols != {self._symbol}:
            raise ValueError(f"expected only symbol {self._symbol}")
        if any(canonical.column(name).null_count for name in canonical.column_names):
            audit_prospective_tick_table(canonical, self._quality)
            self._failed = True
            raise ValueError("prospective ticks contain null values")
        self._check_stream_order(canonical)
        previous_violations = self._quality.critical_violations
        audit_prospective_tick_table(canonical, self._quality)
        if self._quality.critical_violations != previous_violations:
            self._failed = True
            raise ValueError("prospective ticks contain critical violations")
        self._remember_last_tick(canonical)

        combined = (
            canonical
            if self._carry is None
            else pa.concat_tables([self._carry, canonical]).combine_chunks()
        )
        provider_ns = pc.cast(combined.column("provider_timestamp"), pa.int64())
        bucket_values = [
            _provider_bucket_ns(int(provider_ns[index].as_py()))
            for index in range(combined.num_rows)
        ]
        starts = [0]
        starts.extend(
            index
            for index in range(1, combined.num_rows)
            if bucket_values[index] != bucket_values[index - 1]
        )
        if len(starts) == 1:
            self._carry = combined
            return empty_prospective_bars()

        emitted: list[pa.Table] = []
        starts.append(combined.num_rows)
        for group_index in range(len(starts) - 2):
            start = starts[group_index]
            stop = starts[group_index + 1]
            watermark_index = stop
            emitted.append(
                self._finalize(
                    combined.slice(start, stop - start),
                    combined.column("provider_timestamp")[watermark_index],
                    combined.column("received_at")[watermark_index],
                    combined.column("ingest_sequence")[watermark_index],
                )
            )
        carry_start = starts[-2]
        self._carry = combined.slice(carry_start).combine_chunks()
        return pa.concat_tables(emitted)

    def discard_open_bar(self) -> None:
        """Drop, but never publish, state lacking a terminal watermark."""
        self._carry = None

    def _check_stream_order(self, ticks: pa.Table) -> None:
        provider = pc.cast(ticks.column("provider_timestamp"), pa.int64()).to_pylist()
        received = pc.cast(ticks.column("received_at"), pa.int64()).to_pylist()
        sequences = ticks.column("ingest_sequence").to_pylist()
        previous_provider = self._last_provider_ns
        previous_received = self._last_received_ns
        previous_sequence = self._last_sequence
        for provider_value, received_value, sequence_value in zip(
            provider, received, sequences, strict=True
        ):
            provider_ns = int(provider_value)
            received_ns = int(received_value)
            sequence = int(sequence_value)
            if previous_provider is not None and provider_ns < previous_provider:
                self._quality.provider_out_of_order += 1
                self._invalidated_boundaries_ns.add(
                    _provider_bucket_ns(provider_ns) + _INTERVAL_NS
                )
                self._failed = True
                raise ValueError("provider timestamp regressed behind the watermark")
            if previous_received is not None and received_ns < previous_received:
                self._quality.receipt_out_of_order += 1
                self._failed = True
                raise ValueError("receipt timestamp regressed")
            if previous_sequence is not None and sequence <= previous_sequence:
                self._quality.sequence_not_increasing += 1
                self._failed = True
                raise ValueError("ingest sequence must be strictly increasing")
            previous_provider = provider_ns
            previous_received = received_ns
            previous_sequence = sequence

    def _remember_last_tick(self, ticks: pa.Table) -> None:
        index = ticks.num_rows - 1
        self._last_provider_ns = int(
            pc.cast(ticks.column("provider_timestamp"), pa.int64())[index].as_py()
        )
        self._last_received_ns = int(
            pc.cast(ticks.column("received_at"), pa.int64())[index].as_py()
        )
        self._last_sequence = int(ticks.column("ingest_sequence")[index].as_py())

    def _finalize(
        self,
        ticks: pa.Table,
        watermark_provider_timestamp: pa.Scalar,
        watermark_received_at: pa.Scalar,
        watermark_ingest_sequence: pa.Scalar,
    ) -> pa.Table:
        canonical_ticks = pa.table(
            [
                ticks.column("provider_timestamp"),
                *(ticks.column(name) for name in PRICE_COLUMNS),
            ],
            names=TICK_COLUMNS,
        )
        quote_bar = _aggregate_canonical(canonical_ticks, self._symbol, 5)
        if quote_bar.num_rows != 1:
            raise RuntimeError("one provider-time bucket must produce exactly one bar")
        feature_available = pc.add(
            watermark_received_at,
            pa.scalar(_COMPUTE_ALLOWANCE_NS, type=pa.duration("ns")),
        )
        first_received_ns = pc.cast(
            ticks.column("received_at")[0], pa.int64()
        ).as_py()
        last_received_ns = pc.cast(
            ticks.column("received_at")[ticks.num_rows - 1], pa.int64()
        ).as_py()
        watermark_provider_ns = pc.cast(
            watermark_provider_timestamp, pa.int64()
        ).as_py()
        finalized_ns = pc.cast(watermark_received_at, pa.int64()).as_py()
        available_ns = pc.cast(feature_available, pa.int64()).as_py()
        extras: list[pa.Array] = [
            pa.array([first_received_ns], CANONICAL_TIMESTAMP),
            pa.array([last_received_ns], CANONICAL_TIMESTAMP),
            pa.array([watermark_provider_ns], CANONICAL_TIMESTAMP),
            pa.array([watermark_ingest_sequence.as_py()], type=pa.uint64()),
            pa.array([finalized_ns], CANONICAL_TIMESTAMP),
            pa.array([available_ns], CANONICAL_TIMESTAMP),
            pa.array(
                [ticks.column("ingest_sequence")[0].as_py()], type=pa.uint64()
            ),
            pa.array(
                [ticks.column("ingest_sequence")[ticks.num_rows - 1].as_py()],
                type=pa.uint64(),
            ),
        ]
        return pa.Table.from_arrays(
            [*(quote_bar.column(field.name) for field in QUOTE_BAR_SCHEMA), *extras],
            schema=PROSPECTIVE_BAR_SCHEMA,
        )

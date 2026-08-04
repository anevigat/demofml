"""Receipt-time tick contract for prospective, causally published features."""

from dataclasses import dataclass, field
from datetime import datetime

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]

from demofml.data.ticks import CONSISTENCY_TOLERANCE

PROSPECTIVE_TICK_SET_ID = "prospective-ticks-v1"
PROSPECTIVE_TIMESTAMP = pa.timestamp("ns", tz="UTC")
MAX_PROVIDER_CLOCK_LEAD_NS = 100_000_000
PROSPECTIVE_TICK_COLUMNS = (
    "symbol",
    "provider_timestamp",
    "received_at",
    "ingest_sequence",
    "bid",
    "ask",
    "mid",
    "spread",
)
PROSPECTIVE_TICK_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("provider_timestamp", PROSPECTIVE_TIMESTAMP, nullable=False),
        pa.field("received_at", PROSPECTIVE_TIMESTAMP, nullable=False),
        pa.field("ingest_sequence", pa.uint64(), nullable=False),
        pa.field("bid", pa.float64(), nullable=False),
        pa.field("ask", pa.float64(), nullable=False),
        pa.field("mid", pa.float64(), nullable=False),
        pa.field("spread", pa.float64(), nullable=False),
    ],
    metadata={
        b"demofml.tick_set": PROSPECTIVE_TICK_SET_ID.encode(),
        b"demofml.provider_time": b"preserved",
        b"demofml.receipt_clock_attestation": b"required",
        b"demofml.order": b"received_at_then_ingest_sequence",
    },
)


class ProspectiveTickContractError(ValueError):
    """Raised when prospective ticks violate the immutable physical contract."""


@dataclass
class ProspectiveTickQualityReport:
    """Mergeable quality state for one symbol's receipt-ordered stream."""

    rows: int = 0
    null_values: int = 0
    non_finite_values: int = 0
    non_positive_bid: int = 0
    non_positive_ask: int = 0
    crossed_quotes: int = 0
    inconsistent_mid: int = 0
    inconsistent_spread: int = 0
    provider_out_of_order: int = 0
    receipt_out_of_order: int = 0
    sequence_not_increasing: int = 0
    clock_lead_violations: int = 0
    mixed_symbols: int = 0
    max_delivery_latency_ns: int = 0
    first_provider_timestamp: datetime | None = None
    last_provider_timestamp: datetime | None = None
    first_received_at: datetime | None = None
    last_received_at: datetime | None = None
    _symbol: str | None = field(default=None, repr=False)
    _last_provider_ns: int | None = field(default=None, repr=False)
    _last_received_ns: int | None = field(default=None, repr=False)
    _last_sequence: int | None = field(default=None, repr=False)

    @property
    def critical_violations(self) -> int:
        """Return violations that make publication-time replay unsafe."""
        return (
            self.null_values
            + self.non_finite_values
            + self.non_positive_bid
            + self.non_positive_ask
            + self.crossed_quotes
            + self.inconsistent_mid
            + self.inconsistent_spread
            + self.provider_out_of_order
            + self.receipt_out_of_order
            + self.sequence_not_increasing
            + self.clock_lead_violations
            + self.mixed_symbols
        )

    def as_dict(self) -> dict[str, int | str | None]:
        """Return counters and boundaries without private merge state."""
        return {
            "rows": self.rows,
            "null_values": self.null_values,
            "non_finite_values": self.non_finite_values,
            "non_positive_bid": self.non_positive_bid,
            "non_positive_ask": self.non_positive_ask,
            "crossed_quotes": self.crossed_quotes,
            "inconsistent_mid": self.inconsistent_mid,
            "inconsistent_spread": self.inconsistent_spread,
            "provider_out_of_order": self.provider_out_of_order,
            "receipt_out_of_order": self.receipt_out_of_order,
            "sequence_not_increasing": self.sequence_not_increasing,
            "clock_lead_violations": self.clock_lead_violations,
            "mixed_symbols": self.mixed_symbols,
            "max_delivery_latency_ns": self.max_delivery_latency_ns,
            "symbol": self._symbol,
            "first_provider_timestamp": _iso(self.first_provider_timestamp),
            "last_provider_timestamp": _iso(self.last_provider_timestamp),
            "first_received_at": _iso(self.first_received_at),
            "last_received_at": _iso(self.last_received_at),
            "critical_violations": self.critical_violations,
        }


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _count_true(values: pa.Array | pa.ChunkedArray) -> int:
    total = pc.sum(pc.cast(pc.fill_null(values, False), pa.int64())).as_py()
    return int(total or 0)


def validate_prospective_tick_schema(schema: pa.Schema) -> None:
    """Require exact columns, types, nullability, and contract metadata."""
    if tuple(schema.names) != PROSPECTIVE_TICK_COLUMNS:
        raise ProspectiveTickContractError(
            f"Expected columns {PROSPECTIVE_TICK_COLUMNS}, got {tuple(schema.names)}"
        )
    for expected in PROSPECTIVE_TICK_SCHEMA:
        actual = schema.field(expected.name)
        if expected.name in {"provider_timestamp", "received_at"}:
            if not pa.types.is_timestamp(actual.type):
                raise ProspectiveTickContractError(
                    f"{expected.name} must be an Arrow timestamp"
                )
            if actual.type.unit not in {"us", "ns"} or actual.type.tz != "UTC":
                raise ProspectiveTickContractError(
                    f"{expected.name} must use UTC with us or ns precision"
                )
        elif actual.type != expected.type:
            raise ProspectiveTickContractError(
                f"invalid prospective tick type for {expected.name}"
            )
        if actual.nullable:
            raise ProspectiveTickContractError(
                f"prospective tick field {expected.name} cannot be nullable"
            )
    metadata = schema.metadata or {}
    for key, value in (PROSPECTIVE_TICK_SCHEMA.metadata or {}).items():
        if metadata.get(key) != value:
            raise ProspectiveTickContractError(
                f"prospective tick metadata mismatch for {key.decode()}"
            )


def canonicalize_prospective_ticks(ticks: pa.Table) -> pa.Table:
    """Normalize accepted timestamp precision without reordering receipt data."""
    validate_prospective_tick_schema(ticks.schema)
    columns: list[pa.Array | pa.ChunkedArray] = []
    for schema_field in PROSPECTIVE_TICK_SCHEMA:
        column = ticks.column(schema_field.name)
        columns.append(pc.cast(column, schema_field.type))
    return pa.Table.from_arrays(
        columns, schema=PROSPECTIVE_TICK_SCHEMA
    ).combine_chunks()


def audit_prospective_tick_table(
    ticks: pa.Table,
    report: ProspectiveTickQualityReport | None = None,
) -> ProspectiveTickQualityReport:
    """Audit one receipt-ordered symbol stream and update boundary state."""
    canonical = canonicalize_prospective_ticks(ticks)
    result = report if report is not None else ProspectiveTickQualityReport()
    if canonical.num_rows == 0:
        return result

    columns = {name: canonical.column(name).chunk(0) for name in canonical.column_names}
    provider_ns = pc.cast(columns["provider_timestamp"], pa.int64())
    received_ns = pc.cast(columns["received_at"], pa.int64())
    sequence = columns["ingest_sequence"]
    row_count = canonical.num_rows
    result.rows += row_count
    result.null_values += sum(column.null_count for column in columns.values())
    if any(column.null_count for column in columns.values()):
        return result

    for name in ("bid", "ask", "mid", "spread"):
        result.non_finite_values += _count_true(pc.invert(pc.is_finite(columns[name])))
    result.non_positive_bid += _count_true(pc.less_equal(columns["bid"], 0.0))
    result.non_positive_ask += _count_true(pc.less_equal(columns["ask"], 0.0))
    result.crossed_quotes += _count_true(
        pc.greater(columns["bid"], columns["ask"])
    )
    expected_mid = pc.divide(pc.add(columns["bid"], columns["ask"]), 2.0)
    expected_spread = pc.subtract(columns["ask"], columns["bid"])
    result.inconsistent_mid += _count_true(
        pc.greater(
            pc.abs(pc.subtract(columns["mid"], expected_mid)),
            CONSISTENCY_TOLERANCE,
        )
    )
    result.inconsistent_spread += _count_true(
        pc.greater(
            pc.abs(pc.subtract(columns["spread"], expected_spread)),
            CONSISTENCY_TOLERANCE,
        )
    )

    if row_count > 1:
        result.provider_out_of_order += _count_true(
            pc.less(provider_ns.slice(1), provider_ns.slice(0, row_count - 1))
        )
        result.receipt_out_of_order += _count_true(
            pc.less(received_ns.slice(1), received_ns.slice(0, row_count - 1))
        )
        result.sequence_not_increasing += _count_true(
            pc.less_equal(sequence.slice(1), sequence.slice(0, row_count - 1))
        )

    first_provider = int(provider_ns[0].as_py())
    first_received = int(received_ns[0].as_py())
    first_sequence = int(sequence[0].as_py())
    if result._last_provider_ns is not None:
        result.provider_out_of_order += int(first_provider < result._last_provider_ns)
    if result._last_received_ns is not None:
        result.receipt_out_of_order += int(first_received < result._last_received_ns)
    if result._last_sequence is not None:
        result.sequence_not_increasing += int(first_sequence <= result._last_sequence)

    lead = pc.subtract(provider_ns, received_ns)
    result.clock_lead_violations += _count_true(
        pc.greater(lead, MAX_PROVIDER_CLOCK_LEAD_NS)
    )
    delivery = pc.subtract(received_ns, provider_ns)
    maximum_delivery = pc.max(delivery).as_py()
    result.max_delivery_latency_ns = max(
        result.max_delivery_latency_ns, int(maximum_delivery or 0), 0
    )

    symbols = set(columns["symbol"].to_pylist())
    if len(symbols) != 1:
        result.mixed_symbols += 1
    else:
        symbol = str(next(iter(symbols)))
        if result._symbol is None:
            result._symbol = symbol
        elif result._symbol != symbol:
            result.mixed_symbols += 1

    result._last_provider_ns = int(provider_ns[row_count - 1].as_py())
    result._last_received_ns = int(received_ns[row_count - 1].as_py())
    result._last_sequence = int(sequence[row_count - 1].as_py())
    if result.first_provider_timestamp is None:
        result.first_provider_timestamp = columns["provider_timestamp"][0].as_py()
        result.first_received_at = columns["received_at"][0].as_py()
    result.last_provider_timestamp = columns["provider_timestamp"][
        row_count - 1
    ].as_py()
    result.last_received_at = columns["received_at"][row_count - 1].as_py()
    return result

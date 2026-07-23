"""Canonical tick contract and vectorized quality checks."""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

TICK_COLUMNS = ("timestamp", "bid", "ask", "mid", "spread")
PRICE_COLUMNS = ("bid", "ask", "mid", "spread")
CANONICAL_TIMESTAMP = pa.timestamp("ns", tz="UTC")
CONSISTENCY_TOLERANCE = 1e-12


class TickContractError(ValueError):
    """Raised when tick data does not satisfy the structural contract."""


@dataclass
class TickQualityReport:
    """Mergeable quality counters for one ordered tick stream."""

    rows: int = 0
    null_values: int = 0
    non_finite_values: int = 0
    non_positive_bid: int = 0
    non_positive_ask: int = 0
    crossed_quotes: int = 0
    inconsistent_mid: int = 0
    inconsistent_spread: int = 0
    out_of_order: int = 0
    exact_duplicates: int = 0
    max_gap_ns: int = 0
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    _last_timestamp_ns: int | None = field(default=None, repr=False)
    _last_timestamp_values: set[tuple[object, ...]] = field(
        default_factory=set, repr=False
    )

    @property
    def critical_violations(self) -> int:
        """Return violations that make the stream unsafe for bar generation."""
        return (
            self.null_values
            + self.non_finite_values
            + self.non_positive_bid
            + self.non_positive_ask
            + self.crossed_quotes
            + self.inconsistent_mid
            + self.inconsistent_spread
            + self.out_of_order
            + self.exact_duplicates
        )

    def as_dict(self) -> dict[str, int | str | None]:
        """Return a JSON-compatible report without internal boundary state."""
        return {
            "rows": self.rows,
            "null_values": self.null_values,
            "non_finite_values": self.non_finite_values,
            "non_positive_bid": self.non_positive_bid,
            "non_positive_ask": self.non_positive_ask,
            "crossed_quotes": self.crossed_quotes,
            "inconsistent_mid": self.inconsistent_mid,
            "inconsistent_spread": self.inconsistent_spread,
            "out_of_order": self.out_of_order,
            "exact_duplicates": self.exact_duplicates,
            "max_gap_ns": self.max_gap_ns,
            "first_timestamp": (
                self.first_timestamp.isoformat()
                if self.first_timestamp is not None
                else None
            ),
            "last_timestamp": (
                self.last_timestamp.isoformat()
                if self.last_timestamp is not None
                else None
            ),
            "critical_violations": self.critical_violations,
        }


def validate_tick_schema(schema: pa.Schema) -> None:
    """Validate exact columns and accepted physical timestamp precision."""
    if tuple(schema.names) != TICK_COLUMNS:
        raise TickContractError(
            f"Expected columns {TICK_COLUMNS}, received {tuple(schema.names)}"
        )
    timestamp_type = schema.field("timestamp").type
    if not pa.types.is_timestamp(timestamp_type):
        raise TickContractError("timestamp must be an Arrow timestamp")
    if timestamp_type.unit not in {"us", "ns"} or timestamp_type.tz != "UTC":
        raise TickContractError("timestamp must use UTC with us or ns precision")
    for name in PRICE_COLUMNS:
        if not pa.types.is_float64(schema.field(name).type):
            raise TickContractError(f"{name} must be float64")


def canonicalize_ticks(ticks: pa.Table) -> pa.Table:
    """Validate and normalize a tick table to nanosecond UTC timestamps."""
    validate_tick_schema(ticks.schema)
    timestamp = pc.cast(ticks.column("timestamp"), CANONICAL_TIMESTAMP)
    columns: list[pa.Array | pa.ChunkedArray] = [timestamp]
    columns.extend(ticks.column(name) for name in PRICE_COLUMNS)
    return pa.table(columns, names=TICK_COLUMNS).combine_chunks()


def _count_true(values: pa.Array | pa.ChunkedArray) -> int:
    normalized = pc.fill_null(values, False)
    total = pc.sum(pc.cast(normalized, pa.int64())).as_py()
    return int(total or 0)


def _max_positive_gap(timestamp_ns: pa.Array) -> int:
    if len(timestamp_ns) < 2:
        return 0
    values = pc.cast(timestamp_ns, pa.int64())
    gaps = pc.subtract(values.slice(1), values.slice(0, len(values) - 1))
    maximum = pc.max(gaps).as_py()
    return max(int(maximum or 0), 0)


def _row_values(columns: dict[str, pa.Array], index: int) -> tuple[object, ...]:
    return tuple(columns[name][index].as_py() for name in TICK_COLUMNS)


def _count_exact_duplicates(canonical: pa.Table) -> int:
    grouped = canonical.group_by(list(TICK_COLUMNS), use_threads=False).aggregate(
        [("bid", "count")]
    )
    extra = pc.subtract(grouped.column("bid_count"), 1)
    total = pc.sum(pc.max_element_wise(extra, 0)).as_py()
    return int(total or 0)


def audit_canonical_tick_table(
    canonical: pa.Table,
    report: TickQualityReport | None = None,
) -> TickQualityReport:
    """Audit a table already normalized by :func:`canonicalize_ticks`."""
    if canonical.schema != pa.schema(
        [
            pa.field("timestamp", CANONICAL_TIMESTAMP),
            *(pa.field(name, pa.float64()) for name in PRICE_COLUMNS),
        ]
    ):
        canonical = canonicalize_ticks(canonical)
    result = report if report is not None else TickQualityReport()
    row_count = canonical.num_rows
    if row_count == 0:
        return result

    columns = {name: canonical.column(name).chunk(0) for name in TICK_COLUMNS}
    timestamp = columns["timestamp"]
    result.rows += row_count
    result.null_values += sum(column.null_count for column in columns.values())

    for name in PRICE_COLUMNS:
        result.non_finite_values += _count_true(pc.invert(pc.is_finite(columns[name])))
    result.non_positive_bid += _count_true(pc.less_equal(columns["bid"], 0.0))
    result.non_positive_ask += _count_true(pc.less_equal(columns["ask"], 0.0))
    result.crossed_quotes += _count_true(pc.greater(columns["bid"], columns["ask"]))

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
        left_timestamp = timestamp.slice(0, row_count - 1)
        right_timestamp = timestamp.slice(1)
        result.out_of_order += _count_true(
            pc.less(right_timestamp, left_timestamp)
        )
        result.max_gap_ns = max(result.max_gap_ns, _max_positive_gap(timestamp))
    result.exact_duplicates += _count_exact_duplicates(canonical)

    timestamp_values = pc.cast(timestamp, pa.int64())
    first_timestamp_ns = timestamp_values[0].as_py()
    last_timestamp_ns = timestamp_values[row_count - 1].as_py()

    if result._last_timestamp_ns is not None and first_timestamp_ns is not None:
        if int(first_timestamp_ns) < result._last_timestamp_ns:
            result.out_of_order += 1
        result.max_gap_ns = max(
            result.max_gap_ns,
            max(int(first_timestamp_ns) - result._last_timestamp_ns, 0),
        )
        if int(first_timestamp_ns) == result._last_timestamp_ns:
            prefix_values: set[tuple[object, ...]] = set()
            for index in range(row_count):
                value = timestamp_values[index].as_py()
                if value != first_timestamp_ns:
                    break
                prefix_values.add(_row_values(columns, index))
            result.exact_duplicates += len(
                result._last_timestamp_values.intersection(prefix_values)
            )

    valid_timestamps = pc.drop_null(timestamp)
    if len(valid_timestamps):
        if result.first_timestamp is None:
            result.first_timestamp = valid_timestamps[0].as_py()
        result.last_timestamp = valid_timestamps[len(valid_timestamps) - 1].as_py()
    if last_timestamp_ns is not None:
        suffix_values: set[tuple[object, ...]] = set()
        for index in range(row_count - 1, -1, -1):
            value = timestamp_values[index].as_py()
            if value != last_timestamp_ns:
                break
            suffix_values.add(_row_values(columns, index))
        if (
            result._last_timestamp_ns == int(last_timestamp_ns)
            and first_timestamp_ns == last_timestamp_ns
        ):
            suffix_values.update(result._last_timestamp_values)
        result._last_timestamp_ns = int(last_timestamp_ns)
        result._last_timestamp_values = suffix_values
    return result


def audit_tick_table(
    ticks: pa.Table,
    report: TickQualityReport | None = None,
) -> TickQualityReport:
    """Normalize ticks and update vectorized quality counters."""
    return audit_canonical_tick_table(canonicalize_ticks(ticks), report)


def audit_parquet_file(
    path: Path,
    max_row_groups: int | None = None,
    report: TickQualityReport | None = None,
) -> TickQualityReport:
    """Audit a Parquet file row group by row group with bounded memory."""
    parquet = pq.ParquetFile(path)
    validate_tick_schema(parquet.schema_arrow)
    row_group_count = parquet.metadata.num_row_groups
    limit = (
        row_group_count
        if max_row_groups is None
        else min(max_row_groups, row_group_count)
    )
    result = report if report is not None else TickQualityReport()
    for index in range(limit):
        table = parquet.read_row_group(index, columns=list(TICK_COLUMNS))
        audit_tick_table(table, result)
    return result

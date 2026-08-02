"""Causal trailing microstructure features built on quote-bars-v2."""

import math
from collections import deque
from datetime import datetime, timedelta

import pyarrow as pa  # type: ignore[import-untyped]

from demofml.bars.quotes import QUOTE_BAR_SCHEMA
from demofml.bars.quotes_v2 import validate_quote_bar_v2_schema
from demofml.features.causal import FEATURE_SCHEMA, CausalFeatureBuilder

FEATURE_SET_V2_ID = "causal-v2"
MICROSTRUCTURE_FEATURE_COLUMNS = tuple(
    f"{name}_{minutes}m"
    for name in (
        "bid_ask_update_imbalance",
        "mid_tick_imbalance",
        "spread_change_imbalance",
        "interarrival_dispersion_seconds",
    )
    for minutes in (15, 60)
)
FEATURE_V2_SCHEMA = pa.schema(
    [
        *FEATURE_SCHEMA,
        *(
            pa.field(name, pa.float64())
            for name in MICROSTRUCTURE_FEATURE_COLUMNS
        ),
    ],
    metadata={
        b"demofml.feature_set": FEATURE_SET_V2_ID.encode(),
        b"demofml.source_bar_set": b"quote-bars-v2",
        b"demofml.source_bar_interval_minutes": b"5",
        b"demofml.gap_policy": b"reset_trailing_state",
        b"demofml.microstructure_windows_minutes": b"15,60",
        b"demofml.imbalance_aggregation": b"ratio_of_rolling_count_sums",
        b"demofml.interarrival_aggregation": b"mean_of_intrabar_population_std",
    },
)
FEATURE_V2_COLUMNS = tuple(FEATURE_V2_SCHEMA.names[2:])
_MICROSTRUCTURE_BAR_COLUMNS = (
    "symbol",
    "bar_end",
    "bid_update_count",
    "ask_update_count",
    "mid_uptick_count",
    "mid_downtick_count",
    "spread_widening_count",
    "spread_narrowing_count",
    "interarrival_dispersion_seconds",
)


class _CountImbalanceWindow:
    def __init__(self, size: int) -> None:
        self._size = size
        self._values: deque[tuple[int, int]] = deque()
        self._positive = 0
        self._negative = 0

    def append(self, positive: int, negative: int) -> None:
        if len(self._values) == self._size:
            removed_positive, removed_negative = self._values.popleft()
            self._positive -= removed_positive
            self._negative -= removed_negative
        self._values.append((positive, negative))
        self._positive += positive
        self._negative += negative

    def clear(self) -> None:
        self._values.clear()
        self._positive = 0
        self._negative = 0

    def value(self) -> float | None:
        if len(self._values) != self._size:
            return None
        total = self._positive + self._negative
        return (self._positive - self._negative) / total if total else 0.0


class _MeanWindow:
    def __init__(self, size: int) -> None:
        self._size = size
        self._values: deque[float] = deque()
        self._sum = 0.0

    def append(self, value: float) -> None:
        if len(self._values) == self._size:
            self._sum -= self._values.popleft()
        self._values.append(value)
        self._sum += value

    def clear(self) -> None:
        self._values.clear()
        self._sum = 0.0

    def value(self) -> float | None:
        return self._sum / self._size if len(self._values) == self._size else None


def empty_features_v2() -> pa.Table:
    """Return an empty causal-v2 feature table."""
    return pa.Table.from_batches([], schema=FEATURE_V2_SCHEMA)


class CausalV2FeatureBuilder:
    """Append fixed 15- and 60-minute summaries to unchanged causal-v1 features."""

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self._base = CausalFeatureBuilder(symbol)
        self._imbalance_windows = {
            (name, size): _CountImbalanceWindow(size)
            for name in ("bid_ask_update", "mid_tick", "spread_change")
            for size in (3, 12)
        }
        self._dispersion_windows = {size: _MeanWindow(size) for size in (3, 12)}
        self._previous_end: datetime | None = None

    def _clear(self) -> None:
        for count_window in self._imbalance_windows.values():
            count_window.clear()
        for mean_window in self._dispersion_windows.values():
            mean_window.clear()

    def push(self, bars: pa.Table) -> pa.Table:
        """Compute causal-v2 rows using only each completed bar and trailing state."""
        validate_quote_bar_v2_schema(bars.schema)
        if bars.num_rows == 0:
            return empty_features_v2()
        base_bars = bars.select(QUOTE_BAR_SCHEMA.names).replace_schema_metadata(
            QUOTE_BAR_SCHEMA.metadata
        )
        base_rows = self._base.push(base_bars).to_pylist()
        micro_rows = bars.select(_MICROSTRUCTURE_BAR_COLUMNS).to_pylist()
        output: list[dict[str, object]] = []
        for base, micro in zip(base_rows, micro_rows, strict=True):
            symbol = str(micro["symbol"])
            bar_end = micro["bar_end"]
            if symbol != self._symbol:
                raise ValueError(f"Expected symbol {self._symbol}, received {symbol}")
            if not isinstance(bar_end, datetime):
                raise ValueError("bar_end cannot be null")
            if self._previous_end is not None:
                if bar_end <= self._previous_end:
                    raise ValueError("bars must be strictly ordered by bar_end")
                if bar_end - self._previous_end != timedelta(minutes=5):
                    self._clear()

            counts = {
                "bid_ask_update": (
                    int(micro["bid_update_count"]),
                    int(micro["ask_update_count"]),
                ),
                "mid_tick": (
                    int(micro["mid_uptick_count"]),
                    int(micro["mid_downtick_count"]),
                ),
                "spread_change": (
                    int(micro["spread_widening_count"]),
                    int(micro["spread_narrowing_count"]),
                ),
            }
            dispersion = float(micro["interarrival_dispersion_seconds"])
            if any(value < 0 for pair in counts.values() for value in pair):
                raise ValueError("microstructure counts cannot be negative")
            if not math.isfinite(dispersion) or dispersion < 0.0:
                raise ValueError(
                    "interarrival dispersion must be finite and non-negative"
                )
            for (name, _size), count_window in self._imbalance_windows.items():
                count_window.append(*counts[name])
            for mean_window in self._dispersion_windows.values():
                mean_window.append(dispersion)

            row = dict(base)
            for name in ("bid_ask_update", "mid_tick", "spread_change"):
                for size, minutes in ((3, 15), (12, 60)):
                    row[f"{name}_imbalance_{minutes}m"] = self._imbalance_windows[
                        (name, size)
                    ].value()
            for size, minutes in ((3, 15), (12, 60)):
                row[f"interarrival_dispersion_seconds_{minutes}m"] = (
                    self._dispersion_windows[size].value()
                )
            output.append(row)
            self._previous_end = bar_end
        return pa.Table.from_pylist(output, schema=FEATURE_V2_SCHEMA)

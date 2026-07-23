"""Streaming causal features computed from completed quote bars."""

import math
from collections import deque
from datetime import datetime, timedelta

import pyarrow as pa  # type: ignore[import-untyped]

from demofml.bars.quotes import BAR_TIMESTAMP, validate_quote_bar_schema

FEATURE_SET_ID = "causal-v1"
FEATURE_SCHEMA = pa.schema(
    [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("bar_end", BAR_TIMESTAMP, nullable=False),
        pa.field("mid_return_1", pa.float64()),
        pa.field("mid_return_3", pa.float64()),
        pa.field("mid_return_12", pa.float64()),
        pa.field("realized_volatility_12", pa.float64()),
        pa.field("realized_volatility_72", pa.float64()),
        pa.field("spread_bps", pa.float64(), nullable=False),
        pa.field("spread_zscore_72", pa.float64()),
        pa.field("mid_range_bps", pa.float64(), nullable=False),
        pa.field("quote_count_log1p", pa.float64(), nullable=False),
        pa.field("staleness_seconds", pa.float64(), nullable=False),
        pa.field("elapsed_seconds", pa.float64()),
        pa.field("hour_sin", pa.float64(), nullable=False),
        pa.field("hour_cos", pa.float64(), nullable=False),
        pa.field("weekday_sin", pa.float64(), nullable=False),
        pa.field("weekday_cos", pa.float64(), nullable=False),
    ],
    metadata={
        b"demofml.feature_set": FEATURE_SET_ID.encode(),
        b"demofml.source_bar_set": b"quote-bars-v1",
        b"demofml.source_bar_interval_minutes": b"5",
        b"demofml.gap_policy": b"reset_trailing_state",
    },
)

_REQUIRED_BAR_COLUMNS = {
    "symbol",
    "bar_start",
    "bar_end",
    "mid_high",
    "mid_low",
    "mid_close",
    "spread_close",
    "quote_count",
    "staleness_ns",
}


class _RollingWindow:
    def __init__(self, size: int) -> None:
        self._size = size
        self._values: deque[float] = deque()
        self._sum = 0.0
        self._sum_squares = 0.0

    def append(self, value: float) -> None:
        if len(self._values) == self._size:
            removed = self._values.popleft()
            self._sum -= removed
            self._sum_squares -= removed * removed
        self._values.append(value)
        self._sum += value
        self._sum_squares += value * value

    def clear(self) -> None:
        self._values.clear()
        self._sum = 0.0
        self._sum_squares = 0.0

    @property
    def full(self) -> bool:
        return len(self._values) == self._size

    def realized_volatility(self) -> float | None:
        if not self.full:
            return None
        return math.sqrt(max(self._sum_squares, 0.0))

    def zscore(self, value: float) -> float | None:
        if not self.full:
            return None
        mean = self._sum / self._size
        variance = max(self._sum_squares / self._size - mean * mean, 0.0)
        if variance == 0.0:
            return 0.0
        return (value - mean) / math.sqrt(variance)


def empty_features() -> pa.Table:
    """Return an empty table with the versioned feature schema."""
    return pa.Table.from_batches([], schema=FEATURE_SCHEMA)


class CausalFeatureBuilder:
    """Maintain only bounded trailing state while consuming ordered bars."""

    def __init__(self, symbol: str) -> None:
        self._symbol = symbol
        self._closes: deque[float] = deque(maxlen=12)
        self._returns_12 = _RollingWindow(12)
        self._returns_72 = _RollingWindow(72)
        self._spreads_72 = _RollingWindow(72)
        self._previous_end: datetime | None = None

    def push(self, bars: pa.Table) -> pa.Table:
        """Compute features for completed bars without reading future rows."""
        validate_quote_bar_schema(bars.schema)
        missing = _REQUIRED_BAR_COLUMNS.difference(bars.column_names)
        if missing:
            raise ValueError(f"Missing quote-bar columns: {sorted(missing)}")
        rows: list[dict[str, object]] = []
        for bar in bars.select(sorted(_REQUIRED_BAR_COLUMNS)).to_pylist():
            symbol = str(bar["symbol"])
            bar_start = bar["bar_start"]
            bar_end = bar["bar_end"]
            if symbol != self._symbol:
                raise ValueError(f"Expected symbol {self._symbol}, received {symbol}")
            if not isinstance(bar_end, datetime):
                raise ValueError("bar_end cannot be null")
            if not isinstance(bar_start, datetime):
                raise ValueError("bar_start cannot be null")
            if bar_end - bar_start != timedelta(minutes=5):
                raise ValueError("features require five-minute quote bars")
            if self._previous_end is not None and bar_end <= self._previous_end:
                raise ValueError("bars must be strictly ordered by bar_end")
            if (
                bar_end.minute % 5
                or bar_end.second
                or bar_end.microsecond
            ):
                raise ValueError("bar_end must align to the five-minute UTC grid")

            mid_close = float(bar["mid_close"])
            mid_high = float(bar["mid_high"])
            mid_low = float(bar["mid_low"])
            spread_close = float(bar["spread_close"])
            quote_count = int(bar["quote_count"])
            staleness_ns = int(bar["staleness_ns"])
            if not all(
                math.isfinite(value)
                for value in (mid_close, mid_high, mid_low, spread_close)
            ):
                raise ValueError("bar prices must be finite")
            if (
                mid_close <= 0.0
                or mid_low > mid_close
                or mid_close > mid_high
                or spread_close < 0.0
                or quote_count < 0
                or staleness_ns < 0
            ):
                raise ValueError("bar prices and activity metrics must be valid")

            elapsed = (
                (bar_end - self._previous_end).total_seconds()
                if self._previous_end is not None
                else None
            )
            if elapsed is not None and elapsed != 300.0:
                self._closes.clear()
                self._returns_12.clear()
                self._returns_72.clear()
                self._spreads_72.clear()

            return_1 = (
                math.log(mid_close / self._closes[-1]) if self._closes else None
            )
            return_3 = (
                math.log(mid_close / self._closes[-3])
                if len(self._closes) >= 3
                else None
            )
            return_12 = (
                math.log(mid_close / self._closes[-12])
                if len(self._closes) >= 12
                else None
            )
            if return_1 is not None:
                self._returns_12.append(return_1)
                self._returns_72.append(return_1)
            spread_bps = spread_close / mid_close * 10_000.0
            self._spreads_72.append(spread_bps)

            seconds = (
                bar_end.hour * 3600
                + bar_end.minute * 60
                + bar_end.second
                + bar_end.microsecond / 1_000_000
            )
            day_angle = 2.0 * math.pi * seconds / 86_400.0
            week_angle = 2.0 * math.pi * bar_end.weekday() / 7.0
            rows.append(
                {
                    "symbol": symbol,
                    "bar_end": bar_end,
                    "mid_return_1": return_1,
                    "mid_return_3": return_3,
                    "mid_return_12": return_12,
                    "realized_volatility_12": (
                        self._returns_12.realized_volatility()
                    ),
                    "realized_volatility_72": (
                        self._returns_72.realized_volatility()
                    ),
                    "spread_bps": spread_bps,
                    "spread_zscore_72": self._spreads_72.zscore(spread_bps),
                    "mid_range_bps": (mid_high - mid_low) / mid_close * 10_000.0,
                    "quote_count_log1p": math.log1p(quote_count),
                    "staleness_seconds": staleness_ns / 1_000_000_000.0,
                    "elapsed_seconds": elapsed,
                    "hour_sin": math.sin(day_angle),
                    "hour_cos": math.cos(day_angle),
                    "weekday_sin": math.sin(week_angle),
                    "weekday_cos": math.cos(week_angle),
                }
            )
            self._closes.append(mid_close)
            self._previous_end = bar_end
        if not rows:
            return empty_features()
        return pa.Table.from_pylist(rows, schema=FEATURE_SCHEMA)

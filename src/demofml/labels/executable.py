"""Streaming executable bid/ask labels aligned to five-minute decisions."""

import math
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pyarrow as pa  # type: ignore[import-untyped]

from demofml.bars.quotes import BAR_TIMESTAMP, validate_quote_bar_schema

BAR_INTERVAL_MINUTES = 5
MAX_QUOTE_LATENCY_MINUTES = 5
DEFAULT_HORIZONS_MINUTES = (15, 30, 60)
LABEL_SET_ID = "executable-v1"
_REQUIRED_COLUMNS = (
    "symbol",
    "bar_start",
    "bar_end",
    "first_tick",
    "bid_open",
    "ask_open",
)


def _validate_parameters(
    horizons_minutes: Sequence[int], minimum_return_bps: float
) -> tuple[int, ...]:
    horizons = tuple(horizons_minutes)
    if not horizons or any(
        value <= 0 or value % BAR_INTERVAL_MINUTES for value in horizons
    ):
        raise ValueError("horizons must be positive multiples of five minutes")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be unique and increasing")
    if not math.isfinite(minimum_return_bps) or minimum_return_bps < 0.0:
        raise ValueError("minimum_return_bps must be finite and non-negative")
    return horizons


def label_schema(
    horizons_minutes: Sequence[int],
    minimum_return_bps: float = 0.0,
) -> pa.Schema:
    """Build a schema containing every behavior-affecting label parameter."""
    horizons = _validate_parameters(horizons_minutes, minimum_return_bps)
    fields = [
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("decision_time", BAR_TIMESTAMP, nullable=False),
        pa.field("entry_time", BAR_TIMESTAMP),
        pa.field("entry_bid", pa.float64()),
        pa.field("entry_ask", pa.float64()),
    ]
    for horizon in horizons:
        suffix = f"{horizon}m"
        fields.extend(
            [
                pa.field(f"exit_time_{suffix}", BAR_TIMESTAMP),
                pa.field(f"long_return_{suffix}", pa.float64()),
                pa.field(f"short_return_{suffix}", pa.float64()),
                pa.field(f"action_{suffix}", pa.string()),
            ]
        )
    return pa.schema(
        fields,
        metadata={
            b"demofml.label_set": LABEL_SET_ID.encode(),
            b"demofml.horizons_minutes": ",".join(
                str(value) for value in horizons
            ).encode(),
            b"demofml.minimum_return_bps": str(minimum_return_bps).encode(),
            b"demofml.source_bar_set": b"quote-bars-v1",
            b"demofml.source_bar_interval_minutes": str(
                BAR_INTERVAL_MINUTES
            ).encode(),
            b"demofml.max_quote_latency_minutes": str(
                MAX_QUOTE_LATENCY_MINUTES
            ).encode(),
        },
    )


def _action(long_return: float, short_return: float, threshold: float) -> str:
    if long_return <= threshold and short_return <= threshold:
        return "flat"
    return "long" if long_return > short_return else "short"


def _number(value: object, name: str) -> float:
    if not isinstance(value, int | float):
        raise ValueError(f"{name} cannot be null")
    return float(value)


@dataclass
class _PendingDecision:
    symbol: str
    decision_time: datetime
    horizons: tuple[int, ...]
    entry_time: datetime | None = None
    entry_bid: float | None = None
    entry_ask: float | None = None
    entry_resolved: bool = False
    results: dict[int, tuple[datetime, float, float, str] | None] = field(
        default_factory=dict
    )

    @property
    def complete(self) -> bool:
        return self.entry_resolved and len(self.results) == len(self.horizons)

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "symbol": self.symbol,
            "decision_time": self.decision_time,
            "entry_time": self.entry_time,
            "entry_bid": self.entry_bid,
            "entry_ask": self.entry_ask,
        }
        for horizon in self.horizons:
            suffix = f"{horizon}m"
            result = self.results.get(horizon)
            row[f"exit_time_{suffix}"] = result[0] if result is not None else None
            row[f"long_return_{suffix}"] = result[1] if result is not None else None
            row[f"short_return_{suffix}"] = result[2] if result is not None else None
            row[f"action_{suffix}"] = result[3] if result is not None else None
        return row


class ExecutableLabelBuilder:
    """Resolve pending decisions as future executable quotes arrive."""

    def __init__(
        self,
        horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
        minimum_return_bps: float = 0.0,
    ) -> None:
        self._horizons = _validate_parameters(
            horizons_minutes, minimum_return_bps
        )
        self._minimum_return = minimum_return_bps / 10_000.0
        self._schema = label_schema(self._horizons, minimum_return_bps)
        self._pending: deque[_PendingDecision] = deque()
        self._symbol: str | None = None
        self._last_first_tick: datetime | None = None
        self._last_bar_end: datetime | None = None

    @property
    def schema(self) -> pa.Schema:
        return self._schema

    def _validate_bar(self, bar: dict[str, object]) -> None:
        symbol = str(bar["symbol"])
        bar_start = bar["bar_start"]
        bar_end = bar["bar_end"]
        first_tick = bar["first_tick"]
        if not isinstance(bar_start, datetime):
            raise ValueError("bar timestamps cannot be null")
        if not isinstance(bar_end, datetime):
            raise ValueError("bar timestamps cannot be null")
        if not isinstance(first_tick, datetime):
            raise ValueError("bar timestamps cannot be null")
        if self._symbol is None:
            self._symbol = symbol
        if symbol != self._symbol:
            raise ValueError("labels must be generated for exactly one symbol")
        if bar_end - bar_start != timedelta(minutes=BAR_INTERVAL_MINUTES):
            raise ValueError("labels require five-minute quote bars")
        if not bar_start <= first_tick < bar_end:
            raise ValueError("first_tick must fall inside its half-open bar")
        if self._last_first_tick is not None and first_tick <= self._last_first_tick:
            raise ValueError("first_tick values must be strictly ordered")
        if self._last_bar_end is not None and bar_end <= self._last_bar_end:
            raise ValueError("bar_end values must be strictly ordered")
        bid = _number(bar["bid_open"], "bid_open")
        ask = _number(bar["ask_open"], "ask_open")
        if not all(math.isfinite(value) for value in (bid, ask)):
            raise ValueError("executable prices must be finite")
        if bid <= 0.0 or ask < bid:
            raise ValueError("executable bid/ask prices are invalid")

    def _update_pending(self, bar: dict[str, object]) -> None:
        first_tick = bar["first_tick"]
        bid = _number(bar["bid_open"], "bid_open")
        ask = _number(bar["ask_open"], "ask_open")
        if not isinstance(first_tick, datetime):
            raise ValueError("first_tick cannot be null")
        for pending in self._pending:
            if not pending.entry_resolved and first_tick >= pending.decision_time:
                entry_deadline = pending.decision_time + timedelta(
                    minutes=MAX_QUOTE_LATENCY_MINUTES
                )
                pending.entry_resolved = True
                if first_tick <= entry_deadline:
                    pending.entry_time = first_tick
                    pending.entry_bid = bid
                    pending.entry_ask = ask
                else:
                    pending.results = {
                        horizon: None for horizon in pending.horizons
                    }
                    continue

            if pending.entry_time is None:
                continue
            for horizon in pending.horizons:
                if horizon in pending.results:
                    continue
                target = pending.decision_time + timedelta(minutes=horizon)
                if first_tick < target:
                    continue
                exit_deadline = target + timedelta(
                    minutes=MAX_QUOTE_LATENCY_MINUTES
                )
                if first_tick > exit_deadline:
                    pending.results[horizon] = None
                    continue
                if pending.entry_bid is None or pending.entry_ask is None:
                    raise RuntimeError("resolved entry prices are missing")
                long_return = bid / pending.entry_ask - 1.0
                short_return = 1.0 - ask / pending.entry_bid
                pending.results[horizon] = (
                    first_tick,
                    long_return,
                    short_return,
                    _action(long_return, short_return, self._minimum_return),
                )

    def _emit_complete(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        while self._pending and self._pending[0].complete:
            rows.append(self._pending.popleft().as_row())
        return rows

    def push(self, bars: pa.Table) -> pa.Table:
        """Consume ordered bars and emit decisions whose horizons are resolved."""
        validate_quote_bar_schema(bars.schema)
        rows: list[dict[str, object]] = []
        for bar in bars.select(list(_REQUIRED_COLUMNS)).to_pylist():
            self._validate_bar(bar)
            self._update_pending(bar)
            rows.extend(self._emit_complete())
            symbol = str(bar["symbol"])
            decision_time = bar["bar_end"]
            first_tick = bar["first_tick"]
            if not isinstance(decision_time, datetime) or not isinstance(
                first_tick, datetime
            ):
                raise ValueError("bar timestamps cannot be null")
            self._pending.append(
                _PendingDecision(symbol, decision_time, self._horizons)
            )
            self._last_first_tick = first_tick
            self._last_bar_end = decision_time
        if not rows:
            return pa.Table.from_batches([], schema=self._schema)
        return pa.Table.from_pylist(rows, schema=self._schema)

    def finish(self) -> pa.Table:
        """Emit trailing decisions with unresolved horizons represented as null."""
        rows = [pending.as_row() for pending in self._pending]
        self._pending.clear()
        if not rows:
            return pa.Table.from_batches([], schema=self._schema)
        return pa.Table.from_pylist(rows, schema=self._schema)


def generate_executable_labels(
    bars: pa.Table,
    horizons_minutes: Sequence[int] = DEFAULT_HORIZONS_MINUTES,
    minimum_return_bps: float = 0.0,
) -> pa.Table:
    """Generate labels in-memory through the same bounded streaming state."""
    builder = ExecutableLabelBuilder(horizons_minutes, minimum_return_bps)
    completed = builder.push(bars)
    trailing = builder.finish()
    return pa.concat_tables([completed, trailing])

"""Deterministic synchronized cross-pair factors for Campaign 2 engineering."""

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta

import pyarrow as pa  # type: ignore[import-untyped]

from demofml.bars.prospective import (
    project_quote_bars_v1,
    validate_prospective_bar_table,
)
from demofml.features.causal import (
    FEATURE_SCHEMA,
    CausalFeatureBuilder,
)

CROSS_PAIR_FEATURE_SET_ID = "causal-v1-cross-pair-v1"
PAIRS = (
    "AUDUSD",
    "EURCHF",
    "EURJPY",
    "EURUSD",
    "GBPJPY",
    "GBPUSD",
    "USDCAD",
    "USDJPY",
)
CURRENCIES = ("AUD", "CAD", "CHF", "EUR", "GBP", "JPY")
PAIR_CURRENCIES: dict[str, tuple[str, str]] = {
    symbol: (symbol[:3], symbol[3:]) for symbol in PAIRS
}
CROSS_PAIR_COLUMNS = (
    "base_strength_1",
    "base_strength_sum_3",
    "base_strength_sum_12",
    "quote_strength_1",
    "quote_strength_sum_3",
    "quote_strength_sum_12",
    "pair_factor_residual_1",
    "pair_factor_residual_sum_3",
    "pair_factor_residual_sum_12",
    "cross_pair_residual_dispersion_1",
    "cross_pair_residual_dispersion_mean_3",
    "cross_pair_residual_dispersion_mean_12",
)
_CONTROL_FIELDS = [
    pa.field("symbol", pa.string(), nullable=False),
    pa.field("decision_time", pa.timestamp("ns", tz="UTC"), nullable=False),
    pa.field("feature_available_at", pa.timestamp("ns", tz="UTC"), nullable=False),
    *[
        pa.field(field.name, field.type, nullable=field.nullable)
        for field in list(FEATURE_SCHEMA)[2:]
    ],
]
CONTROL_FEATURE_SCHEMA = pa.schema(
    _CONTROL_FIELDS,
    metadata={
        b"demofml.feature_set": b"causal-v1-prospective-control-v1",
        b"demofml.source_bar_set": b"prospective-quote-bars-v1",
        b"demofml.availability": b"max_bar_finalized_at_plus_one_second",
    },
)
CANDIDATE_FEATURE_SCHEMA = pa.schema(
    [
        *_CONTROL_FIELDS,
        *(pa.field(name, pa.float64()) for name in CROSS_PAIR_COLUMNS),
    ],
    metadata={
        b"demofml.feature_set": CROSS_PAIR_FEATURE_SET_ID.encode(),
        b"demofml.source_feature_set": b"causal-v1",
        b"demofml.source_bar_set": b"prospective-quote-bars-v1",
        b"demofml.gap_policy": b"reset_cross_pair_windows",
    },
)


@dataclass(frozen=True)
class FactorSolution:
    """Fixed-order currency strengths, pair residuals, and dispersion."""

    strengths: tuple[float, ...]
    residuals: tuple[float, ...]
    residual_dispersion: float

    def strength(self, currency: str) -> float:
        """Return a fitted currency strength; USD is the exact anchor zero."""
        if currency == "USD":
            return 0.0
        try:
            return self.strengths[CURRENCIES.index(currency)]
        except ValueError as error:
            raise KeyError(currency) from error

    def residual(self, symbol: str) -> float:
        """Return a pair residual in the frozen pair ordering."""
        try:
            return self.residuals[PAIRS.index(symbol)]
        except ValueError as error:
            raise KeyError(symbol) from error


def solve_cross_pair_factor(pair_returns: Mapping[str, float]) -> FactorSolution:
    """Apply the frozen full-rank closed form without numerical rank decisions."""
    if set(pair_returns) != set(PAIRS):
        missing = sorted(set(PAIRS).difference(pair_returns))
        extra = sorted(set(pair_returns).difference(PAIRS))
        raise ValueError(f"factor inputs mismatch; missing={missing}, extra={extra}")
    values = tuple(float(pair_returns[symbol]) for symbol in PAIRS)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("factor returns must be finite")
    a, x, u, e, v, g, d, j = values

    aud = a
    cad = -d
    jpy = (-u - v + e + g - 2.0 * j) / 4.0
    eur = (u + e + jpy) / 2.0
    gbp = (v + g + jpy) / 2.0
    chf = eur - x
    strengths = (aud, cad, chf, eur, gbp, jpy)
    predictions = (
        aud,
        eur - chf,
        eur - jpy,
        eur,
        gbp - jpy,
        gbp,
        -cad,
        -jpy,
    )
    residuals = tuple(
        observed - predicted
        for observed, predicted in zip(values, predictions, strict=True)
    )
    mean = sum(residuals) / len(residuals)
    variance = sum((value - mean) ** 2 for value in residuals) / len(residuals)
    return FactorSolution(strengths, residuals, math.sqrt(max(variance, 0.0)))


@dataclass(frozen=True)
class SynchronizedCrossSection:
    """Closed expected boundary with either eight bars or explicit missingness."""

    decision_time: datetime
    bars: Mapping[str, pa.Table]
    missing_symbols: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return not self.missing_symbols


class EightPairSynchronizer:
    """Buffer symbol bars until the scheduler explicitly closes a boundary."""

    def __init__(self) -> None:
        self._pending: dict[datetime, dict[str, pa.Table]] = {}
        self._last_closed: datetime | None = None

    def push(self, bars: pa.Table) -> None:
        """Buffer receipt-aware bars without deciding whether a section is late."""
        validate_prospective_bar_table(bars)
        for index in range(bars.num_rows):
            row = bars.slice(index, 1)
            symbol = str(row.column("symbol")[0].as_py())
            decision_time = row.column("bar_end")[0].as_py()
            if symbol not in PAIRS:
                raise ValueError(f"unsupported Campaign 2 symbol {symbol}")
            if not isinstance(decision_time, datetime):
                raise ValueError("bar_end must be a datetime")
            if self._last_closed is not None and decision_time <= self._last_closed:
                raise ValueError("cannot add a bar to a closed boundary")
            section = self._pending.setdefault(decision_time, {})
            if symbol in section:
                raise ValueError(
                    f"duplicate {symbol} bar at {decision_time.isoformat()}"
                )
            section[symbol] = row

    def close(self, decision_time: datetime) -> SynchronizedCrossSection:
        """Close one scheduled boundary; absent symbols remain explicitly missing."""
        if self._last_closed is not None and decision_time <= self._last_closed:
            raise ValueError("boundaries must close in strictly increasing order")
        earlier = [value for value in self._pending if value < decision_time]
        if earlier:
            raise ValueError("earlier buffered boundaries must be closed first")
        bars = self._pending.pop(decision_time, {})
        self._last_closed = decision_time
        missing = tuple(symbol for symbol in PAIRS if symbol not in bars)
        return SynchronizedCrossSection(decision_time, bars, missing)


@dataclass(frozen=True)
class PairedFeatureBatch:
    """Outcome-free paired features and readiness for one expected boundary."""

    decision_time: datetime
    control: pa.Table
    candidate: pa.Table
    missing_symbols: tuple[str, ...]
    feature_available_at: datetime | None
    ready: bool


def _empty_control() -> pa.Table:
    return pa.Table.from_batches([], schema=CONTROL_FEATURE_SCHEMA)


def _empty_candidate() -> pa.Table:
    return pa.Table.from_batches([], schema=CANDIDATE_FEATURE_SCHEMA)


class PairedCrossPairFeatureBuilder:
    """Build the causal-v1 control and fixed cross-pair candidate together."""

    def __init__(self) -> None:
        self._causal = {symbol: CausalFeatureBuilder(symbol) for symbol in PAIRS}
        self._strengths = {
            currency: deque[float](maxlen=12) for currency in CURRENCIES
        }
        self._residuals = {symbol: deque[float](maxlen=12) for symbol in PAIRS}
        self._dispersion: deque[float] = deque(maxlen=12)
        self._previous_boundary: datetime | None = None

    def push(self, section: SynchronizedCrossSection) -> PairedFeatureBatch:
        """Consume one closed section, resetting cross state on every gap."""
        expected_missing = tuple(
            symbol for symbol in PAIRS if symbol not in section.bars
        )
        if section.missing_symbols != expected_missing:
            raise ValueError("cross-section missing_symbols do not match its bars")
        unexpected = set(section.bars).difference(PAIRS)
        if unexpected:
            raise ValueError(f"cross-section contains unexpected symbols: {unexpected}")
        if self._previous_boundary is not None:
            if section.decision_time <= self._previous_boundary:
                raise ValueError("cross-sections must be strictly ordered")
            if section.decision_time - self._previous_boundary != timedelta(minutes=5):
                self._clear_cross_state()
        self._previous_boundary = section.decision_time
        if not section.complete:
            for symbol, bar in section.bars.items():
                self._push_causal_feature(symbol, bar, section.decision_time)
            self._clear_cross_state()
            return PairedFeatureBatch(
                section.decision_time,
                _empty_control(),
                _empty_candidate(),
                section.missing_symbols,
                None,
                False,
            )

        control_rows: list[dict[str, object]] = []
        availability: list[datetime] = []
        returns: dict[str, float] = {}
        for symbol in PAIRS:
            bar = section.bars[symbol]
            feature_row = self._push_causal_feature(
                symbol, bar, section.decision_time
            )
            available = bar.column("feature_available_at")[0].as_py()
            if not isinstance(available, datetime):
                raise ValueError("feature_available_at must be a datetime")
            availability.append(available)
            return_1 = feature_row["mid_return_1"]
            if return_1 is not None:
                if not isinstance(return_1, (int, float)):
                    raise TypeError("mid_return_1 must be numeric")
                returns[symbol] = float(return_1)
            control_rows.append(
                {
                    "symbol": symbol,
                    "decision_time": section.decision_time,
                    "feature_available_at": max(availability),
                    **{
                        field.name: feature_row[field.name]
                        for field in list(FEATURE_SCHEMA)[2:]
                    },
                }
            )

        common_available = max(availability)
        for row in control_rows:
            row["feature_available_at"] = common_available
        control = pa.Table.from_pylist(control_rows, schema=CONTROL_FEATURE_SCHEMA)
        if set(returns) != set(PAIRS):
            self._clear_cross_state()
            candidate_rows = [
                {**row, **dict.fromkeys(CROSS_PAIR_COLUMNS)} for row in control_rows
            ]
            candidate = pa.Table.from_pylist(
                candidate_rows, schema=CANDIDATE_FEATURE_SCHEMA
            )
            return PairedFeatureBatch(
                section.decision_time,
                control,
                candidate,
                (),
                common_available,
                False,
            )

        solution = solve_cross_pair_factor(returns)
        for currency in CURRENCIES:
            self._strengths[currency].append(solution.strength(currency))
        for symbol in PAIRS:
            self._residuals[symbol].append(solution.residual(symbol))
        self._dispersion.append(solution.residual_dispersion)

        candidate_rows = []
        for control_row in control_rows:
            symbol = str(control_row["symbol"])
            base, quote = PAIR_CURRENCIES[symbol]
            residual = self._residuals[symbol]
            candidate_rows.append(
                {
                    **control_row,
                    "base_strength_1": self._latest_strength(base),
                    "base_strength_sum_3": self._strength_sum(base, 3),
                    "base_strength_sum_12": self._strength_sum(base, 12),
                    "quote_strength_1": self._latest_strength(quote),
                    "quote_strength_sum_3": self._strength_sum(quote, 3),
                    "quote_strength_sum_12": self._strength_sum(quote, 12),
                    "pair_factor_residual_1": residual[-1],
                    "pair_factor_residual_sum_3": self._sum_if_full(residual, 3),
                    "pair_factor_residual_sum_12": self._sum_if_full(residual, 12),
                    "cross_pair_residual_dispersion_1": self._dispersion[-1],
                    "cross_pair_residual_dispersion_mean_3": self._mean_if_full(
                        self._dispersion, 3
                    ),
                    "cross_pair_residual_dispersion_mean_12": self._mean_if_full(
                        self._dispersion, 12
                    ),
                }
            )
        candidate = pa.Table.from_pylist(
            candidate_rows, schema=CANDIDATE_FEATURE_SCHEMA
        )
        return PairedFeatureBatch(
            section.decision_time,
            control,
            candidate,
            (),
            common_available,
            True,
        )

    def _push_causal_feature(
        self,
        symbol: str,
        bar: pa.Table,
        decision_time: datetime,
    ) -> dict[str, object]:
        validate_prospective_bar_table(bar)
        if bar.num_rows != 1:
            raise ValueError("each synchronized symbol must provide one bar")
        if bar.column("symbol")[0].as_py() != symbol:
            raise ValueError("cross-section key does not match bar symbol")
        if bar.column("bar_end")[0].as_py() != decision_time:
            raise ValueError("synchronized bar_end does not match decision time")
        feature = self._causal[symbol].push(project_quote_bars_v1(bar))
        row = feature.to_pylist()[0]
        return {str(name): value for name, value in row.items()}

    def _clear_cross_state(self) -> None:
        for values in self._strengths.values():
            values.clear()
        for values in self._residuals.values():
            values.clear()
        self._dispersion.clear()

    def _latest_strength(self, currency: str) -> float:
        if currency == "USD":
            return 0.0
        return self._strengths[currency][-1]

    def _strength_sum(self, currency: str, size: int) -> float | None:
        if currency == "USD":
            if len(self._dispersion) >= size:
                return 0.0
            return None
        return self._sum_if_full(self._strengths[currency], size)

    @staticmethod
    def _sum_if_full(values: deque[float], size: int) -> float | None:
        if len(values) < size:
            return None
        return sum(list(values)[-size:])

    @staticmethod
    def _mean_if_full(values: deque[float], size: int) -> float | None:
        total = PairedCrossPairFeatureBuilder._sum_if_full(values, size)
        return None if total is None else total / size

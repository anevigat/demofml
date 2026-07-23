"""Causal multi-symbol portfolio accounting for executable predictions."""

from __future__ import annotations

import heapq
import math
import tomllib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]

from demofml.labels.executable import MAX_QUOTE_LATENCY_MINUTES
from demofml.models.baseline import MODEL_SET_ID, PREDICTION_SET_ID
from demofml.validation.splits import VALIDATION_SET_ID

PORTFOLIO_SET_ID = "normalized-sleeve-portfolio-v1"
PORTFOLIO_SYMBOLS = (
    "AUDUSD",
    "EURCHF",
    "EURJPY",
    "EURUSD",
    "GBPJPY",
    "GBPUSD",
    "USDCAD",
    "USDJPY",
)
PORTFOLIO_HORIZONS = (15, 30, 60)
_ACTIONS = frozenset({"long", "short", "flat"})
_TIMESTAMP = pa.timestamp("ns", tz="UTC")
_PREDICTION_FIELDS = {
    "model_set": pa.string(),
    "validation_set": pa.string(),
    "fold_id": pa.string(),
    "symbol": pa.string(),
    "decision_time": _TIMESTAMP,
    "entry_time": _TIMESTAMP,
    "exit_time": _TIMESTAMP,
    "horizon_minutes": pa.int16(),
    "predicted_long_return": pa.float64(),
    "predicted_short_return": pa.float64(),
    "action": pa.string(),
    "realized_return": pa.float64(),
}


@dataclass(frozen=True)
class PortfolioConfig:
    """Immutable construction and risk rules for the development portfolio."""

    id: str
    prediction_set: str
    model_set: str
    validation_set: str
    symbols: tuple[str, ...]
    horizons_minutes: tuple[int, ...]
    initial_capital_usd: float
    decision_interval_minutes: int
    allocation: str
    overlap_policy: str
    return_accounting: str
    pnl_recognition: str
    fold_state_policy: str
    missing_symbol_policy: str
    annualization_periods: int
    volatility_lookback_periods: int
    volatility_min_observations: int
    target_annual_volatility: float
    warmup_leverage: float
    maximum_leverage: float
    maximum_drawdown: float
    drawdown_policy: str
    locked_test_policy: str

    def __post_init__(self) -> None:
        if self.id != PORTFOLIO_SET_ID:
            raise ValueError(f"portfolio id must be {PORTFOLIO_SET_ID}")
        if (
            self.prediction_set != PREDICTION_SET_ID
            or self.model_set != MODEL_SET_ID
            or self.validation_set != VALIDATION_SET_ID
        ):
            raise ValueError("portfolio prediction provenance is incompatible")
        if self.symbols != PORTFOLIO_SYMBOLS:
            raise ValueError("portfolio must contain the canonical eight symbols")
        if self.horizons_minutes != PORTFOLIO_HORIZONS:
            raise ValueError("portfolio horizons must be 15, 30, and 60 minutes")
        if not math.isfinite(self.initial_capital_usd) or self.initial_capital_usd <= 0:
            raise ValueError("initial capital must be finite and positive")
        if self.decision_interval_minutes != 5:
            raise ValueError("portfolio decisions must use five-minute intervals")
        policies = (
            (self.allocation, "equal_symbol_horizon_overlap_adjusted"),
            (self.overlap_policy, "independent_lots"),
            (self.return_accounting, "normalized_executable_return"),
            (self.pnl_recognition, "actual_exit_time"),
            (self.fold_state_policy, "continuous"),
            (self.missing_symbol_policy, "event_driven"),
            (self.drawdown_policy, "halt_new_positions"),
            (self.locked_test_policy, "forbidden"),
        )
        if any(actual != expected for actual, expected in policies):
            raise ValueError("portfolio policy is not supported")
        if self.annualization_periods <= 0:
            raise ValueError("annualization_periods must be positive")
        if not (
            2 <= self.volatility_min_observations <= self.volatility_lookback_periods
        ):
            raise ValueError("volatility observation window is invalid")
        for value, name in (
            (self.target_annual_volatility, "target annual volatility"),
            (self.warmup_leverage, "warmup leverage"),
            (self.maximum_leverage, "maximum leverage"),
            (self.maximum_drawdown, "maximum drawdown"),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.target_annual_volatility >= 1.0:
            raise ValueError("target annual volatility must be below one")
        if self.warmup_leverage > self.maximum_leverage:
            raise ValueError("warmup leverage cannot exceed maximum leverage")
        if self.maximum_drawdown >= 1.0:
            raise ValueError("maximum drawdown must be below one")

    def lot_weight(self, horizon_minutes: int) -> float:
        """Allocate one sleeve across its maximum scheduled overlapping lots."""
        if horizon_minutes not in self.horizons_minutes:
            raise ValueError(f"unsupported portfolio horizon: {horizon_minutes}")
        overlap_slots = horizon_minutes // self.decision_interval_minutes
        return 1.0 / len(self.symbols) / len(self.horizons_minutes) / overlap_slots


@dataclass(frozen=True)
class PortfolioSimulation:
    """Portfolio event ledger, equity path, and causal risk state."""

    ledger: pa.Table
    equity: pa.Table
    period_returns: tuple[float, ...]
    suppressed_signals: int
    maximum_active_positions: int
    maximum_gross_leverage: float


@dataclass(frozen=True)
class _Lot:
    sequence: int
    fold_id: str
    symbol: str
    decision_time: datetime
    entry_time: datetime
    exit_time: datetime
    horizon_minutes: int
    action: str
    base_weight: float
    risk_leverage: float
    notional_usd: float
    realized_return: float


class _CausalVolatility:
    def __init__(self, config: PortfolioConfig) -> None:
        self._config = config
        self._interval = timedelta(minutes=config.decision_interval_minutes)
        self._rolling: deque[float] = deque(maxlen=config.volatility_lookback_periods)
        self._all: list[float] = []
        self._pending: dict[datetime, tuple[float, float]] = {}
        self._last_observed: datetime | None = None
        self._finalized: set[datetime] = set()

    def _bucket(self, value: datetime) -> datetime:
        minute = value.minute - value.minute % self._config.decision_interval_minutes
        return value.replace(minute=minute, second=0, microsecond=0)

    def record(self, exit_time: datetime, pnl_usd: float, equity_before: float) -> None:
        bucket = self._bucket(exit_time)
        if bucket in self._pending:
            previous_pnl, opening_equity = self._pending[bucket]
            self._pending[bucket] = (previous_pnl + pnl_usd, opening_equity)
        else:
            self._pending[bucket] = (pnl_usd, equity_before)

    def advance(self, boundary: datetime) -> None:
        boundary_bucket = self._bucket(boundary)
        periods = {
            bucket for bucket in self._pending if bucket < boundary_bucket
        }
        if (
            self._last_observed is not None
            and self._last_observed < boundary_bucket
        ):
            periods.add(self._last_observed)
        for period in sorted(periods.difference(self._finalized)):
            pending = self._pending.pop(period, None)
            period_return = (
                pending[0] / pending[1]
                if pending is not None and pending[1] > 0.0
                else 0.0
            )
            self._rolling.append(period_return)
            self._all.append(period_return)
            self._finalized.add(period)
        self._last_observed = boundary_bucket

    @property
    def annualized_volatility(self) -> float | None:
        if len(self._rolling) < self._config.volatility_min_observations:
            return None
        volatility = float(np.std(np.asarray(self._rolling), ddof=1))
        return volatility * math.sqrt(self._config.annualization_periods)

    @property
    def leverage(self) -> float:
        volatility = self.annualized_volatility
        if volatility is None:
            return self._config.warmup_leverage
        if volatility <= 0.0:
            return self._config.maximum_leverage
        return min(
            self._config.maximum_leverage,
            self._config.target_annual_volatility / volatility,
        )

    @property
    def all_returns(self) -> tuple[float, ...]:
        return tuple(self._all)

    def finish(self) -> None:
        for period in sorted(set(self._pending).difference(self._finalized)):
            pnl_usd, opening_equity = self._pending.pop(period)
            period_return = pnl_usd / opening_equity if opening_equity > 0.0 else 0.0
            self._rolling.append(period_return)
            self._all.append(period_return)
            self._finalized.add(period)


def load_portfolio_config(path: Path) -> PortfolioConfig:
    """Load the immutable portfolio construction contract."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Portfolio config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    try:
        return PortfolioConfig(
            id=str(values["id"]),
            prediction_set=str(values["prediction_set"]),
            model_set=str(values["model_set"]),
            validation_set=str(values["validation_set"]),
            symbols=tuple(str(value) for value in values["symbols"]),
            horizons_minutes=tuple(int(value) for value in values["horizons_minutes"]),
            initial_capital_usd=float(values["initial_capital_usd"]),
            decision_interval_minutes=int(values["decision_interval_minutes"]),
            allocation=str(values["allocation"]),
            overlap_policy=str(values["overlap_policy"]),
            return_accounting=str(values["return_accounting"]),
            pnl_recognition=str(values["pnl_recognition"]),
            fold_state_policy=str(values["fold_state_policy"]),
            missing_symbol_policy=str(values["missing_symbol_policy"]),
            annualization_periods=int(values["annualization_periods"]),
            volatility_lookback_periods=int(values["volatility_lookback_periods"]),
            volatility_min_observations=int(values["volatility_min_observations"]),
            target_annual_volatility=float(values["target_annual_volatility"]),
            warmup_leverage=float(values["warmup_leverage"]),
            maximum_leverage=float(values["maximum_leverage"]),
            maximum_drawdown=float(values["maximum_drawdown"]),
            drawdown_policy=str(values["drawdown_policy"]),
            locked_test_policy=str(values["locked_test_policy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid portfolio config field: {error}") from error


def _schema(config: PortfolioConfig, fields: list[pa.Field]) -> pa.Schema:
    return pa.schema(
        fields,
        metadata={
            b"demofml.portfolio_set": config.id.encode(),
            b"demofml.prediction_set": config.prediction_set.encode(),
            b"demofml.model_set": config.model_set.encode(),
            b"demofml.validation_set": config.validation_set.encode(),
            b"demofml.return_accounting": config.return_accounting.encode(),
        },
    )


def ledger_schema(config: PortfolioConfig) -> pa.Schema:
    """Build the immutable settled-lot schema."""
    return _schema(
        config,
        [
            pa.field("fold_id", pa.string(), nullable=False),
            pa.field("symbol", pa.string(), nullable=False),
            pa.field("decision_time", _TIMESTAMP, nullable=False),
            pa.field("entry_time", _TIMESTAMP, nullable=False),
            pa.field("exit_time", _TIMESTAMP, nullable=False),
            pa.field("horizon_minutes", pa.int16(), nullable=False),
            pa.field("action", pa.string(), nullable=False),
            pa.field("base_weight", pa.float64(), nullable=False),
            pa.field("risk_leverage", pa.float64(), nullable=False),
            pa.field("notional_usd", pa.float64(), nullable=False),
            pa.field("realized_return", pa.float64(), nullable=False),
            pa.field("pnl_usd", pa.float64(), nullable=False),
            pa.field("equity_after_exit", pa.float64(), nullable=False),
            pa.field("drawdown_after_exit", pa.float64(), nullable=False),
        ],
    )


def equity_schema(config: PortfolioConfig) -> pa.Schema:
    """Build the immutable event-time equity schema."""
    return _schema(
        config,
        [
            pa.field("event_time", _TIMESTAMP, nullable=False),
            pa.field("pnl_usd", pa.float64(), nullable=False),
            pa.field("equity_usd", pa.float64(), nullable=False),
            pa.field("running_peak_usd", pa.float64(), nullable=False),
            pa.field("drawdown", pa.float64(), nullable=False),
            pa.field("active_positions", pa.int32(), nullable=False),
            pa.field("gross_notional_usd", pa.float64(), nullable=False),
            pa.field("halted", pa.bool_(), nullable=False),
        ],
    )


def _validate_prediction_schema(table: pa.Table, config: PortfolioConfig) -> None:
    metadata = table.schema.metadata or {}
    for key, expected in (
        (b"demofml.prediction_set", config.prediction_set),
        (b"demofml.model_set", config.model_set),
        (b"demofml.validation_set", config.validation_set),
    ):
        if metadata.get(key) != expected.encode():
            raise ValueError(f"prediction metadata mismatch for {key.decode()}")
    for name, expected_type in _PREDICTION_FIELDS.items():
        if name not in table.column_names:
            raise ValueError(f"prediction schema is missing {name}")
        field = table.schema.field(name)
        if field.type != expected_type or field.nullable:
            raise ValueError(f"prediction field {name} has an invalid contract")


def _integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    return value


def _prediction_rows(
    tables: Sequence[pa.Table],
    config: PortfolioConfig,
    locked_test_start: datetime,
) -> list[dict[str, object]]:
    if not tables:
        raise ValueError("at least one prediction table is required")
    rows: list[dict[str, object]] = []
    for table in tables:
        _validate_prediction_schema(table, config)
        rows.extend(table.select(list(_PREDICTION_FIELDS)).to_pylist())
    if not rows:
        raise ValueError("portfolio predictions cannot be empty")

    symbols: set[str] = set()
    horizons: set[int] = set()
    keys: set[tuple[str, datetime, int]] = set()
    for row in rows:
        symbol = str(row["symbol"])
        model_set = str(row["model_set"])
        validation_set = str(row["validation_set"])
        fold_id = str(row["fold_id"])
        action = str(row["action"])
        horizon = _integer(row["horizon_minutes"], "horizon_minutes")
        decision_time = row["decision_time"]
        entry_time = row["entry_time"]
        exit_time = row["exit_time"]
        if symbol not in config.symbols or not fold_id:
            raise ValueError("prediction symbol or fold is invalid")
        if model_set != config.model_set or validation_set != config.validation_set:
            raise ValueError("prediction row provenance is incompatible")
        if horizon not in config.horizons_minutes or action not in _ACTIONS:
            raise ValueError("prediction horizon or action is invalid")
        if not all(
            isinstance(value, datetime)
            for value in (decision_time, entry_time, exit_time)
        ):
            raise ValueError("prediction execution timestamps cannot be null")
        if not isinstance(decision_time, datetime):
            raise ValueError("decision_time cannot be null")
        if not isinstance(entry_time, datetime) or not isinstance(exit_time, datetime):
            raise ValueError("execution times cannot be null")
        if decision_time >= locked_test_start or exit_time >= locked_test_start:
            raise ValueError("locked-test predictions are forbidden")
        entry_deadline = decision_time + timedelta(minutes=MAX_QUOTE_LATENCY_MINUTES)
        exit_target = decision_time + timedelta(minutes=horizon)
        exit_deadline = exit_target + timedelta(minutes=MAX_QUOTE_LATENCY_MINUTES)
        if not decision_time <= entry_time <= entry_deadline:
            raise ValueError("prediction entry_time violates executable latency")
        if not exit_target <= exit_time <= exit_deadline:
            raise ValueError("prediction exit_time violates executable latency")
        realized_return = row["realized_return"]
        if not isinstance(realized_return, int | float) or not math.isfinite(
            realized_return
        ):
            raise ValueError("prediction realized_return must be finite")
        for name in ("predicted_long_return", "predicted_short_return"):
            prediction = row[name]
            if not isinstance(prediction, int | float) or not math.isfinite(
                prediction
            ):
                raise ValueError(f"{name} must be finite")
        if action == "flat" and realized_return != 0.0:
            raise ValueError("flat predictions must have zero realized_return")
        key = (symbol, decision_time, horizon)
        if key in keys:
            raise ValueError("duplicate portfolio prediction key")
        keys.add(key)
        symbols.add(symbol)
        horizons.add(horizon)
    if symbols != set(config.symbols):
        raise ValueError("predictions do not cover the canonical symbol universe")
    if horizons != set(config.horizons_minutes):
        raise ValueError("predictions do not cover every portfolio horizon")
    rows.sort(
        key=lambda row: (
            row["decision_time"],
            str(row["symbol"]),
            _integer(row["horizon_minutes"], "horizon_minutes"),
        )
    )
    return rows


def simulate_portfolio(
    prediction_tables: Sequence[pa.Table],
    config: PortfolioConfig,
    locked_test_start: datetime,
) -> PortfolioSimulation:
    """Settle independent lots with causal sizing and permanent drawdown halt."""
    rows = _prediction_rows(prediction_tables, config, locked_test_start)
    first_time = rows[0]["decision_time"]
    if not isinstance(first_time, datetime):
        raise ValueError("first prediction time is invalid")

    equity = config.initial_capital_usd
    running_peak = equity
    halted = False
    suppressed_signals = 0
    active_notional = 0.0
    maximum_active = 0
    maximum_gross_leverage = 0.0
    sequence = 0
    pending_entries: list[tuple[datetime, int, _Lot]] = []
    open_lots: list[tuple[datetime, int, _Lot]] = []
    ledger_rows: list[dict[str, object]] = []
    equity_rows: list[dict[str, object]] = [
        {
            "event_time": first_time,
            "pnl_usd": 0.0,
            "equity_usd": equity,
            "running_peak_usd": running_peak,
            "drawdown": 0.0,
            "active_positions": 0,
            "gross_notional_usd": 0.0,
            "halted": False,
        }
    ]
    volatility = _CausalVolatility(config)

    def observe_exposure() -> None:
        nonlocal maximum_active, maximum_gross_leverage
        maximum_active = max(maximum_active, len(open_lots))
        if equity > 0.0:
            maximum_gross_leverage = max(
                maximum_gross_leverage, active_notional / equity
            )

    def settle(exit_time: datetime) -> None:
        nonlocal equity, running_peak, halted, active_notional
        settled: list[_Lot] = []
        while open_lots and open_lots[0][0] == exit_time:
            _, _, lot = heapq.heappop(open_lots)
            settled.append(lot)
        equity_before = equity
        pnl_usd = sum(lot.notional_usd * lot.realized_return for lot in settled)
        equity += pnl_usd
        active_notional -= sum(lot.notional_usd for lot in settled)
        running_peak = max(running_peak, equity)
        drawdown = 1.0 - equity / running_peak
        if drawdown >= config.maximum_drawdown:
            halted = True
        volatility.record(exit_time, pnl_usd, equity_before)
        for lot in settled:
            ledger_rows.append(
                {
                    "fold_id": lot.fold_id,
                    "symbol": lot.symbol,
                    "decision_time": lot.decision_time,
                    "entry_time": lot.entry_time,
                    "exit_time": lot.exit_time,
                    "horizon_minutes": lot.horizon_minutes,
                    "action": lot.action,
                    "base_weight": lot.base_weight,
                    "risk_leverage": lot.risk_leverage,
                    "notional_usd": lot.notional_usd,
                    "realized_return": lot.realized_return,
                    "pnl_usd": lot.notional_usd * lot.realized_return,
                    "equity_after_exit": equity,
                    "drawdown_after_exit": drawdown,
                }
            )
        observe_exposure()
        equity_rows.append(
            {
                "event_time": exit_time,
                "pnl_usd": pnl_usd,
                "equity_usd": equity,
                "running_peak_usd": running_peak,
                "drawdown": drawdown,
                "active_positions": len(open_lots),
                "gross_notional_usd": max(active_notional, 0.0),
                "halted": halted,
            }
        )

    def activate(entry_time: datetime) -> None:
        nonlocal active_notional, suppressed_signals
        entering: list[_Lot] = []
        while pending_entries and pending_entries[0][0] == entry_time:
            _, _, lot = heapq.heappop(pending_entries)
            entering.append(lot)
        if halted:
            suppressed_signals += len(entering)
            return
        for lot in entering:
            heapq.heappush(open_lots, (lot.exit_time, lot.sequence, lot))
            active_notional += lot.notional_usd
        observe_exposure()

    def process_until(boundary: datetime | None) -> None:
        while pending_entries or open_lots:
            next_entry = pending_entries[0][0] if pending_entries else None
            next_exit = open_lots[0][0] if open_lots else None
            candidates = [
                value for value in (next_entry, next_exit) if value is not None
            ]
            if not candidates:
                return
            event_time = min(candidates)
            if boundary is not None and event_time >= boundary:
                return
            if next_entry == event_time:
                activate(event_time)
            if next_exit == event_time:
                settle(event_time)

    index = 0
    while index < len(rows):
        decision_time = rows[index]["decision_time"]
        if not isinstance(decision_time, datetime):
            raise ValueError("decision_time cannot be null")
        process_until(decision_time)
        volatility.advance(decision_time)
        group_end = index + 1
        while (
            group_end < len(rows) and rows[group_end]["decision_time"] == decision_time
        ):
            group_end += 1
        decision_rows = rows[index:group_end]
        if halted:
            suppressed_signals += sum(
                str(row["action"]) != "flat" for row in decision_rows
            )
            index = group_end
            continue
        risk_leverage = volatility.leverage
        for row in decision_rows:
            action = str(row["action"])
            if action == "flat":
                continue
            horizon = _integer(row["horizon_minutes"], "horizon_minutes")
            entry_time = row["entry_time"]
            exit_time = row["exit_time"]
            realized_return = row["realized_return"]
            if not isinstance(entry_time, datetime) or not isinstance(
                exit_time, datetime
            ):
                raise ValueError("execution times cannot be null")
            if not isinstance(realized_return, int | float):
                raise ValueError("realized_return must be numeric")
            base_weight = config.lot_weight(horizon)
            notional_usd = max(equity, 0.0) * base_weight * risk_leverage
            lot = _Lot(
                sequence=sequence,
                fold_id=str(row["fold_id"]),
                symbol=str(row["symbol"]),
                decision_time=decision_time,
                entry_time=entry_time,
                exit_time=exit_time,
                horizon_minutes=horizon,
                action=action,
                base_weight=base_weight,
                risk_leverage=risk_leverage,
                notional_usd=notional_usd,
                realized_return=float(realized_return),
            )
            heapq.heappush(pending_entries, (entry_time, sequence, lot))
            sequence += 1
        index = group_end

    process_until(None)
    volatility.finish()
    return PortfolioSimulation(
        ledger=pa.Table.from_pylist(ledger_rows, schema=ledger_schema(config)),
        equity=pa.Table.from_pylist(equity_rows, schema=equity_schema(config)),
        period_returns=volatility.all_returns,
        suppressed_signals=suppressed_signals,
        maximum_active_positions=maximum_active,
        maximum_gross_leverage=maximum_gross_leverage,
    )

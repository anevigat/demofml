"""Versioned purged walk-forward folds and leakage controls."""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]

VALIDATION_SET_ID = "purged-walk-forward-v1"
VALIDATION_STRATEGY = "expanding"
INTERVAL_SEMANTICS = "half_open_utc"


def _require_utc(value: datetime, name: str) -> None:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _add_months(value: datetime, months: int) -> datetime:
    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    try:
        return value.replace(year=year, month=month)
    except ValueError as error:
        raise ValueError("monthly fold boundaries must preserve the day") from error


def _parse_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be a valid ISO-8601 timestamp") from error
    _require_utc(parsed, name)
    return parsed.astimezone(UTC)


@dataclass(frozen=True)
class TemporalFold:
    """One expanding training interval and its purged validation interval."""

    id: str
    train_start: datetime
    train_end_exclusive: datetime
    validation_start: datetime
    validation_end_exclusive: datetime
    purge_minutes: int

    def __post_init__(self) -> None:
        for name in (
            "train_start",
            "train_end_exclusive",
            "validation_start",
            "validation_end_exclusive",
        ):
            _require_utc(getattr(self, name), name)
        if self.purge_minutes <= 0:
            raise ValueError("purge_minutes must be positive")
        if not self.train_start < self.train_end_exclusive:
            raise ValueError("training interval must be non-empty")
        if not self.train_end_exclusive < self.validation_start:
            raise ValueError("training and validation require a purge interval")
        if not self.validation_start < self.validation_end_exclusive:
            raise ValueError("validation interval must be non-empty")
        expected_end = self.validation_start - timedelta(minutes=self.purge_minutes)
        if self.train_end_exclusive != expected_end:
            raise ValueError("training interval does not match purge_minutes")


@dataclass(frozen=True)
class FoldSelection:
    """Row positions selected for one temporal fold."""

    train: tuple[int, ...]
    validation: tuple[int, ...]


@dataclass(frozen=True)
class ValidationPlan:
    """Immutable parameters for purged expanding walk-forward validation."""

    id: str
    strategy: str
    interval_semantics: str
    feature_set: str
    label_set: str
    train_start: datetime
    first_validation_start: datetime
    validation_window_months: int
    step_months: int
    development_end_exclusive: datetime
    purge_minutes: int
    max_horizon_minutes: int
    max_quote_latency_minutes: int
    locked_test_start: datetime
    locked_test_end_exclusive: datetime

    def __post_init__(self) -> None:
        for name in (
            "train_start",
            "first_validation_start",
            "development_end_exclusive",
            "locked_test_start",
            "locked_test_end_exclusive",
        ):
            _require_utc(getattr(self, name), name)
        if self.id != VALIDATION_SET_ID:
            raise ValueError(f"validation id must be {VALIDATION_SET_ID}")
        if self.strategy != VALIDATION_STRATEGY:
            raise ValueError("only expanding walk-forward validation is supported")
        if self.interval_semantics != INTERVAL_SEMANTICS:
            raise ValueError("validation intervals must be half-open UTC")
        if not self.feature_set or not self.label_set:
            raise ValueError("feature_set and label_set must be identified")
        if self.validation_window_months <= 0 or self.step_months <= 0:
            raise ValueError("validation and step months must be positive")
        if self.step_months < self.validation_window_months:
            raise ValueError("validation folds cannot overlap")
        if self.max_horizon_minutes <= 0 or self.max_quote_latency_minutes < 0:
            raise ValueError("horizon must be positive and latency non-negative")
        required_purge = self.max_horizon_minutes + self.max_quote_latency_minutes
        if self.purge_minutes < required_purge:
            raise ValueError(
                f"purge_minutes must cover the {required_purge}-minute label window"
            )
        if not self.train_start < self.first_validation_start:
            raise ValueError("initial training interval must be non-empty")
        if not self.first_validation_start < self.development_end_exclusive:
            raise ValueError("development validation interval must be non-empty")
        if self.development_end_exclusive != self.locked_test_start:
            raise ValueError("development must end exactly when the locked test starts")
        if not self.locked_test_start < self.locked_test_end_exclusive:
            raise ValueError("locked test interval must be non-empty")
        if self.development_decision_end <= self.first_validation_start:
            raise ValueError("purge leaves no development validation decisions")
        if self.locked_test_decision_end <= self.locked_test_start:
            raise ValueError("purge leaves no locked test decisions")

    @property
    def information_window(self) -> timedelta:
        """Maximum time needed to resolve one executable label."""
        return timedelta(
            minutes=self.max_horizon_minutes + self.max_quote_latency_minutes
        )

    @property
    def development_decision_end(self) -> datetime:
        """Last exclusive decision boundary that cannot read locked-test data."""
        return self.locked_test_start - self.information_window

    @property
    def locked_test_decision_end(self) -> datetime:
        """Last exclusive decision boundary resolvable inside the locked period."""
        return self.locked_test_end_exclusive - self.information_window

    def folds(self) -> tuple[TemporalFold, ...]:
        """Generate deterministic expanding folds from the versioned plan."""
        folds: list[TemporalFold] = []
        validation_start = self.first_validation_start
        while validation_start < self.development_decision_end:
            nominal_end = _add_months(validation_start, self.validation_window_months)
            validation_end = min(nominal_end, self.development_decision_end)
            folds.append(
                TemporalFold(
                    id=f"wf-{validation_start:%Y-%m}",
                    train_start=self.train_start,
                    train_end_exclusive=validation_start
                    - timedelta(minutes=self.purge_minutes),
                    validation_start=validation_start,
                    validation_end_exclusive=validation_end,
                    purge_minutes=self.purge_minutes,
                )
            )
            validation_start = _add_months(validation_start, self.step_months)
        if not folds:
            raise ValueError("validation plan generates no folds")
        return tuple(folds)


def load_validation_plan(path: Path) -> ValidationPlan:
    """Load and validate a versioned walk-forward TOML definition."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Validation config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    try:
        return ValidationPlan(
            id=str(values["id"]),
            strategy=str(values["strategy"]),
            interval_semantics=str(values["interval_semantics"]),
            feature_set=str(values["feature_set"]),
            label_set=str(values["label_set"]),
            train_start=_parse_utc(values["train_start"], "train_start"),
            first_validation_start=_parse_utc(
                values["first_validation_start"], "first_validation_start"
            ),
            validation_window_months=int(values["validation_window_months"]),
            step_months=int(values["step_months"]),
            development_end_exclusive=_parse_utc(
                values["development_end_exclusive"],
                "development_end_exclusive",
            ),
            purge_minutes=int(values["purge_minutes"]),
            max_horizon_minutes=int(values["max_horizon_minutes"]),
            max_quote_latency_minutes=int(values["max_quote_latency_minutes"]),
            locked_test_start=_parse_utc(
                values["locked_test_start"], "locked_test_start"
            ),
            locked_test_end_exclusive=_parse_utc(
                values["locked_test_end_exclusive"],
                "locked_test_end_exclusive",
            ),
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"invalid validation config field: {error}") from error


def _validate_decision_times(decision_times: Sequence[datetime]) -> None:
    previous: datetime | None = None
    for decision_time in decision_times:
        _require_utc(decision_time, "decision_time")
        if previous is not None and decision_time <= previous:
            raise ValueError("decision times must be strictly ordered")
        previous = decision_time


def select_fold_rows(
    decision_times: Sequence[datetime], fold: TemporalFold
) -> FoldSelection:
    """Select ordered row positions without admitting the purge interval."""
    _validate_decision_times(decision_times)
    train: list[int] = []
    validation: list[int] = []
    for index, decision_time in enumerate(decision_times):
        if fold.train_start <= decision_time < fold.train_end_exclusive:
            train.append(index)
        elif fold.validation_start <= decision_time < fold.validation_end_exclusive:
            validation.append(index)
    return FoldSelection(tuple(train), tuple(validation))


def select_locked_test_rows(
    decision_times: Sequence[datetime], plan: ValidationPlan
) -> tuple[int, ...]:
    """Select locked decisions whose full labels remain inside the lock."""
    _validate_decision_times(decision_times)
    return tuple(
        index
        for index, decision_time in enumerate(decision_times)
        if plan.locked_test_start <= decision_time < plan.locked_test_decision_end
    )


def validate_feature_label_schemas(
    feature_schema: pa.Schema,
    label_schema: pa.Schema,
    plan: ValidationPlan,
) -> None:
    """Reject feature/label contracts incompatible with a validation plan."""
    feature_metadata = feature_schema.metadata or {}
    label_metadata = label_schema.metadata or {}
    if feature_metadata.get(b"demofml.feature_set") != plan.feature_set.encode():
        raise ValueError("feature schema does not match validation feature_set")
    if label_metadata.get(b"demofml.label_set") != plan.label_set.encode():
        raise ValueError("label schema does not match validation label_set")
    if feature_metadata.get(b"demofml.source_bar_set") != label_metadata.get(
        b"demofml.source_bar_set"
    ):
        raise ValueError("feature and label source bar sets differ")
    feature_interval = feature_metadata.get(b"demofml.source_bar_interval_minutes")
    label_interval = label_metadata.get(b"demofml.source_bar_interval_minutes")
    if feature_interval != label_interval:
        raise ValueError("feature and label bar intervals differ")
    try:
        horizons = tuple(
            int(value)
            for value in label_metadata[b"demofml.horizons_minutes"].split(b",")
        )
        latency = int(label_metadata[b"demofml.max_quote_latency_minutes"])
    except (KeyError, ValueError) as error:
        raise ValueError("label schema lacks leakage-window metadata") from error
    if not horizons or max(horizons) != plan.max_horizon_minutes:
        raise ValueError("label horizons do not match validation plan")
    if latency != plan.max_quote_latency_minutes:
        raise ValueError("label quote latency does not match validation plan")

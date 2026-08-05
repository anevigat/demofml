"""Standalone exploratory Campaign 2 cross-pair historical screen."""

from __future__ import annotations

import argparse
import hashlib
import math
import platform
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.impute import SimpleImputer  # type: ignore[import-untyped]
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from demofml.calendars.prospective_fx import (
    CALENDAR_ID,
    expected_decision_boundaries,
    is_expected_decision_boundary,
)
from demofml.features.causal import FEATURE_SCHEMA
from demofml.features.cross_pair import (
    CROSS_PAIR_COLUMNS,
    PAIRS,
    CrossPairFactorState,
)
from demofml.labels.executable import label_schema
from demofml.models.baseline import FEATURE_COLUMNS
from demofml.prospective.records import (
    IMAGE_DIGEST_PATTERN,
    canonical_json,
    content_id,
    write_immutable_json,
)

SCREEN_ID = "campaign2-cross-pair-historical-screen-2020-v1"
FROZEN_CONFIG_SHA256 = (
    "1a2bdf972bbafc350ee4b4038580ebaf44c6148a5e55c9121fa9ad810d5d4d2b"
)
CONTROL_ARM = "control"
CANDIDATE_ARM = "candidate"
ARMS = (CONTROL_ARM, CANDIDATE_ARM)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_TOP_LEVEL_KEYS = {
    "format_version",
    "id",
    "status",
    "declared_at",
    "source_dataset",
    "source_feature_set",
    "source_label_set",
    "candidate_feature_set",
    "symbols",
    "horizons_minutes",
    "train_start",
    "train_end_exclusive",
    "screen_start",
    "decision_end_exclusive",
    "source_end_exclusive",
    "locked_test_start",
    "purge_minutes",
    "model_type",
    "alpha",
    "solver",
    "imputation",
    "standardize",
    "action_threshold_bps",
    "random_seed",
    "missing_opportunity_return",
    "denominator",
    "authorization",
    "feature_sha256",
    "label_sha256",
    "expected_rows",
}
_AUTHORIZATION = {
    "historical_fitting": True,
    "historical_scoring": True,
    "historical_evaluation": True,
    "locked_test_access": False,
    "prospective_collection": False,
    "prospective_scoring": False,
    "confirmatory_claim": False,
}
_EXPECTED_FEATURE_SHA256 = {
    "AUDUSD": "fb3b00d3ac128b77697f7247b3f02fcfa5310a6751d23f47fbda458a07af2ba5",
    "EURCHF": "6867a836247e5725c4264753efc78a793e68fbeaafa2e80268f4cac3c6508ce0",
    "EURJPY": "0f20d89c5f2d5f621c84c7c5bbf513f6f4f66277667eae4dfed126367d81b43c",
    "EURUSD": "e7fe7767f36bb55fb1ea771d809ae151adafe7f3fee84b9a08b2bd85c4ed9b90",
    "GBPJPY": "2082ef278bfbd05b468704bddd5d19095d6661a9715ef800609177ce30a6859c",
    "GBPUSD": "eff4db935d8621bfe7557a89a3481bd2c3d84b6348ba9e1d513ca6236e6fab93",
    "USDCAD": "22e2858d32a1ca97ccbc50fb695b0fa145f2dee981d3eff742d763edc1d3c7bd",
    "USDJPY": "752cd67d198462a2c4ad71b39b78f386ef37e45a5b8c991964a6e3aeb4be7c7e",
}
_EXPECTED_LABEL_SHA256 = {
    "AUDUSD": "17c710c1150168e393c4ea1cde79919e7c56b70a13afeedf69c93f14e96a8ef3",
    "EURCHF": "290848e1f8eecfcc3a49c1c0ff5ad99724b61e6ad31f3a94273a4564d9b91bdb",
    "EURJPY": "0c2dc9d6bf8eed647de150018128c83efcccbbf103763b5481a45a447e4fb109",
    "EURUSD": "bb2a95017583fa0ade20d3a39437ac6406f57e6569af7f0afbc4f101dc67c172",
    "GBPJPY": "4bc29c96cf946c91742965ba85f084424aed69f9181a2827b32bd2fe9c1f4ba6",
    "GBPUSD": "93a4286237b119ae18aa761edcffffbe40a5e847b40f63a497bd6318b42d5c25",
    "USDCAD": "5f3964f6194a0aecc673e772ae2043a3355476304fb4d051bab52e0260dbc277",
    "USDJPY": "b7a73bed93072d7c40ee9d2e8d24c156de2156b9f62d4c9c6d1fc71a66af6830",
}
_EXPECTED_ROWS = {
    "AUDUSD": 520792,
    "EURCHF": 520423,
    "EURJPY": 520588,
    "EURUSD": 523696,
    "GBPJPY": 520497,
    "GBPUSD": 522378,
    "USDCAD": 520718,
    "USDJPY": 520915,
}


def _parse_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} is not a valid timestamp") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")
    return parsed


def _strict_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


@dataclass(frozen=True)
class HistoricalScreenConfig:
    """Strict frozen behavior and input inventory for the exploratory screen."""

    id: str
    declared_at: datetime
    symbols: tuple[str, ...]
    horizons_minutes: tuple[int, ...]
    train_start: datetime
    train_end_exclusive: datetime
    screen_start: datetime
    decision_end_exclusive: datetime
    source_end_exclusive: datetime
    locked_test_start: datetime
    purge_minutes: int
    alpha: float
    random_seed: int
    feature_sha256: Mapping[str, str]
    label_sha256: Mapping[str, str]
    expected_rows: Mapping[str, int]

    def __post_init__(self) -> None:
        if (
            self.id != SCREEN_ID
            or self.declared_at != datetime(2026, 8, 5, 11, 24, 26, tzinfo=UTC)
            or self.symbols != PAIRS
            or self.horizons_minutes != (15, 30, 60)
            or self.train_start != datetime(2018, 1, 1, tzinfo=UTC)
            or self.train_end_exclusive != datetime(2019, 12, 31, 22, 55, tzinfo=UTC)
            or self.screen_start != datetime(2020, 1, 1, tzinfo=UTC)
            or self.decision_end_exclusive != datetime(2020, 12, 31, 22, 55, tzinfo=UTC)
            or self.source_end_exclusive != datetime(2021, 1, 1, tzinfo=UTC)
            or self.locked_test_start != datetime(2025, 1, 1, tzinfo=UTC)
            or self.purge_minutes != 65
            or self.alpha != 1.0
            or self.random_seed != 1729
        ):
            raise ValueError("historical screen contract is incompatible")
        if self.screen_start - self.train_end_exclusive != timedelta(
            minutes=self.purge_minutes
        ):
            raise ValueError("historical screen purge is incompatible")
        expected = set(PAIRS)
        for name, values in (
            ("feature_sha256", self.feature_sha256),
            ("label_sha256", self.label_sha256),
            ("expected_rows", self.expected_rows),
        ):
            if set(values) != expected:
                raise ValueError(f"{name} must contain exactly the canonical symbols")
        if any(not _SHA256.fullmatch(value) for value in self.feature_sha256.values()):
            raise ValueError("feature_sha256 contains an invalid digest")
        if any(not _SHA256.fullmatch(value) for value in self.label_sha256.values()):
            raise ValueError("label_sha256 contains an invalid digest")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.expected_rows.values()
        ):
            raise ValueError("expected_rows must contain positive integers")
        if (
            dict(self.feature_sha256) != _EXPECTED_FEATURE_SHA256
            or dict(self.label_sha256) != _EXPECTED_LABEL_SHA256
            or dict(self.expected_rows) != _EXPECTED_ROWS
        ):
            raise ValueError("historical screen input inventory is incompatible")


def _require_exact_scalar(
    values: Mapping[str, object], name: str, expected: object
) -> None:
    value = values.get(name)
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{name} does not match the frozen screen")


def _string_map(value: object, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError(f"{name} must be a string table")
    return dict(value)


def _row_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, int)
        for key, item in value.items()
    ):
        raise ValueError("expected_rows must be an integer table")
    return dict(value)


def load_historical_screen_config(path: Path) -> HistoricalScreenConfig:
    """Load the exact Campaign 2 exploratory screen contract."""
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise RuntimeError(f"Historical screen config is not a file: {path}")
    if _file_sha256(path) != FROZEN_CONFIG_SHA256:
        raise ValueError("historical screen config differs from the frozen contract")
    with path.open("rb") as source:
        values = tomllib.load(source)
    if set(values) != _TOP_LEVEL_KEYS:
        missing = sorted(_TOP_LEVEL_KEYS.difference(values))
        extra = sorted(set(values).difference(_TOP_LEVEL_KEYS))
        raise ValueError(
            f"historical screen config keys differ; missing={missing}, extra={extra}"
        )
    for name, expected in {
        "format_version": 1,
        "id": SCREEN_ID,
        "status": "exploratory_only_non_confirmatory",
        "source_dataset": "cleaned-ticks-development-v1",
        "source_feature_set": "causal-v1",
        "source_label_set": "executable-v1",
        "candidate_feature_set": "causal-v1-cross-pair-historical-screen-v1",
        "model_type": "ridge",
        "solver": "lsqr",
        "imputation": "training_median",
        "standardize": True,
        "action_threshold_bps": 0.0,
        "missing_opportunity_return": 0.0,
        "denominator": "all_expected_calendar_boundaries_per_symbol_horizon",
    }.items():
        _require_exact_scalar(values, name, expected)
    if values["authorization"] != _AUTHORIZATION:
        raise ValueError("authorization does not match the frozen screen")
    symbols = values["symbols"]
    horizons = values["horizons_minutes"]
    if not isinstance(symbols, list) or any(
        not isinstance(item, str) for item in symbols
    ):
        raise ValueError("symbols must be a string array")
    if not isinstance(horizons, list) or any(
        isinstance(item, bool) or not isinstance(item, int) for item in horizons
    ):
        raise ValueError("horizons_minutes must be an integer array")
    return HistoricalScreenConfig(
        id=str(values["id"]),
        declared_at=_parse_utc(values["declared_at"], "declared_at"),
        symbols=tuple(symbols),
        horizons_minutes=tuple(horizons),
        train_start=_parse_utc(values["train_start"], "train_start"),
        train_end_exclusive=_parse_utc(
            values["train_end_exclusive"], "train_end_exclusive"
        ),
        screen_start=_parse_utc(values["screen_start"], "screen_start"),
        decision_end_exclusive=_parse_utc(
            values["decision_end_exclusive"], "decision_end_exclusive"
        ),
        source_end_exclusive=_parse_utc(
            values["source_end_exclusive"], "source_end_exclusive"
        ),
        locked_test_start=_parse_utc(values["locked_test_start"], "locked_test_start"),
        purge_minutes=_strict_int(values["purge_minutes"], "purge_minutes"),
        alpha=_strict_number(values["alpha"], "alpha"),
        random_seed=_strict_int(values["random_seed"], "random_seed"),
        feature_sha256=_string_map(values["feature_sha256"], "feature_sha256"),
        label_sha256=_string_map(values["label_sha256"], "label_sha256"),
        expected_rows=_row_map(values["expected_rows"]),
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    before = path.stat()
    with path.open("rb") as source:
        while block := source.read(8 * 1024 * 1024):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while hashing: {path}")
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return canonical_json(value)


def _validate_path_map(paths: Mapping[str, Path], name: str) -> dict[str, Path]:
    if set(paths) != set(PAIRS):
        raise ValueError(f"{name} paths must contain exactly the canonical symbols")
    requested = {symbol: paths[symbol].expanduser().absolute() for symbol in PAIRS}
    if any(path.is_symlink() for path in requested.values()):
        raise ValueError(f"{name} inputs cannot be symlinks")
    resolved = {symbol: requested[symbol].resolve() for symbol in PAIRS}
    for symbol, path in resolved.items():
        if not path.is_file():
            raise RuntimeError(f"{name} input for {symbol} is not a file: {path}")
    return resolved


def verify_historical_inputs(
    feature_paths: Mapping[str, Path],
    label_paths: Mapping[str, Path],
    config: HistoricalScreenConfig,
) -> dict[str, dict[str, str]]:
    """Verify all sixteen complete files without selecting outcome rows."""
    features = _validate_path_map(feature_paths, "feature")
    labels = _validate_path_map(label_paths, "label")
    expected_label_schema = label_schema(config.horizons_minutes)
    inventory: dict[str, dict[str, str]] = {"features": {}, "labels": {}}
    for kind, paths, expected_hashes, expected_schema in (
        ("features", features, config.feature_sha256, FEATURE_SCHEMA),
        ("labels", labels, config.label_sha256, expected_label_schema),
    ):
        for symbol in PAIRS:
            path = paths[symbol]
            digest = _file_sha256(path)
            if digest != expected_hashes[symbol]:
                raise ValueError(f"{kind} hash mismatch for {symbol}")
            metadata = pq.read_metadata(path)
            if int(metadata.num_rows) != config.expected_rows[symbol]:
                raise ValueError(f"{kind} row count mismatch for {symbol}")
            if not metadata.schema.to_arrow_schema().equals(
                expected_schema, check_metadata=True
            ):
                raise ValueError(f"{kind} schema or metadata mismatch for {symbol}")
            inventory[kind][symbol] = digest
    return inventory


def _predicate_read(
    path: Path,
    timestamp_column: str,
    start: datetime,
    end_exclusive: datetime,
) -> pa.Table:
    return pq.read_table(
        path,
        filters=[
            (timestamp_column, ">=", start),
            (timestamp_column, "<", end_exclusive),
        ],
    )


def _validate_selected_rows(
    table: pa.Table,
    symbol: str,
    timestamp_column: str,
    start: datetime,
    end_exclusive: datetime,
    config: HistoricalScreenConfig,
) -> tuple[datetime, ...]:
    if table.num_rows and set(table.column("symbol").to_pylist()) != {symbol}:
        raise ValueError(f"selected {symbol} rows contain another symbol")
    times = tuple(table.column(timestamp_column).to_pylist())
    previous: datetime | None = None
    for value in times:
        if not isinstance(value, datetime):
            raise ValueError(f"selected {symbol} timestamp is null")
        value = value.astimezone(UTC)
        if (
            not start <= value < end_exclusive
            or value >= config.source_end_exclusive
            or value >= config.locked_test_start
        ):
            raise ValueError(f"selected {symbol} row crosses the source cutoff")
        if previous is not None and value <= previous:
            raise ValueError(f"selected {symbol} keys are not strictly ordered")
        if not is_expected_decision_boundary(value):
            raise ValueError(f"selected {symbol} key is outside the expected calendar")
        previous = value
    return tuple(value.astimezone(UTC) for value in times)


@dataclass(frozen=True)
class FactorReadyFeatures:
    """Identically keyed control and candidate matrices for one symbol."""

    decision_times: tuple[datetime, ...]
    control: NDArray[np.float64]
    candidate: NDArray[np.float64]


class _CrossPairWindows:
    def __init__(self) -> None:
        self._state = CrossPairFactorState()

    def clear(self) -> None:
        self._state.clear()

    def push(self, pair_returns: Mapping[str, float]) -> dict[str, tuple[float, ...]]:
        values = self._state.push(pair_returns)
        return {
            symbol: tuple(float("nan") if value is None else value for value in row)
            for symbol, row in values.items()
        }


def synchronize_historical_features(
    tables: Mapping[str, pa.Table], config: HistoricalScreenConfig
) -> dict[str, FactorReadyFeatures]:
    """Build paired factors on the expected calendar with reset-on-gap semantics."""
    if set(tables) != set(PAIRS):
        raise ValueError("feature tables must contain exactly the canonical symbols")
    times: dict[str, tuple[datetime, ...]] = {}
    matrices: dict[str, NDArray[np.float64]] = {}
    for symbol in PAIRS:
        table = tables[symbol]
        if not table.schema.equals(FEATURE_SCHEMA, check_metadata=True):
            raise ValueError(f"feature schema or metadata mismatch for {symbol}")
        times[symbol] = _validate_selected_rows(
            table,
            symbol,
            "bar_end",
            config.train_start,
            config.source_end_exclusive,
            config,
        )
        matrix = np.column_stack(
            [
                np.asarray(
                    table.column(name).to_numpy(zero_copy_only=False), dtype=float
                )
                for name in FEATURE_COLUMNS
            ]
        )
        if np.isinf(matrix).any():
            raise ValueError(f"features contain infinity for {symbol}")
        matrices[symbol] = matrix

    calendar = expected_decision_boundaries(
        config.train_start, config.source_end_exclusive
    )
    pointers = {symbol: 0 for symbol in PAIRS}
    output_times: list[datetime] = []
    controls = {
        symbol: np.empty((len(calendar), len(FEATURE_COLUMNS)), dtype=float)
        for symbol in PAIRS
    }
    candidates = {
        symbol: np.empty(
            (len(calendar), len(FEATURE_COLUMNS) + len(CROSS_PAIR_COLUMNS)),
            dtype=float,
        )
        for symbol in PAIRS
    }
    windows = _CrossPairWindows()
    previous: datetime | None = None
    return_index = FEATURE_COLUMNS.index("mid_return_1")
    for boundary in calendar:
        if previous is not None and boundary - previous != timedelta(minutes=5):
            windows.clear()
        previous = boundary
        present = {
            symbol: pointers[symbol] < len(times[symbol])
            and times[symbol][pointers[symbol]] == boundary
            for symbol in PAIRS
        }
        if not all(present.values()):
            windows.clear()
            for symbol in PAIRS:
                if present[symbol]:
                    pointers[symbol] += 1
            continue
        indices = {symbol: pointers[symbol] for symbol in PAIRS}
        returns = {
            symbol: float(matrices[symbol][indices[symbol], return_index])
            for symbol in PAIRS
        }
        for symbol in PAIRS:
            pointers[symbol] += 1
        if not all(math.isfinite(value) for value in returns.values()):
            windows.clear()
            continue
        cross = windows.push(returns)
        output_index = len(output_times)
        output_times.append(boundary)
        for symbol in PAIRS:
            control = matrices[symbol][indices[symbol]]
            controls[symbol][output_index] = control
            candidates[symbol][output_index, : len(FEATURE_COLUMNS)] = control
            candidates[symbol][output_index, len(FEATURE_COLUMNS) :] = cross[symbol]
    if any(pointers[symbol] != len(times[symbol]) for symbol in PAIRS):
        raise ValueError("selected feature rows were not consumed by the calendar")
    return {
        symbol: FactorReadyFeatures(
            tuple(output_times),
            controls[symbol][: len(output_times)],
            candidates[symbol][: len(output_times)],
        )
        for symbol in PAIRS
    }


@dataclass(frozen=True)
class LabelRows:
    times: tuple[datetime, ...]
    rows: tuple[Mapping[str, object], ...]


def _load_labels(
    paths: Mapping[str, Path],
    start: datetime,
    end_exclusive: datetime,
    config: HistoricalScreenConfig,
) -> dict[str, LabelRows]:
    result: dict[str, LabelRows] = {}
    for symbol in PAIRS:
        table = _predicate_read(paths[symbol], "decision_time", start, end_exclusive)
        if not table.schema.equals(
            label_schema(config.horizons_minutes), check_metadata=True
        ):
            raise ValueError(f"label schema or metadata mismatch for {symbol}")
        times = _validate_selected_rows(
            table, symbol, "decision_time", start, end_exclusive, config
        )
        rows = tuple(table.to_pylist())
        for row in rows:
            for name in ("entry_time",) + tuple(
                f"exit_time_{horizon}m" for horizon in config.horizons_minutes
            ):
                value = row[name]
                if isinstance(value, datetime) and (
                    value >= config.source_end_exclusive
                    or value >= config.locked_test_start
                ):
                    raise ValueError(
                        f"selected {symbol} label crosses the source cutoff"
                    )
        result[symbol] = LabelRows(times, rows)
    return result


def _resolved_pair(
    row: Mapping[str, object], decision_time: datetime, horizon: int
) -> tuple[float, float] | None:
    long_value = row[f"long_return_{horizon}m"]
    short_value = row[f"short_return_{horizon}m"]
    entry = row["entry_time"]
    exit_time = row[f"exit_time_{horizon}m"]
    resolution_values = (long_value, short_value, exit_time)
    if all(value is None for value in resolution_values):
        return None
    if (
        isinstance(long_value, bool)
        or not isinstance(long_value, int | float)
        or isinstance(short_value, bool)
        or not isinstance(short_value, int | float)
        or not isinstance(entry, datetime)
        or not isinstance(exit_time, datetime)
    ):
        raise ValueError("executable label resolution fields are inconsistent")
    long_return = float(long_value)
    short_return = float(short_value)
    if (
        not math.isfinite(long_return)
        or not math.isfinite(short_return)
        or not decision_time <= entry <= decision_time + timedelta(minutes=5)
        or not decision_time + timedelta(minutes=horizon)
        <= exit_time
        <= decision_time + timedelta(minutes=horizon + 5)
    ):
        raise ValueError("executable label values or timestamps are invalid")
    return long_return, short_return


def _report_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RuntimeError("report metric is not numeric")
    return float(value)


def _report_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError("report count is not an integer")
    return value


def _key_digest(times: Sequence[datetime]) -> str:
    values = [value.isoformat().replace("+00:00", "Z") for value in times]
    return hashlib.sha256(_canonical_json(values)).hexdigest()


@dataclass(frozen=True)
class ScoredCell:
    times: tuple[datetime, ...]
    actions: tuple[str, ...]
    training_rows: int
    training_keys_sha256: str


def _fit_and_score(
    features: Mapping[str, FactorReadyFeatures],
    training_labels: Mapping[str, LabelRows],
    config: HistoricalScreenConfig,
) -> dict[str, dict[tuple[str, int], ScoredCell]]:
    scored: dict[str, dict[tuple[str, int], ScoredCell]] = {arm: {} for arm in ARMS}
    for symbol in PAIRS:
        factor = features[symbol]
        label_by_time = dict(
            zip(
                training_labels[symbol].times, training_labels[symbol].rows, strict=True
            )
        )
        training_candidates = [
            index
            for index, value in enumerate(factor.decision_times)
            if config.train_start <= value < config.train_end_exclusive
        ]
        screen_indices = np.asarray(
            [
                index
                for index, value in enumerate(factor.decision_times)
                if config.screen_start <= value < config.decision_end_exclusive
            ],
            dtype=np.int64,
        )
        screen_times = tuple(factor.decision_times[index] for index in screen_indices)
        for horizon in config.horizons_minutes:
            usable: list[int] = []
            targets: list[tuple[float, float]] = []
            for index in training_candidates:
                decision_time = factor.decision_times[index]
                row = label_by_time.get(decision_time)
                if row is None:
                    continue
                resolved = _resolved_pair(row, decision_time, horizon)
                if resolved is not None:
                    usable.append(index)
                    targets.append(resolved)
            if len(usable) < 2:
                raise ValueError(f"{symbol}/{horizon} has insufficient training rows")
            training_indices = np.asarray(usable, dtype=np.int64)
            training_times = tuple(factor.decision_times[index] for index in usable)
            keys_sha256 = _key_digest(training_times)
            target_matrix = np.asarray(targets, dtype=float)
            for arm, matrix in (
                (CONTROL_ARM, factor.control),
                (CANDIDATE_ARM, factor.candidate),
            ):
                model = make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True),
                    StandardScaler(),
                    Ridge(alpha=config.alpha, solver="lsqr"),
                )
                model.fit(matrix[training_indices], target_matrix)
                predictions = np.asarray(
                    model.predict(matrix[screen_indices]), dtype=float
                )
                if (
                    predictions.shape != (len(screen_indices), 2)
                    or not np.isfinite(predictions).all()
                ):
                    raise RuntimeError("ridge produced invalid screen predictions")
                actions = tuple(
                    "flat"
                    if max(float(long_value), float(short_value)) <= 0.0
                    else "long"
                    if long_value > short_value
                    else "short"
                    for long_value, short_value in predictions
                )
                scored[arm][(symbol, horizon)] = ScoredCell(
                    screen_times, actions, len(usable), keys_sha256
                )
    return scored


def _evaluate(
    scored: Mapping[str, Mapping[tuple[str, int], ScoredCell]],
    labels: Mapping[str, LabelRows],
    config: HistoricalScreenConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = expected_decision_boundaries(
        config.screen_start, config.decision_end_exclusive
    )
    arms: dict[str, Any] = {}
    cell_lookup: dict[tuple[str, str, int], dict[str, object]] = {}
    for arm in ARMS:
        diagnostics: list[dict[str, object]] = []
        for symbol in PAIRS:
            labels_by_time = dict(
                zip(labels[symbol].times, labels[symbol].rows, strict=True)
            )
            for horizon in config.horizons_minutes:
                cell = scored[arm][(symbol, horizon)]
                actions = dict(zip(cell.times, cell.actions, strict=True))
                total = 0.0
                trades = 0
                signals = 0
                flat_actions = 0
                resolved_labels = 0
                for decision_time in expected:
                    row = labels_by_time.get(decision_time)
                    resolved = (
                        None
                        if row is None
                        else _resolved_pair(row, decision_time, horizon)
                    )
                    resolved_labels += resolved is not None
                    action = actions.get(decision_time)
                    if action is None:
                        continue
                    if action == "flat":
                        flat_actions += 1
                        continue
                    signals += 1
                    if resolved is None:
                        continue
                    trades += 1
                    total += resolved[0] if action == "long" else resolved[1]
                mean_bps = total / len(expected) * 10_000.0
                row_result: dict[str, object] = {
                    "symbol": symbol,
                    "horizon_minutes": horizon,
                    "expected_opportunities": len(expected),
                    "factor_ready_and_scored": len(cell.times),
                    "unscored_opportunities": len(expected) - len(cell.times),
                    "resolved_labels": resolved_labels,
                    "absent_or_unresolved_labels": len(expected) - resolved_labels,
                    "nonflat_signals": signals,
                    "flat_actions": flat_actions,
                    "trade_count": trades,
                    "training_rows": cell.training_rows,
                    "training_keys_sha256": cell.training_keys_sha256,
                    "sum_overlapping_executable_returns": total,
                    "expected_opportunity_mean_bps": mean_bps,
                }
                diagnostics.append(row_result)
                cell_lookup[(arm, symbol, horizon)] = row_result
        horizons: list[dict[str, object]] = []
        for horizon in config.horizons_minutes:
            rows = [cell_lookup[(arm, symbol, horizon)] for symbol in PAIRS]
            total = sum(
                _report_float(row["sum_overlapping_executable_returns"])
                for row in rows
            )
            denominator = len(expected) * len(PAIRS)
            horizons.append(
                {
                    "horizon_minutes": horizon,
                    "expected_opportunities": denominator,
                    "factor_ready_and_scored": sum(
                        _report_int(row["factor_ready_and_scored"]) for row in rows
                    ),
                    "trade_count": sum(_report_int(row["trade_count"]) for row in rows),
                    "positive_symbols": len(
                        [
                            row
                            for row in rows
                            if _report_float(row["expected_opportunity_mean_bps"]) > 0.0
                        ]
                    ),
                    "pooled_expected_opportunity_mean_bps": total
                    / denominator
                    * 10_000.0,
                }
            )
        arms[arm] = {"horizons": horizons, "symbols": diagnostics}

    horizon_deltas: list[dict[str, object]] = []
    symbol_deltas: list[dict[str, object]] = []
    for horizon in config.horizons_minutes:
        control_horizon = next(
            row
            for row in arms[CONTROL_ARM]["horizons"]
            if row["horizon_minutes"] == horizon
        )
        candidate_horizon = next(
            row
            for row in arms[CANDIDATE_ARM]["horizons"]
            if row["horizon_minutes"] == horizon
        )
        horizon_deltas.append(
            {
                "horizon_minutes": horizon,
                "pooled_expected_opportunity_mean_bps_delta": _report_float(
                    candidate_horizon["pooled_expected_opportunity_mean_bps"]
                )
                - _report_float(
                    control_horizon["pooled_expected_opportunity_mean_bps"]
                ),
                "trade_count_delta": _report_int(candidate_horizon["trade_count"])
                - _report_int(control_horizon["trade_count"]),
                "positive_symbols_delta": _report_int(
                    candidate_horizon["positive_symbols"]
                )
                - _report_int(control_horizon["positive_symbols"]),
            }
        )
        for symbol in PAIRS:
            control = cell_lookup[(CONTROL_ARM, symbol, horizon)]
            candidate = cell_lookup[(CANDIDATE_ARM, symbol, horizon)]
            symbol_deltas.append(
                {
                    "symbol": symbol,
                    "horizon_minutes": horizon,
                    "expected_opportunity_mean_bps_delta": _report_float(
                        candidate["expected_opportunity_mean_bps"]
                    )
                    - _report_float(control["expected_opportunity_mean_bps"]),
                    "trade_count_delta": _report_int(candidate["trade_count"])
                    - _report_int(control["trade_count"]),
                }
            )
    return arms, {"horizons": horizon_deltas, "symbols": symbol_deltas}


def run_cross_pair_historical_screen(
    feature_paths: Mapping[str, Path],
    label_paths: Mapping[str, Path],
    config_path: Path,
    output: Path,
    code_reference: str,
) -> dict[str, Any]:
    """Verify, run, evaluate, and immutably publish the exploratory report."""
    if IMAGE_DIGEST_PATTERN.fullmatch(code_reference) is None:
        raise ValueError("code_reference must be an immutable image digest")
    config_path = config_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace historical screen report: {output}")
    config = load_historical_screen_config(config_path)
    features = _validate_path_map(feature_paths, "feature")
    labels = _validate_path_map(label_paths, "label")
    inventory = verify_historical_inputs(features, labels, config)

    feature_tables = {
        symbol: _predicate_read(
            features[symbol], "bar_end", config.train_start, config.source_end_exclusive
        )
        for symbol in PAIRS
    }
    paired = synchronize_historical_features(feature_tables, config)
    del feature_tables
    training_labels = _load_labels(
        labels, config.train_start, config.train_end_exclusive, config
    )
    scored = _fit_and_score(paired, training_labels, config)
    del paired, training_labels
    # Screen outcomes are deliberately unavailable until every arm has been scored.
    screen_labels = _load_labels(
        labels, config.screen_start, config.decision_end_exclusive, config
    )
    arms, deltas = _evaluate(scored, screen_labels, config)
    del screen_labels, scored
    if verify_historical_inputs(features, labels, config) != inventory:
        raise RuntimeError("historical inputs changed during the screen")
    if _file_sha256(config_path) != FROZEN_CONFIG_SHA256:
        raise RuntimeError("historical screen config changed during execution")
    report: dict[str, Any] = {
        "format_version": 1,
        "screen_id": config.id,
        "screen_config_sha256": FROZEN_CONFIG_SHA256,
        "declared_at": config.declared_at.isoformat().replace("+00:00", "Z"),
        "exploratory_only": True,
        "confirmatory": False,
        "promotion_authorized": False,
        "interpretation": {
            "portfolio_return_computed": False,
            "returns_overlap": True,
            "profitability_metric": "executable_return_per_expected_opportunity",
            "claim": "historical_diagnostic_only",
        },
        "provenance": {
            "code_reference": code_reference,
            "implementation_sha256": _file_sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "numpy": metadata.version("numpy"),
            "pyarrow": metadata.version("pyarrow"),
            "scikit_learn": metadata.version("scikit-learn"),
            "tzdata": metadata.version("tzdata"),
        },
        "calendar": {
            "id": CALENDAR_ID,
            "interval_minutes": 5,
            "screen_boundaries": len(
                expected_decision_boundaries(
                    config.screen_start, config.decision_end_exclusive
                )
            ),
            "missing_opportunity_return": 0.0,
            "denominator": "all_expected_calendar_boundaries_per_symbol_horizon",
        },
        "intervals": {
            "train_start": config.train_start.isoformat().replace("+00:00", "Z"),
            "train_end_exclusive": config.train_end_exclusive.isoformat().replace(
                "+00:00", "Z"
            ),
            "screen_start": config.screen_start.isoformat().replace("+00:00", "Z"),
            "decision_end_exclusive": config.decision_end_exclusive.isoformat().replace(
                "+00:00", "Z"
            ),
            "source_end_exclusive": config.source_end_exclusive.isoformat().replace(
                "+00:00", "Z"
            ),
            "locked_test_start": config.locked_test_start.isoformat().replace(
                "+00:00", "Z"
            ),
        },
        "model": {
            "type": "ridge",
            "alpha": config.alpha,
            "solver": "lsqr",
            "imputation": "training_median",
            "standardize": True,
            "action_threshold_bps": 0.0,
            "fit_scope": "per_symbol_per_horizon",
            "paired_training_keys": True,
            "control_features": list(FEATURE_COLUMNS),
            "candidate_appended_features": list(CROSS_PAIR_COLUMNS),
        },
        "inputs": inventory,
        "arms": arms,
        "candidate_minus_control": deltas,
    }
    report_id = content_id(report)
    report = {**report, "report_id": report_id}
    write_immutable_json(output, report)
    return report


def _parse_inputs(values: Sequence[str], name: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        symbol, separator, raw_path = value.partition("=")
        if not separator or not raw_path or symbol in result:
            raise ValueError(f"{name} inputs must be unique SYMBOL=PATH values")
        result[symbol] = Path(raw_path)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen Campaign 2 cross-pair historical screen."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--feature", action="append", required=True, metavar="SYMBOL=PATH"
    )
    parser.add_argument(
        "--label", action="append", required=True, metavar="SYMBOL=PATH"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-reference", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the standalone exploratory runner."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        report = run_cross_pair_historical_screen(
            _parse_inputs(arguments.feature, "feature"),
            _parse_inputs(arguments.label, "label"),
            arguments.config,
            arguments.output,
            arguments.code_reference,
        )
        print(f"historical screen report {report['report_id']}: {arguments.output}")
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

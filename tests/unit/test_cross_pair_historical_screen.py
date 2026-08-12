import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import demofml.reporting.cross_pair_historical as historical
from demofml.features.causal import FEATURE_SCHEMA
from demofml.features.cross_pair import (
    CROSS_PAIR_COLUMNS,
    PAIRS,
    solve_cross_pair_factor,
)
from demofml.labels.executable import label_schema
from demofml.models.baseline import FEATURE_COLUMNS
from demofml.reporting.cross_pair_historical import (
    ARMS,
    LabelRows,
    ScoredCell,
    load_historical_screen_config,
    run_cross_pair_historical_screen,
    synchronize_historical_features,
    verify_historical_inputs,
)

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = (
    PROJECT_ROOT
    / "configs/experiments/campaign2-cross-pair-historical-screen-2020-v1.toml"
)
FROZEN_DOC = (
    PROJECT_ROOT
    / "docs/research/campaign-2-v2-exploratory-reclassification-2026-08-05.md"
)
TRAIN_TIMES = (
    datetime(2018, 1, 1, 0, 0, tzinfo=UTC),
    datetime(2018, 1, 1, 0, 5, tzinfo=UTC),
)
SCREEN_TIMES = (
    datetime(2020, 1, 1, 0, 0, tzinfo=UTC),
    datetime(2020, 1, 1, 0, 5, tzinfo=UTC),
)
CODE_REFERENCE = "sha256:" + "a" * 64


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feature_table(symbol: str, times: tuple[datetime, ...]) -> pa.Table:
    rows: list[dict[str, object]] = []
    symbol_index = PAIRS.index(symbol) + 1
    for index, decision_time in enumerate(times):
        row: dict[str, object] = {
            "symbol": symbol,
            "bar_end": decision_time,
        }
        row.update(
            {
                name: (index + 1) * symbol_index * 1e-5 + feature_index * 1e-7
                for feature_index, name in enumerate(FEATURE_COLUMNS)
            }
        )
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=FEATURE_SCHEMA)


def _label_table(symbol: str, times: tuple[datetime, ...]) -> pa.Table:
    rows: list[dict[str, object]] = []
    symbol_index = PAIRS.index(symbol) + 1
    for index, decision_time in enumerate(times):
        row: dict[str, object] = {
            "symbol": symbol,
            "decision_time": decision_time,
            "entry_time": decision_time,
            "entry_bid": 1.0,
            "entry_ask": 1.0001,
        }
        for horizon in (15, 30, 60):
            signal = (index * 2 - 1) * symbol_index / 100_000.0
            row[f"exit_time_{horizon}m"] = decision_time + timedelta(minutes=horizon)
            row[f"long_return_{horizon}m"] = signal
            row[f"short_return_{horizon}m"] = -signal
            row[f"action_{horizon}m"] = "long" if signal > 0.0 else "short"
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=label_schema((15, 30, 60)))


def _write_inputs(
    root: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    feature_paths: dict[str, Path] = {}
    label_paths: dict[str, Path] = {}
    all_times = (*TRAIN_TIMES, SCREEN_TIMES[0])
    for symbol in PAIRS:
        feature_path = root / f"{symbol}-features.parquet"
        label_path = root / f"{symbol}-labels.parquet"
        pq.write_table(_feature_table(symbol, all_times), feature_path)
        pq.write_table(_label_table(symbol, all_times), label_path)
        feature_paths[symbol] = feature_path
        label_paths[symbol] = label_path

    return feature_paths, label_paths


def _verified_inventory(
    _features: dict[str, Path],
    _labels: dict[str, Path],
    config: historical.HistoricalScreenConfig,
) -> dict[str, dict[str, str]]:
    return {
        "features": dict(config.feature_sha256),
        "labels": dict(config.label_sha256),
    }


def _small_calendar(start: datetime, end: datetime) -> tuple[datetime, ...]:
    if start.year == 2018 and end.year == 2021:
        return (*TRAIN_TIMES, SCREEN_TIMES[0])
    if start.year == 2020:
        return SCREEN_TIMES
    raise AssertionError((start, end))


def test_frozen_config_is_strict_and_frozen_files_are_unchanged(tmp_path: Path) -> None:
    config = load_historical_screen_config(CONFIG)

    assert config.symbols == PAIRS
    assert config.horizons_minutes == (15, 30, 60)
    assert (
        _sha256(CONFIG)
        == "1a2bdf972bbafc350ee4b4038580ebaf44c6148a5e55c9121fa9ad810d5d4d2b"
    )
    assert (
        _sha256(FROZEN_DOC)
        == "e1925cfd78363f65cf124b8d39fae5f54e6bf8d48d06258fcc2df70ebdce00af"
    )

    extra = tmp_path / "extra.toml"
    extra.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "[authorization]", "unknown = 1\n\n[authorization]"
        )
    )
    with pytest.raises(ValueError, match="frozen contract"):
        load_historical_screen_config(extra)

    unsafe = tmp_path / "unsafe.toml"
    unsafe.write_text(
        CONFIG.read_text(encoding="utf-8").replace(
            "locked_test_access = false", "locked_test_access = true"
        )
    )
    with pytest.raises(ValueError, match="frozen contract"):
        load_historical_screen_config(unsafe)

    with pytest.raises(ValueError, match="input inventory"):
        replace(config, expected_rows={**config.expected_rows, "AUDUSD": 1})


def test_input_verification_checks_hash_rows_schema_and_metadata(
    tmp_path: Path,
) -> None:
    feature_paths, label_paths = _write_inputs(tmp_path)
    config = load_historical_screen_config(CONFIG)

    with pytest.raises(ValueError, match="hash mismatch"):
        verify_historical_inputs(feature_paths, label_paths, config)


def test_historical_factor_matches_closed_form_and_resets_on_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_historical_screen_config(CONFIG)
    start = TRAIN_TIMES[0]
    boundaries = tuple(start + timedelta(minutes=5 * index) for index in range(7))
    monkeypatch.setattr(
        historical,
        "expected_decision_boundaries",
        lambda _start, _end: boundaries,
    )
    tables: dict[str, pa.Table] = {}
    for symbol in PAIRS:
        symbol_times = tuple(
            boundary
            for index, boundary in enumerate(boundaries)
            if not (index == 4 and symbol == "USDJPY")
        )
        table = _feature_table(symbol, symbol_times)
        if symbol == "USDJPY":
            rows = table.to_pylist()
            # causal-v1 resets this return after the symbol-specific missing bar.
            rows[4]["mid_return_1"] = None
            table = pa.Table.from_pylist(rows, schema=FEATURE_SCHEMA)
        tables[symbol] = table

    paired = synchronize_historical_features(tables, config)

    assert all(
        value.decision_times == paired[PAIRS[0]].decision_times
        for value in paired.values()
    )
    assert paired["AUDUSD"].decision_times == (*boundaries[:4], boundaries[6])
    first_returns = {
        symbol: float(tables[symbol].column("mid_return_1")[0].as_py())
        for symbol in PAIRS
    }
    solution = solve_cross_pair_factor(first_returns)
    first_cross = paired["AUDUSD"].candidate[0, -len(CROSS_PAIR_COLUMNS) :]
    assert first_cross[0] == pytest.approx(solution.strength("AUD"))
    assert first_cross[6] == pytest.approx(solution.residual("AUDUSD"))
    assert first_cross[9] == pytest.approx(solution.residual_dispersion)
    assert np.isnan(first_cross[1])
    assert np.isfinite(paired["AUDUSD"].candidate[2, -11])
    assert np.isnan(paired["AUDUSD"].candidate[-1, -11])


def test_historical_source_rows_outside_decision_calendar_are_ignored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_historical_screen_config(CONFIG)
    expected = TRAIN_TIMES
    source_only = datetime(2018, 1, 6, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(
        historical,
        "expected_decision_boundaries",
        lambda _start, _end: expected,
    )
    monkeypatch.setattr(
        historical,
        "is_expected_decision_boundary",
        lambda value: value in expected,
    )
    tables = {
        symbol: _feature_table(symbol, (*expected, source_only)) for symbol in PAIRS
    }

    paired = synchronize_historical_features(tables, config)

    assert paired["AUDUSD"].decision_times == expected
    assert paired["AUDUSD"].control.shape[0] == len(expected)


def test_runner_scores_before_screen_labels_and_uses_identical_paired_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_paths, label_paths = _write_inputs(tmp_path)
    monkeypatch.setattr(historical, "expected_decision_boundaries", _small_calendar)
    monkeypatch.setattr(historical, "verify_historical_inputs", _verified_inventory)
    events: list[str] = []
    original_fit = historical._fit_and_score
    original_load = historical._load_labels

    def tracked_fit(*args: Any, **kwargs: Any) -> Any:
        events.append("score")
        return original_fit(*args, **kwargs)

    def tracked_load(
        paths: dict[str, Path],
        start: datetime,
        end: datetime,
        config: historical.HistoricalScreenConfig,
    ) -> dict[str, LabelRows]:
        events.append("screen-labels" if start.year == 2020 else "training-labels")
        return original_load(paths, start, end, config)

    monkeypatch.setattr(historical, "_fit_and_score", tracked_fit)
    monkeypatch.setattr(historical, "_load_labels", tracked_load)
    report = run_cross_pair_historical_screen(
        feature_paths,
        label_paths,
        CONFIG,
        tmp_path / "report.json",
        CODE_REFERENCE,
    )

    assert events == ["training-labels", "score", "screen-labels"]
    assert report["exploratory_only"] is True
    assert report["confirmatory"] is False
    assert report["promotion_authorized"] is False
    control = report["arms"]["control"]["symbols"]
    candidate = report["arms"]["candidate"]["symbols"]
    for control_row, candidate_row in zip(control, candidate, strict=True):
        assert control_row["training_rows"] == candidate_row["training_rows"]
        assert (
            control_row["training_keys_sha256"] == candidate_row["training_keys_sha256"]
        )


def test_runner_is_deterministic_missing_flat_and_immutable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_paths, label_paths = _write_inputs(tmp_path)
    monkeypatch.setattr(historical, "expected_decision_boundaries", _small_calendar)
    monkeypatch.setattr(historical, "verify_historical_inputs", _verified_inventory)
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    first = run_cross_pair_historical_screen(
        feature_paths, label_paths, CONFIG, first_path, CODE_REFERENCE
    )
    second = run_cross_pair_historical_screen(
        feature_paths, label_paths, CONFIG, second_path, CODE_REFERENCE
    )

    assert first == second
    assert first_path.read_bytes() == second_path.read_bytes()
    assert first["report_id"].startswith("sha256-")
    horizon = first["arms"]["control"]["horizons"][0]
    assert horizon["expected_opportunities"] == len(PAIRS) * len(SCREEN_TIMES)
    cell = first["arms"]["control"]["symbols"][0]
    assert cell["expected_opportunities"] == 2
    assert cell["factor_ready_and_scored"] == 1
    assert cell["unscored_opportunities"] == 1
    expected_mean = cell["sum_overlapping_executable_returns"] / 2 * 10_000.0
    assert cell["expected_opportunity_mean_bps"] == pytest.approx(expected_mean)
    with pytest.raises(RuntimeError, match="Refusing to replace"):
        run_cross_pair_historical_screen(
            feature_paths, label_paths, CONFIG, first_path, CODE_REFERENCE
        )


def test_selected_rows_reject_source_and_locked_cutoffs() -> None:
    config = load_historical_screen_config(CONFIG)
    post_cutoff = _feature_table("AUDUSD", (datetime(2021, 1, 1, 0, 0, tzinfo=UTC),))
    with pytest.raises(ValueError, match="source cutoff"):
        historical._validate_selected_rows(
            post_cutoff,
            "AUDUSD",
            "bar_end",
            config.train_start,
            config.locked_test_start + timedelta(minutes=5),
            config,
        )

    locked = _feature_table("AUDUSD", (config.locked_test_start,))
    with pytest.raises(ValueError, match="source cutoff"):
        historical._validate_selected_rows(
            locked,
            "AUDUSD",
            "bar_end",
            config.train_start,
            config.locked_test_start + timedelta(minutes=5),
            config,
        )


def test_missing_unresolved_and_unscored_are_zero_denominator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_historical_screen_config(CONFIG)
    monkeypatch.setattr(
        historical,
        "expected_decision_boundaries",
        lambda _start, _end: SCREEN_TIMES,
    )
    scored: dict[str, dict[tuple[str, int], ScoredCell]] = {arm: {} for arm in ARMS}
    labels: dict[str, LabelRows] = {}
    unresolved_rows = _label_table("AUDUSD", (SCREEN_TIMES[0],)).to_pylist()
    for horizon in (15, 30, 60):
        unresolved_rows[0][f"exit_time_{horizon}m"] = None
        unresolved_rows[0][f"long_return_{horizon}m"] = None
        unresolved_rows[0][f"short_return_{horizon}m"] = None
        unresolved_rows[0][f"action_{horizon}m"] = None
    for symbol in PAIRS:
        labels[symbol] = (
            LabelRows((SCREEN_TIMES[0],), tuple(unresolved_rows))
            if symbol == "AUDUSD"
            else LabelRows((), ())
        )
        for horizon in (15, 30, 60):
            for arm in ARMS:
                scored[arm][(symbol, horizon)] = ScoredCell(
                    (SCREEN_TIMES[0],), ("long",), 2, "0" * 64
                )

    arms, _deltas = historical._evaluate(scored, labels, config)

    for arm in ARMS:
        assert arms[arm]["horizons"][0]["pooled_expected_opportunity_mean_bps"] == 0.0
        assert arms[arm]["horizons"][0]["trade_count"] == 0
        assert arms[arm]["symbols"][0]["expected_opportunities"] == 2
        assert arms[arm]["symbols"][0]["absent_or_unresolved_labels"] == 2


def test_report_id_hashes_the_deterministic_report_core(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    feature_paths, label_paths = _write_inputs(tmp_path)
    monkeypatch.setattr(historical, "expected_decision_boundaries", _small_calendar)
    monkeypatch.setattr(historical, "verify_historical_inputs", _verified_inventory)
    report = run_cross_pair_historical_screen(
        feature_paths,
        label_paths,
        CONFIG,
        tmp_path / "report.json",
        CODE_REFERENCE,
    )
    core = {key: value for key, value in report.items() if key != "report_id"}
    expected = hashlib.sha256(historical._canonical_json(core)).hexdigest()

    assert report["report_id"] == f"sha256-{expected}"
    assert json.loads((tmp_path / "report.json").read_text()) == report

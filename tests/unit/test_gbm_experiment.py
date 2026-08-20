"""Campaign 3 Stage A: gradient boosting over causal-v2 features.

These cover the two things that make the line trustworthy rather than merely
runnable: candidate selection never reads a validation fold, and the published
metrics.json reproduces exactly from the raw predictions — the invariant the
development acceptance gate checks with strict equality.
"""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

from demofml.evaluation.signals import evaluate_predictions
from demofml.features.causal_v2 import FEATURE_V2_COLUMNS, FEATURE_V2_SCHEMA
from demofml.labels.executable import label_schema
from demofml.models.baseline import align_research_tables
from demofml.models.gbm import (
    GBM_MODEL_SET_ID,
    GBM_MODEL_SET_V2_ID,
    GBM_PREDICTION_SET_ID,
    GBM_PREDICTION_SET_V2_ID,
    GateCandidate,
    GBMConfig,
    _conviction_threshold,
    load_gbm_config,
    run_walk_forward_gbm,
    select_candidate,
    select_candidate_and_gate,
)
from demofml.models.gbm_build import main as gbm_main
from demofml.models.gbm_build import run_gbm_experiment
from demofml.validation.splits import ValidationPlan, load_validation_plan

PROJECT_ROOT = Path(__file__).parents[2]
CONFIGS = PROJECT_ROOT / "configs/experiments"
MODEL_CONFIG = CONFIGS / "campaign-3-lightgbm-causal-v2-model-v1.toml"
VALIDATION_CONFIG = CONFIGS / "campaign-3-walk-forward-v1.toml"

_TRAINING_ROWS = 160
_VALIDATION_ROWS = 24
_TRAINING_START = datetime(2021, 12, 20, tzinfo=UTC)
_VALIDATION_START = datetime(2022, 1, 1, tzinfo=UTC)


def _plan() -> ValidationPlan:
    """One month of validation, so the fixture stays small but real."""
    return replace(
        load_validation_plan(VALIDATION_CONFIG),
        train_start=datetime(2021, 12, 1, tzinfo=UTC),
        first_validation_start=_VALIDATION_START,
        development_end_exclusive=datetime(2022, 2, 1, tzinfo=UTC),
        locked_test_start=datetime(2022, 2, 1, tzinfo=UTC),
        locked_test_end_exclusive=datetime(2022, 3, 1, tzinfo=UTC),
    )


def _config() -> GBMConfig:
    """The sealed contract, shrunk to fixture size without changing its shape."""
    sealed = load_gbm_config(MODEL_CONFIG)
    return replace(
        sealed,
        minimum_training_rows=10,
        minimum_inner_training_rows=2,
        candidates=tuple(
            replace(candidate, n_estimators=8, min_child_samples=2)
            for candidate in sealed.candidates[:2]
        ),
    )


def _tables(validation_signal: float = 1.0) -> tuple[pa.Table, pa.Table]:
    """Features and labels whose only signal is a linearly drifting level."""
    times = [
        _TRAINING_START + timedelta(minutes=5 * index)
        for index in range(_TRAINING_ROWS)
    ]
    times.extend(
        _VALIDATION_START + timedelta(minutes=5 * index)
        for index in range(_VALIDATION_ROWS)
    )
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for index, decision_time in enumerate(times):
        in_validation = decision_time >= _VALIDATION_START
        signal = float(index % 20 - 10)
        if in_validation:
            signal *= validation_signal
        feature_row: dict[str, object] = {"symbol": "EURUSD", "bar_end": decision_time}
        for position, name in enumerate(FEATURE_V2_COLUMNS):
            feature_row[name] = signal + position / 100.0
        feature_rows.append(feature_row)

        long_return = signal / 10_000.0
        label_row: dict[str, object] = {
            "symbol": "EURUSD",
            "decision_time": decision_time,
            "entry_time": decision_time,
            "entry_bid": 1.0,
            "entry_ask": 1.0001,
        }
        for horizon in (15, 30, 60):
            label_row[f"exit_time_{horizon}m"] = decision_time + timedelta(
                minutes=horizon
            )
            label_row[f"long_return_{horizon}m"] = long_return
            label_row[f"short_return_{horizon}m"] = -long_return
            label_row[f"action_{horizon}m"] = "long" if long_return > 0 else "short"
        label_rows.append(label_row)
    return (
        pa.Table.from_pylist(feature_rows, schema=FEATURE_V2_SCHEMA),
        pa.Table.from_pylist(
            label_rows,
            schema=label_schema((15, 30, 60), 0.0, label_set="executable-v2"),
        ),
    )


def test_walk_forward_publishes_provenance_and_a_sealed_candidate() -> None:
    features, labels = _tables()
    config = _config()

    predictions = run_walk_forward_gbm(features, labels, _plan(), config)

    metadata = predictions.schema.metadata
    assert metadata[b"demofml.prediction_set"] == GBM_PREDICTION_SET_ID.encode()
    assert metadata[b"demofml.model_set"] == GBM_MODEL_SET_ID.encode()
    assert metadata[b"demofml.selection_policy"] == b"inner-purged-cv-first-fold-v1"
    assert predictions.num_rows == _VALIDATION_ROWS * 3
    sealed_ids = {candidate.id for candidate in config.candidates}
    assert set(predictions.column("selected_candidate").to_pylist()) <= sealed_ids
    assert set(predictions.column("action").to_pylist()) <= {"long", "short", "flat"}
    assert set(predictions.column("fold_id").to_pylist()) == {"wf-2022-01"}
    rows = predictions.to_pylist()
    for row in rows:
        expected = (
            0.0
            if row["action"] == "flat"
            else row["realized_return"]
        )
        assert row["realized_return"] == expected
        assert row["decision_time"] <= row["entry_time"] < row["exit_time"]


def test_candidate_selection_cannot_see_the_validation_fold() -> None:
    """Rule 2 of the protocol, as an executable check rather than a promise."""
    plan = _plan()
    config = _config()
    fold = plan.folds()[0]

    baseline_features, baseline_labels = _tables()
    inverted_features, inverted_labels = _tables(validation_signal=-25.0)

    chosen = select_candidate(
        align_research_tables(baseline_features, baseline_labels, plan, config),
        fold,
        60,
        config,
    )
    chosen_after_inversion = select_candidate(
        align_research_tables(inverted_features, inverted_labels, plan, config),
        fold,
        60,
        config,
    )

    assert chosen == chosen_after_inversion

    # The perturbation must actually be visible downstream, or the invariance
    # above would be vacuous.
    unchanged = run_walk_forward_gbm(baseline_features, baseline_labels, plan, config)
    inverted = run_walk_forward_gbm(inverted_features, inverted_labels, plan, config)
    assert unchanged.column("predicted_long_return").to_pylist() != (
        inverted.column("predicted_long_return").to_pylist()
    )


def _fixture_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Write fixture-sized copies of the sealed contracts plus their inputs."""
    features, labels = _tables()
    model_config = tmp_path / "model.toml"
    model_config.write_text(
        MODEL_CONFIG.read_text(encoding="utf-8")
        .replace("minimum_training_rows = 20", "minimum_training_rows = 10")
        .replace(
            "minimum_inner_training_rows = 5000",
            "minimum_inner_training_rows = 2",
        )
        .replace("n_estimators = 400", "n_estimators = 8")
        .replace("n_estimators = 300", "n_estimators = 8")
        .replace("n_estimators = 200", "n_estimators = 8")
        .replace("min_child_samples = 1000", "min_child_samples = 2")
        .replace("min_child_samples = 500", "min_child_samples = 2")
        .replace("min_child_samples = 200", "min_child_samples = 2"),
        encoding="utf-8",
    )
    validation_config = tmp_path / "validation.toml"
    validation_config.write_text(
        VALIDATION_CONFIG.read_text(encoding="utf-8")
        .replace(
            'train_start = "2018-01-01T00:00:00Z"',
            'train_start = "2021-12-01T00:00:00Z"',
        )
        .replace(
            'development_end_exclusive = "2025-01-01T00:00:00Z"',
            'development_end_exclusive = "2022-02-01T00:00:00Z"',
        )
        .replace(
            'locked_test_start = "2025-01-01T00:00:00Z"',
            'locked_test_start = "2022-02-01T00:00:00Z"',
        )
        .replace(
            'locked_test_end_exclusive = "2026-03-11T00:00:00Z"',
            'locked_test_end_exclusive = "2022-03-01T00:00:00Z"',
        ),
        encoding="utf-8",
    )
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    pq.write_table(features, features_path)
    pq.write_table(labels, labels_path)
    return features_path, labels_path, validation_config, model_config


def test_experiment_artifacts_reproduce_their_own_metrics(tmp_path: Path) -> None:
    features_path, labels_path, validation_config, model_config = _fixture_inputs(
        tmp_path
    )
    output = tmp_path / "model"

    result = run_gbm_experiment(
        features_path, labels_path, validation_config, model_config, output
    )

    assert result.symbol == "EURUSD"
    assert result.fold_count == 1
    stored_metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    recomputed = evaluate_predictions(pq.read_table(output / "predictions.parquet"))
    # The acceptance gate compares these with strict equality.
    assert stored_metrics == recomputed
    diagnostics = json.loads(
        (output / "feature_null_diagnostics.json").read_text(encoding="utf-8")
    )
    assert set(diagnostics) == {"wf-2022-01"}
    selected = json.loads(
        (output / "selected_candidates.json").read_text(encoding="utf-8")
    )
    assert selected["model_set"] == GBM_MODEL_SET_ID
    assert selected["selected_candidates"]

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        run_gbm_experiment(
            features_path, labels_path, validation_config, model_config, output
        )


def test_config_rejects_contracts_it_was_not_written_for() -> None:
    sealed = load_gbm_config(MODEL_CONFIG)

    for override, expected in (
        ({"id": "gbm-lightgbm-v9"}, "model id is not supported"),
        ({"feature_set": "causal-v1"}, "data contracts are incompatible"),
        ({"validation_set": "purged-walk-forward-v1"}, "campaign-3-walk-forward-v1"),
        ({"model_type": "ridge"}, "must be lightgbm"),
        ({"inner_folds": 2}, "between three and five"),
        ({"inner_purge_minutes": 60}, "cover the full label window"),
        ({"num_threads": 0}, "deterministic fitting"),
        ({"deterministic": False}, "deterministic fitting"),
        ({"locked_test_policy": "allowed"}, "must remain forbidden"),
        ({"features": ("mid_return_1",)}, "do not match causal-v2"),
        ({"candidates": sealed.candidates[:1]}, "at least two candidates"),
        (
            {"candidates": (sealed.candidates[0], sealed.candidates[0])},
            "candidate ids must be unique",
        ),
    ):
        with pytest.raises(ValueError, match=expected):
            replace(sealed, **override)


def test_candidate_rejects_an_incoherent_search_point() -> None:
    sealed = load_gbm_config(MODEL_CONFIG).candidates[0]

    for override, expected in (
        ({"id": ""}, "candidate id cannot be empty"),
        ({"num_leaves": 1}, "tree size must be positive"),
        ({"num_leaves": 1024}, "exceeds its depth limit"),
        ({"learning_rate": 0.0}, "learning_rate must be in"),
        ({"n_estimators": 0}, "estimator counts must be positive"),
        ({"colsample_bytree": 1.5}, "sampling fractions must be in"),
        ({"subsample_freq": 0}, "requires a positive subsample_freq"),
        ({"reg_lambda": -1.0}, "finite and non-negative"),
    ):
        with pytest.raises(ValueError, match=expected):
            replace(sealed, **override)


def test_load_gbm_config_rejects_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="not a file"):
        load_gbm_config(tmp_path / "absent.toml")

    broken = tmp_path / "broken.toml"
    broken.write_text('id = "gbm-lightgbm-v1"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid gbm config field"):
        load_gbm_config(broken)


def test_config_rejects_the_remaining_contract_violations() -> None:
    sealed = load_gbm_config(MODEL_CONFIG)

    for override, expected in (
        ({"horizons_minutes": (60, 15)}, "unique and increasing"),
        ({"training_scope": "pooled"}, "must be per_symbol"),
        ({"objective": "huber"}, "must be lightgbm"),
        ({"selection_policy": "grid-search-v1"}, "selection policy does not match"),
        ({"selection_metric": "sharpe"}, "selection metric is not supported"),
        ({"minimum_training_rows": 1}, "training row minima"),
        ({"minimum_inner_training_rows": 1}, "training row minima"),
        ({"action_threshold_bps": float("nan")}, "must be finite"),
        ({"random_seed": -1}, "cannot be negative"),
    ):
        with pytest.raises(ValueError, match=expected):
            replace(sealed, **override)

    with pytest.raises(KeyError, match="unknown candidate"):
        sealed.candidate("does-not-exist")


def test_walk_forward_refuses_folds_it_cannot_train_or_score() -> None:
    features, labels = _tables()
    plan = _plan()

    starved = replace(_config(), minimum_training_rows=10_000)
    with pytest.raises(ValueError, match="insufficient training rows"):
        run_walk_forward_gbm(features, labels, plan, starved)

    unresolved_rows = labels.to_pylist()
    for row in unresolved_rows:
        if row["decision_time"] >= _VALIDATION_START:
            row["long_return_60m"] = None
    unresolved = pa.Table.from_pylist(unresolved_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="no resolved validation labels"):
        run_walk_forward_gbm(features, unresolved, plan, _config())


def test_inner_selection_needs_enough_training_rows() -> None:
    features, labels = _tables()
    plan = _plan()
    config = replace(_config(), minimum_inner_training_rows=10_000)
    data = align_research_tables(features, labels, plan, config)

    with pytest.raises(ValueError, match="no usable inner training rows"):
        select_candidate(data, plan.folds()[0], 60, config)


def test_gbm_command_reports_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_error:
        gbm_main(
            [
                "--features",
                str(tmp_path / "absent.parquet"),
                "--labels",
                str(tmp_path / "absent.parquet"),
                "--validation-config",
                str(VALIDATION_CONFIG),
                "--model-config",
                str(MODEL_CONFIG),
                "--output",
                str(tmp_path / "out"),
            ]
        )
    assert exit_error.value.code == 1
    assert "Feature input is not a file" in capsys.readouterr().err


def test_gbm_command_publishes_and_reports_a_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    features_path, labels_path, validation_config, model_config = _fixture_inputs(
        tmp_path
    )
    output = tmp_path / "cli-model"

    gbm_main(
        [
            "--features",
            str(features_path),
            "--labels",
            str(labels_path),
            "--validation-config",
            str(validation_config),
            "--model-config",
            str(model_config),
            "--output",
            str(output),
        ]
    )

    assert "predictions across 1 folds for EURUSD" in capsys.readouterr().out
    assert (output / "predictions.parquet").is_file()

    with pytest.raises(RuntimeError, match="Label input is not a file"):
        run_gbm_experiment(
            features_path,
            tmp_path / "absent.parquet",
            validation_config,
            model_config,
            tmp_path / "unused",
        )


GATED_MODEL_CONFIG = CONFIGS / "campaign-3-lightgbm-gated-causal-v2-model-v1.toml"


def _gated_config() -> GBMConfig:
    """The sealed Stage B contract, shrunk to fixture size."""
    sealed = load_gbm_config(GATED_MODEL_CONFIG)
    return replace(
        sealed,
        minimum_training_rows=10,
        minimum_inner_training_rows=2,
        candidates=tuple(
            replace(candidate, n_estimators=8, min_child_samples=2)
            for candidate in sealed.candidates[:2]
        ),
    )


def test_gated_walk_forward_publishes_its_gate_provenance() -> None:
    features, labels = _tables()
    config = _gated_config()

    predictions = run_walk_forward_gbm(features, labels, _plan(), config)

    metadata = predictions.schema.metadata
    assert metadata[b"demofml.prediction_set"] == GBM_PREDICTION_SET_V2_ID.encode()
    assert metadata[b"demofml.model_set"] == GBM_MODEL_SET_V2_ID.encode()
    assert metadata[b"demofml.gate_calibration"] == b"per_fold_training_quantile"
    sealed_gates = {gate.id for gate in config.gates}
    assert set(predictions.column("selected_gate").to_pylist()) <= sealed_gates
    assert set(predictions.column("action").to_pylist()) <= {"long", "short", "flat"}
    # Stage A's schema must stay free of Stage B's columns.
    assert "selected_gate" not in run_walk_forward_gbm(
        features, labels, _plan(), _config()
    ).column_names


def test_the_ungated_gate_reproduces_stage_a_exactly() -> None:
    """`all-signals` is in the search space so gating has to earn its place.

    If it did not reproduce the Stage A rule, the search would be comparing the
    gates against a straw man rather than against the rule they must beat.
    """
    features, labels = _tables()
    plan = _plan()
    stage_a = _config()
    only_ungated = replace(
        _gated_config(),
        candidates=stage_a.candidates,
        gates=tuple(g for g in _gated_config().gates if g.id == "all-signals")
        + (GateCandidate("unreachable", 0.25, -99.0),),
    )

    gated = run_walk_forward_gbm(features, labels, plan, only_ungated)
    ungated = run_walk_forward_gbm(features, labels, plan, stage_a)

    assert gated.column("selected_gate").to_pylist()[0] == "all-signals"
    assert gated.column("action").to_pylist() == ungated.column("action").to_pylist()
    assert (
        gated.column("realized_return").to_pylist()
        == ungated.column("realized_return").to_pylist()
    )


def test_a_tighter_gate_never_trades_more_than_a_looser_one() -> None:
    """Monotonicity: conviction and spread gates can only remove trades."""
    features, labels = _tables()
    plan = _plan()
    base = _gated_config()

    def traded(gate: GateCandidate) -> int:
        config = replace(base, gates=(gate, GateCandidate("decoy", 0.25, -99.0)))
        table = run_walk_forward_gbm(features, labels, plan, config)
        return sum(1 for a in table.column("action").to_pylist() if a != "flat")

    loose = traded(GateCandidate("loose", 1.0, None))
    tight = traded(GateCandidate("tight", 0.25, None))
    spread_gated = traded(GateCandidate("spread", 1.0, 0.0))

    assert tight <= loose
    assert spread_gated <= loose


def test_gate_selection_cannot_see_the_validation_fold() -> None:
    plan = _plan()
    config = _gated_config()
    fold = plan.folds()[0]

    chosen = select_candidate_and_gate(
        align_research_tables(*_tables(), plan, config), fold, 60, config
    )
    inverted = select_candidate_and_gate(
        align_research_tables(*_tables(validation_signal=-25.0), plan, config),
        fold,
        60,
        config,
    )

    assert chosen == inverted


def test_conviction_threshold_is_computed_from_training_rows_only() -> None:
    scores = np.array([-1.0, 0.1, 0.2, 0.3, 0.4], dtype=float)

    keep_all = _conviction_threshold(scores, GateCandidate("a", 1.0, None), 0.0)
    keep_half = _conviction_threshold(scores, GateCandidate("b", 0.5, None), 0.0)

    assert keep_all == 0.1
    assert keep_half > keep_all
    # No positive score at all means the gate refuses to trade, never crashes.
    negative = np.array([-1.0, -2.0], dtype=float)
    assert _conviction_threshold(
        negative, GateCandidate("c", 0.5, None), 0.0
    ) == float("inf")


def test_gated_config_rejects_contract_violations() -> None:
    sealed = load_gbm_config(GATED_MODEL_CONFIG)
    stage_a = load_gbm_config(MODEL_CONFIG)

    with pytest.raises(ValueError, match="cannot define action gates"):
        replace(stage_a, gates=sealed.gates)
    with pytest.raises(ValueError, match="at least two gates"):
        replace(sealed, gates=sealed.gates[:1])
    with pytest.raises(ValueError, match="gate ids must be unique"):
        replace(sealed, gates=(sealed.gates[0], sealed.gates[0]))
    with pytest.raises(ValueError, match="selection policy does not match"):
        replace(sealed, selection_policy="inner-purged-cv-first-fold-v1")
    with pytest.raises(KeyError, match="unknown gate"):
        sealed.gate("does-not-exist")

    for override, expected in (
        ({"id": ""}, "gate id cannot be empty"),
        ({"conviction_quantile": 0.0}, "conviction_quantile must be in"),
        ({"conviction_quantile": 1.5}, "conviction_quantile must be in"),
        ({"spread_zscore_maximum": float("nan")}, "must be finite when set"),
    ):
        with pytest.raises(ValueError, match=expected):
            replace(sealed.gates[0], **override)

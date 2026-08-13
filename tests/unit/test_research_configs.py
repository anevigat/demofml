import tomllib
from pathlib import Path

from demofml.evaluation.portfolio import (
    PORTFOLIO_HORIZONS,
    PORTFOLIO_SET_ID,
    PORTFOLIO_SET_V2_ID,
    PORTFOLIO_SET_V3_ID,
    PORTFOLIO_SYMBOLS,
    load_portfolio_config,
)
from demofml.features.causal import FEATURE_SET_ID
from demofml.features.causal_v2 import FEATURE_SET_V2_ID, FEATURE_V2_COLUMNS
from demofml.labels.executable import (
    BAR_INTERVAL_MINUTES,
    DEFAULT_HORIZONS_MINUTES,
    LABEL_SET_ID,
    LABEL_SET_V2_ID,
    MAX_QUOTE_LATENCY_MINUTES,
)
from demofml.models.baseline import (
    FEATURE_COLUMNS,
    MODEL_SET_ID,
    MODEL_SET_V2_ID,
    MODEL_SET_V3_ID,
    PREDICTION_SET_V3_ID,
    PREDICTION_SET_V4_ID,
    load_baseline_config,
)
from demofml.orchestration.locked import (
    LOCKED_TEST_SET_ID,
    ONE_SHOT_POLICY,
    load_locked_test_config,
)
from demofml.reporting.acceptance import (
    ACCEPTANCE_SET_ID,
    ACCEPTANCE_SET_V2_ID,
    load_acceptance_config,
)
from demofml.validation.splits import (
    INTERVAL_SEMANTICS,
    SCREEN_VALIDATION_SET_ID,
    VALIDATION_SET_ID,
    VALIDATION_STRATEGY,
    load_validation_plan,
)

PROJECT_ROOT = Path(__file__).parents[2]


def test_feature_config_matches_implementation() -> None:
    path = PROJECT_ROOT / "configs/features/causal-v1.toml"
    with path.open("rb") as source:
        config = tomllib.load(source)

    assert config["id"] == FEATURE_SET_ID
    assert config["bar_interval_minutes"] == BAR_INTERVAL_MINUTES
    assert config["gap_policy"] == "reset_trailing_state"

    with (PROJECT_ROOT / "configs/features/causal-v2.toml").open("rb") as source:
        v2 = tomllib.load(source)
    assert v2["id"] == FEATURE_SET_V2_ID
    assert tuple(v2["features"]) == FEATURE_V2_COLUMNS
    assert v2["microstructure_windows_minutes"] == [15, 60]


def test_label_config_matches_implementation() -> None:
    path = PROJECT_ROOT / "configs/experiments/executable-labels-v1.toml"
    with path.open("rb") as source:
        config = tomllib.load(source)

    assert config["id"] == LABEL_SET_ID
    assert tuple(config["horizons_minutes"]) == DEFAULT_HORIZONS_MINUTES
    assert config["source_bar_interval_minutes"] == BAR_INTERVAL_MINUTES
    assert config["max_entry_latency_minutes"] == MAX_QUOTE_LATENCY_MINUTES
    assert config["max_exit_latency_minutes"] == MAX_QUOTE_LATENCY_MINUTES
    assert config["returns"]["short"] == "1 - exit_ask / entry_bid"

    with (
        PROJECT_ROOT / "configs/experiments/executable-labels-v2.toml"
    ).open("rb") as source:
        v2 = tomllib.load(source)
    assert v2["id"] == LABEL_SET_V2_ID
    assert v2["source"] == "quote-bars-v2"
    assert v2["returns"] == config["returns"]


def test_validation_config_matches_implementation() -> None:
    path = PROJECT_ROOT / "configs/experiments/purged-walk-forward-v1.toml"
    plan = load_validation_plan(path)

    assert plan.id == VALIDATION_SET_ID
    assert plan.strategy == VALIDATION_STRATEGY
    assert plan.interval_semantics == INTERVAL_SEMANTICS
    assert plan.max_horizon_minutes == max(DEFAULT_HORIZONS_MINUTES)
    assert plan.max_quote_latency_minutes == MAX_QUOTE_LATENCY_MINUTES
    assert plan.purge_minutes == (
        max(DEFAULT_HORIZONS_MINUTES) + MAX_QUOTE_LATENCY_MINUTES
    )

    screen = load_validation_plan(
        PROJECT_ROOT / "configs/experiments/causal-v2-screen-2021-v1.toml"
    )
    assert screen.id == SCREEN_VALIDATION_SET_ID
    assert len(screen.folds()) == 1
    assert screen.folds()[0].id == "wf-2021-01"
    assert screen.development_end_exclusive.year == 2022


def test_baseline_config_matches_implementation() -> None:
    path = PROJECT_ROOT / "configs/experiments/baseline-ridge-v1.toml"
    config = load_baseline_config(path)

    assert config.id == MODEL_SET_ID
    assert config.features == FEATURE_COLUMNS
    assert config.horizons_minutes == DEFAULT_HORIZONS_MINUTES
    assert config.action_threshold_bps == 0.0
    assert config.locked_test_policy == "forbidden"

    calibrated = load_baseline_config(
        PROJECT_ROOT / "configs/experiments/baseline-ridge-v2.toml"
    )
    assert calibrated.id == MODEL_SET_V2_ID
    assert calibrated.prediction_set == PREDICTION_SET_V3_ID
    assert calibrated.features == config.features
    assert calibrated.alpha == config.alpha
    assert calibrated.calibration_window_months == 1
    assert calibrated.calibration_purge_minutes == 65

    microstructure = load_baseline_config(
        PROJECT_ROOT / "configs/experiments/baseline-ridge-v3.toml"
    )
    assert microstructure.id == MODEL_SET_V3_ID
    assert microstructure.prediction_set == PREDICTION_SET_V4_ID
    assert microstructure.features == FEATURE_V2_COLUMNS
    assert microstructure.alpha == config.alpha
    assert microstructure.selection_policy == config.selection_policy


def test_portfolio_config_matches_implementation() -> None:
    path = PROJECT_ROOT / "configs/experiments/portfolio-v1.toml"
    config = load_portfolio_config(path)

    assert config.id == PORTFOLIO_SET_ID
    assert config.symbols == PORTFOLIO_SYMBOLS
    assert config.horizons_minutes == PORTFOLIO_HORIZONS
    assert config.initial_capital_usd == 100_000.0
    assert config.target_annual_volatility == 0.10
    assert config.maximum_drawdown == 0.10

    calibrated = load_portfolio_config(
        PROJECT_ROOT / "configs/experiments/portfolio-v2.toml"
    )
    assert calibrated.id == PORTFOLIO_SET_V2_ID
    assert calibrated.prediction_set == PREDICTION_SET_V3_ID
    assert calibrated.initial_capital_usd == config.initial_capital_usd
    assert calibrated.maximum_drawdown == config.maximum_drawdown

    microstructure = load_portfolio_config(
        PROJECT_ROOT / "configs/experiments/portfolio-v3.toml"
    )
    assert microstructure.id == PORTFOLIO_SET_V3_ID
    assert microstructure.prediction_set == PREDICTION_SET_V4_ID
    assert microstructure.validation_set == SCREEN_VALIDATION_SET_ID
    assert microstructure.initial_capital_usd == config.initial_capital_usd


def test_development_acceptance_is_frozen_before_execution() -> None:
    path = PROJECT_ROOT / "configs/experiments/development-acceptance-v1.toml"
    config = load_acceptance_config(path)

    assert config.id == ACCEPTANCE_SET_ID
    assert config.symbols == PORTFOLIO_SYMBOLS
    assert config.horizons_minutes == PORTFOLIO_HORIZONS
    assert config.expected_fold_count == 36
    assert config.expected_stage_count == 42
    assert config.expected_authorized_files == 14
    assert config.expected_source_rows == 1_624_981_795
    assert config.locked_test_policy == "forbidden"
    assert config.validation_config is None
    assert config.sealed_envelope is None

    calibrated = load_acceptance_config(
        PROJECT_ROOT / "configs/experiments/development-acceptance-v2.toml"
    )
    assert calibrated.id == ACCEPTANCE_SET_V2_ID
    assert calibrated.validation_config is None
    assert calibrated.sealed_envelope is None
    assert calibrated.minimum_positive_folds_per_horizon == (
        config.minimum_positive_folds_per_horizon
    )
    assert calibrated.minimum_positive_symbols_per_horizon == (
        config.minimum_positive_symbols_per_horizon
    )
    assert calibrated.minimum_trades_per_symbol_horizon == (
        config.minimum_trades_per_symbol_horizon
    )
    assert calibrated.minimum_total_return_exclusive == (
        config.minimum_total_return_exclusive
    )


def test_locked_test_protocol_is_frozen_before_candidate_selection() -> None:
    path = PROJECT_ROOT / "configs/experiments/locked-test-evaluation-v1.toml"
    config = load_locked_test_config(path)

    assert config.id == LOCKED_TEST_SET_ID
    assert config.one_shot_policy == ONE_SHOT_POLICY
    assert config.symbols == PORTFOLIO_SYMBOLS
    assert config.horizons_minutes == PORTFOLIO_HORIZONS
    assert config.feature_context_bars == 73

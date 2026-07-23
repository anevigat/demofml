import tomllib
from pathlib import Path

from demofml.features.causal import FEATURE_SET_ID
from demofml.labels.executable import (
    BAR_INTERVAL_MINUTES,
    DEFAULT_HORIZONS_MINUTES,
    LABEL_SET_ID,
    MAX_QUOTE_LATENCY_MINUTES,
)
from demofml.models.baseline import FEATURE_COLUMNS, MODEL_SET_ID, load_baseline_config
from demofml.validation.splits import (
    INTERVAL_SEMANTICS,
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


def test_baseline_config_matches_implementation() -> None:
    path = PROJECT_ROOT / "configs/experiments/baseline-ridge-v1.toml"
    config = load_baseline_config(path)

    assert config.id == MODEL_SET_ID
    assert config.features == FEATURE_COLUMNS
    assert config.horizons_minutes == DEFAULT_HORIZONS_MINUTES
    assert config.action_threshold_bps == 0.0
    assert config.locked_test_policy == "forbidden"

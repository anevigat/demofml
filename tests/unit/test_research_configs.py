import tomllib
from pathlib import Path

from demofml.features.causal import FEATURE_SET_ID
from demofml.labels.executable import (
    BAR_INTERVAL_MINUTES,
    DEFAULT_HORIZONS_MINUTES,
    LABEL_SET_ID,
    MAX_QUOTE_LATENCY_MINUTES,
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

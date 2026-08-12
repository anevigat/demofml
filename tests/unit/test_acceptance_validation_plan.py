"""Regression tests for audit findings A7/A8 (Campaign 3 Fase 0).

Before this fix, `reporting/acceptance.py` hardcoded the locked-test boundary
(`2025-01-01`) and the fold-id epoch (`wf-2022-*`) in several places instead
of deriving them from the acceptance contract's own validation plan. That
made the module silently incorrect for any validation plan other than
`purged-walk-forward-v1`'s exact Campaign 1/2 dates. These tests prove: (a)
existing Campaign 1/2 configs keep behaving identically via the documented
fallback, and (b) a config that declares a `validation_config` now derives
both values from the real plan instead.
"""

from datetime import UTC, datetime
from pathlib import Path

from demofml.reporting import acceptance as acceptance_module
from demofml.validation.splits import load_validation_plan

PROJECT_ROOT = Path(__file__).parents[2]
VALIDATION_CONFIG = PROJECT_ROOT / "configs/experiments/purged-walk-forward-v1.toml"
ACCEPTANCE_V1_CONFIG = (
    PROJECT_ROOT / "configs/experiments/development-acceptance-v1.toml"
)
PORTFOLIO_V1_CONFIG = PROJECT_ROOT / "configs/experiments/portfolio-v1.toml"


def test_legacy_acceptance_configs_have_no_validation_config() -> None:
    """Campaign 1/2 configs predate this field; the fallback must be exact."""
    config = acceptance_module.load_acceptance_config(ACCEPTANCE_V1_CONFIG)
    assert config.validation_config is None

    plan = acceptance_module._resolved_validation_plan(config)
    assert plan is None
    assert acceptance_module._locked_test_start(plan) == datetime(
        2025, 1, 1, tzinfo=UTC
    )
    legacy_fold_ids = acceptance_module._expected_fold_ids(plan, 36)
    assert "wf-2022-01" in legacy_fold_ids
    assert "wf-2024-12" in legacy_fold_ids
    assert len(legacy_fold_ids) == 36


def _shifted_validation_config(tmp_path: Path) -> Path:
    """purged-walk-forward-v1, shifted a year later, same id and span."""
    path = tmp_path / "validation.toml"
    path.write_text(
        VALIDATION_CONFIG.read_text()
        .replace("2018-01-01T00:00:00Z", "2019-01-01T00:00:00Z")
        .replace("2022-01-01T00:00:00Z", "2023-01-01T00:00:00Z")
        .replace("2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z")
        .replace("2026-03-11T00:00:00Z", "2027-03-11T00:00:00Z")
    )
    return path


def test_declared_validation_config_overrides_the_legacy_epoch(
    tmp_path: Path,
) -> None:
    """The helpers derive from the real plan, not the 2022/2025 constants."""
    plan = load_validation_plan(_shifted_validation_config(tmp_path))

    locked_test_start = acceptance_module._locked_test_start(plan)
    assert locked_test_start == datetime(2026, 1, 1, tzinfo=UTC)

    fold_ids = acceptance_module._expected_fold_ids(plan, len(plan.folds()))
    assert fold_ids == {fold.id for fold in plan.folds()}
    assert "wf-2022-01" not in fold_ids
    assert "wf-2023-01" in fold_ids
    assert "wf-2025-12" in fold_ids


def test_load_acceptance_config_parses_and_resolves_validation_config(
    tmp_path: Path,
) -> None:
    """End-to-end: the TOML key round-trips into a loadable plan path."""
    validation_path = _shifted_validation_config(tmp_path)
    acceptance_path = tmp_path / "acceptance.toml"
    acceptance_path.write_text(
        ACCEPTANCE_V1_CONFIG.read_text()
        .replace(
            'portfolio_config = "portfolio-v1.toml"',
            f'portfolio_config = "{PORTFOLIO_V1_CONFIG}"',
        )
        .replace(
            'locked_test_policy = "forbidden"',
            'locked_test_policy = "forbidden"\n'
            f'validation_config = "{validation_path.name}"',
        )
    )

    config = acceptance_module.load_acceptance_config(acceptance_path)

    assert config.validation_config == validation_path.resolve()
    plan = acceptance_module._resolved_validation_plan(config)
    assert plan is not None
    assert acceptance_module._locked_test_start(plan) == datetime(
        2026, 1, 1, tzinfo=UTC
    )

"""The acceptance gate refuses to evaluate an unsealed or broken contract.

Campaign 3's protocol (`docs/research/campaign-3-protocol-v1.md`) pre-registers
four documents before the first fold runs. These tests prove the gate enforces
that seal when a contract declares one, refuses to evaluate a contract that is
not the sealed one, and leaves every Campaign 1/2 contract untouched.
"""

from pathlib import Path

import pytest

from demofml.reporting import acceptance as acceptance_module
from demofml.research.envelope import file_sha256

PROJECT_ROOT = Path(__file__).parents[2]
ACCEPTANCE_V1_CONFIG = (
    PROJECT_ROOT / "configs/experiments/development-acceptance-v1.toml"
)
PORTFOLIO_V1_CONFIG = PROJECT_ROOT / "configs/experiments/portfolio-v1.toml"
VALIDATION_CONFIG = PROJECT_ROOT / "configs/experiments/purged-walk-forward-v1.toml"


def _sealed_acceptance(tmp_path: Path) -> tuple[Path, Path]:
    """Write a sealed acceptance contract and the envelope that seals it."""
    root = tmp_path / "repo"
    configs = root / "configs" / "experiments"
    docs = root / "docs" / "research"
    configs.mkdir(parents=True)
    docs.mkdir(parents=True)

    (docs / "hypothesis.md").write_text("hypothesis\n", encoding="utf-8")
    (configs / "model.toml").write_text('id = "model"\n', encoding="utf-8")
    (configs / "validation.toml").write_text(
        VALIDATION_CONFIG.read_text(encoding="utf-8"), encoding="utf-8"
    )
    envelope_path = configs / "envelope.toml"
    acceptance_path = configs / "acceptance.toml"
    acceptance_path.write_text(
        ACCEPTANCE_V1_CONFIG.read_text(encoding="utf-8")
        .replace(
            'portfolio_config = "portfolio-v1.toml"',
            f'portfolio_config = "{PORTFOLIO_V1_CONFIG}"',
        )
        .replace(
            'locked_test_policy = "forbidden"',
            'locked_test_policy = "forbidden"\nsealed_envelope = "envelope.toml"',
        ),
        encoding="utf-8",
    )
    documents = {
        "hypothesis": "docs/research/hypothesis.md",
        "validation": "configs/experiments/validation.toml",
        "model": "configs/experiments/model.toml",
        "acceptance": "configs/experiments/acceptance.toml",
    }
    lines = [
        "format_version = 1",
        'schema = "sealed-envelope-v1"',
        'id = "campaign-3-test-envelope-v1"',
        'campaign = "campaign-3"',
        "sealed_at = 2026-08-13T00:00:00Z",
        'root = "../.."',
    ]
    for role, relative in documents.items():
        lines.append(f"[documents.{role}]")
        lines.append(f'path = "{relative}"')
        lines.append(f'sha256 = "{file_sha256(root / relative)}"')
    envelope_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return acceptance_path, envelope_path


def test_legacy_acceptance_configs_declare_no_seal() -> None:
    """Campaign 1/2 contracts predate the envelope and must stay unaffected."""
    config = acceptance_module.load_acceptance_config(ACCEPTANCE_V1_CONFIG)

    assert config.sealed_envelope is None
    assert (
        acceptance_module._verified_sealed_envelope(config, ACCEPTANCE_V1_CONFIG)
        is None
    )


def test_declared_seal_is_loaded_and_verified(tmp_path: Path) -> None:
    acceptance_path, envelope_path = _sealed_acceptance(tmp_path)

    config = acceptance_module.load_acceptance_config(acceptance_path)
    assert config.sealed_envelope == envelope_path.resolve()

    envelope = acceptance_module._verified_sealed_envelope(config, acceptance_path)
    assert envelope is not None
    assert envelope.id == "campaign-3-test-envelope-v1"


def test_relaxing_a_sealed_threshold_breaks_the_gate(tmp_path: Path) -> None:
    """The failure mode the seal exists for: post-hoc threshold edits."""
    acceptance_path, _ = _sealed_acceptance(tmp_path)
    config = acceptance_module.load_acceptance_config(acceptance_path)
    acceptance_path.write_text(
        acceptance_path.read_text(encoding="utf-8").replace(
            "minimum_positive_symbols_per_horizon = 6",
            "minimum_positive_symbols_per_horizon = 1",
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="is broken"):
        acceptance_module._verified_sealed_envelope(config, acceptance_path)


def test_a_substituted_contract_is_rejected_even_when_the_seal_holds(
    tmp_path: Path,
) -> None:
    """Evaluating a different file leaves the sealed documents intact."""
    acceptance_path, _ = _sealed_acceptance(tmp_path)
    config = acceptance_module.load_acceptance_config(acceptance_path)
    substitute = acceptance_path.with_name("acceptance-copy.toml")
    substitute.write_text(
        acceptance_path.read_text(encoding="utf-8"), encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="not the sealed one"):
        acceptance_module._verified_sealed_envelope(config, substitute)


def test_the_seal_is_checked_before_any_run_artifact_is_read(tmp_path: Path) -> None:
    """A broken seal must fail loudly, not fall through to a missing-run error."""
    acceptance_path, _ = _sealed_acceptance(tmp_path)
    (tmp_path / "repo" / "docs" / "research" / "hypothesis.md").write_text(
        "rewritten after the fact\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="is broken"):
        acceptance_module.evaluate_development_run(
            tmp_path / "absent-run", acceptance_path
        )

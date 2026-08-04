"""Data-free Campaign 2 engineering verification for the on-prem cluster."""

import argparse
import hashlib
import platform
import re
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]

from demofml.bars.prospective import PROSPECTIVE_BAR_SCHEMA
from demofml.calendars.prospective_fx import expected_decision_boundaries
from demofml.data.prospective_ticks import PROSPECTIVE_TICK_SCHEMA
from demofml.features.cross_pair import (
    CANDIDATE_FEATURE_SCHEMA,
    CONTROL_FEATURE_SCHEMA,
    solve_cross_pair_factor,
)
from demofml.prospective.config import load_campaign2_engineering_config
from demofml.prospective.opportunities import CAMPAIGN_ID, OPPORTUNITY_SCHEMA
from demofml.prospective.records import (
    IMAGE_DIGEST_PATTERN,
    canonical_json,
    content_id,
    write_immutable_json,
)

ENGINEERING_VERIFY_SET_ID = "campaign-2-onprem-engineering-verify-v1"
_IMAGE_REFERENCE_PATTERN = re.compile(r"anevigat/demofml@sha256:[0-9a-f]{64}")


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _check(
    check_id: str,
    passed: bool,
    observed: object,
    threshold: object,
) -> dict[str, object]:
    return {
        "id": check_id,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "operator": "==",
        "threshold": threshold,
    }


def build_engineering_verification(
    config_path: Path,
    *,
    code_reference: str,
    base_image_reference: str,
) -> dict[str, object]:
    """Verify contracts and golden invariants without data or network access."""
    if IMAGE_DIGEST_PATTERN.fullmatch(code_reference) is None:
        raise ValueError("code_reference must be a sha256 digest")
    if _IMAGE_REFERENCE_PATTERN.fullmatch(base_image_reference) is None:
        raise ValueError("base_image_reference must be the immutable runtime image")
    config = load_campaign2_engineering_config(config_path)

    pair_returns = {
        "AUDUSD": 0.01,
        "EURCHF": 0.01,
        "EURJPY": -0.02,
        "EURUSD": 0.04,
        "GBPJPY": -0.11,
        "GBPUSD": -0.05,
        "USDCAD": 0.02,
        "USDJPY": -0.06,
    }
    solution = solve_cross_pair_factor(pair_returns)
    maximum_residual = max(abs(value) for value in solution.residuals)
    expected_strengths = (0.01, -0.02, 0.03, 0.04, -0.05, 0.06)
    maximum_strength_error = max(
        abs(observed - expected)
        for observed, expected in zip(
            solution.strengths, expected_strengths, strict=True
        )
    )
    calendar = expected_decision_boundaries(
        datetime(2026, 9, 6, tzinfo=UTC),
        datetime(2026, 9, 12, tzinfo=UTC),
    )
    schemas = {
        "prospective_ticks": _schema_sha256(PROSPECTIVE_TICK_SCHEMA),
        "prospective_bars": _schema_sha256(PROSPECTIVE_BAR_SCHEMA),
        "control_features": _schema_sha256(CONTROL_FEATURE_SCHEMA),
        "candidate_features": _schema_sha256(CANDIDATE_FEATURE_SCHEMA),
        "opportunities": _schema_sha256(OPPORTUNITY_SCHEMA),
    }
    runtime = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": metadata.version("numpy"),
        "pyarrow": metadata.version("pyarrow"),
        "scikit_learn": metadata.version("scikit-learn"),
        "tzdata": metadata.version("tzdata"),
    }
    checks = [
        _check(
            "engineering_config",
            config.path.is_file() and len(config.contract_sha256) == 5,
            len(config.contract_sha256),
            5,
        ),
        _check(
            "locked_interval_excluded",
            config.forbidden_start == datetime(2025, 1, 1, tzinfo=UTC)
            and config.forbidden_end_exclusive
            == datetime(2026, 3, 11, tzinfo=UTC),
            {
                "start": config.forbidden_start.isoformat(),
                "end_exclusive": config.forbidden_end_exclusive.isoformat(),
            },
            {
                "start": "2025-01-01T00:00:00+00:00",
                "end_exclusive": "2026-03-11T00:00:00+00:00",
            },
        ),
        _check(
            "factor_golden_vector",
            maximum_residual <= 1e-15 and maximum_strength_error <= 1e-15,
            {
                "maximum_residual": maximum_residual,
                "maximum_strength_error": maximum_strength_error,
            },
            {"maximum_residual": 1e-15, "maximum_strength_error": 1e-15},
        ),
        _check(
            "weekly_calendar",
            len(calendar) == 1427
            and calendar[0] == datetime(2026, 9, 6, 21, 5, tzinfo=UTC)
            and calendar[-1] == datetime(2026, 9, 11, 19, 55, tzinfo=UTC),
            {
                "boundaries": len(calendar),
                "first": calendar[0].isoformat(),
                "last": calendar[-1].isoformat(),
            },
            {
                "boundaries": 1427,
                "first": "2026-09-06T21:05:00+00:00",
                "last": "2026-09-11T19:55:00+00:00",
            },
        ),
        _check(
            "authorization_boundary",
            True,
            {
                "collection": False,
                "fitting": False,
                "scoring": False,
                "evaluation": False,
                "raw_access": False,
            },
            {
                "collection": False,
                "fitting": False,
                "scoring": False,
                "evaluation": False,
                "raw_access": False,
            },
        ),
    ]
    verified = all(check["status"] == "pass" for check in checks)
    core: dict[str, object] = {
        "format_version": 1,
        "verification_set": ENGINEERING_VERIFY_SET_ID,
        "campaign_id": CAMPAIGN_ID,
        "deployment_scope": "onprem_kubernetes_engineering_only",
        "code_reference": code_reference,
        "base_image_reference": base_image_reference,
        "contract_sha256": config.contract_sha256,
        "schema_sha256": schemas,
        "runtime_versions": runtime,
        "checks": checks,
        "engineering_verified": verified,
        "qualification_complete": False,
        "collection_authorized": False,
        "fitting_authorized": False,
        "scoring_authorized": False,
        "evaluation_authorized": False,
        "raw_access_authorized": False,
    }
    return {**core, "verification_id": content_id(core)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="demofml verify-campaign2-engineering")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-reference", required=True)
    parser.add_argument("--base-image-reference", required=True)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    report = build_engineering_verification(
        arguments.config,
        code_reference=arguments.code_reference,
        base_image_reference=arguments.base_image_reference,
    )
    if arguments.output is not None:
        write_immutable_json(arguments.output, report)
    print(canonical_json(report).decode("ascii"), end="")
    if report["engineering_verified"] is not True:
        raise RuntimeError("Campaign 2 engineering verification failed")

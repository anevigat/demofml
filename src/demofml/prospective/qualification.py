"""Outcome-free Campaign 2 engineering qualification envelope."""

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from demofml.calendars.prospective_fx import expected_decision_boundaries
from demofml.features.cross_pair import PAIRS
from demofml.prospective.bundle import (
    ENGINEERING_BUNDLE_SCOPE,
    verify_engineering_bundle,
)
from demofml.prospective.config import (
    MAXIMUM_CONSECUTIVE_MISSING_BARS,
    MAXIMUM_FEATURE_BUILD_SECONDS_PER_BOUNDARY,
    MAXIMUM_PEAK_RSS_BYTES,
    MINIMUM_COMPLETE_RATIO,
    Campaign2EngineeringConfig,
)
from demofml.prospective.custody import validate_collection_terminal
from demofml.prospective.opportunities import CAMPAIGN_ID, CoverageReport
from demofml.prospective.records import (
    CONTENT_ID_PATTERN,
    SHA256_PATTERN,
    canonical_json,
    content_id,
    write_immutable_json,
)

QUALIFICATION_SET_ID = "campaign-2-engineering-qualification-v1"
QUALIFICATION_SCOPE = "engineering_prequalification_claims_no_scores_or_outcomes"
_CHECK_OPERATORS = {
    "engineering_bundle_scope": "==",
    "schema_contracts": "==",
    "collection_interval": "==",
    "collection_quality": "==",
    "collection_row_floor": ">=",
    "coverage_global": ">=",
    "coverage_monthly": "all >=",
    "maximum_consecutive_missing": "<=",
    "determinism": "digests_equal",
    "feature_runtime": "<=",
    "peak_memory": "<=",
}


@dataclass(frozen=True)
class QualificationMeasurements:
    """Supervision-free determinism and resource measurements."""

    schema_valid: bool
    first_determinism_sha256: str
    second_determinism_sha256: str
    feature_build_seconds_per_boundary: float
    peak_rss_bytes: int
    opportunity_ledger_sha256: str


def _check(
    check_id: str,
    passed: bool,
    observed: object,
    operator: str,
    threshold: object,
) -> dict[str, object]:
    return {
        "id": check_id,
        "status": "pass" if passed else "fail",
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
    }


def _parse_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} is not a valid UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")
    return parsed.astimezone(UTC)


def _collection_within_qualification(
    config: Campaign2EngineeringConfig,
    chains: Mapping[str, Sequence[Mapping[str, object]]],
) -> bool:
    for chain in chains.values():
        for segment in chain:
            object_record = segment.get("object")
            if not isinstance(object_record, dict):
                raise ValueError("collection segment object must be a mapping")
            provider_start = _parse_utc(
                object_record.get("provider_start"), "provider_start"
            )
            provider_end = _parse_utc(
                object_record.get("provider_end_exclusive"),
                "provider_end_exclusive",
            )
            received_start = _parse_utc(
                object_record.get("received_start"), "received_start"
            )
            received_end = _parse_utc(
                object_record.get("received_end_exclusive"),
                "received_end_exclusive",
            )
            if not (
                config.qualification_start <= provider_start
                and provider_end <= config.prospective_start
                and config.qualification_start <= received_start
                and received_end <= config.prospective_start
            ):
                return False
    return True


def build_qualification_envelope(
    *,
    config: Campaign2EngineeringConfig,
    bundle_root: Path,
    expected_bundle_id: str,
    collection_terminal: Mapping[str, object],
    collection_chains: Mapping[str, Sequence[Mapping[str, object]]],
    coverage: CoverageReport,
    expected_boundaries: tuple[datetime, ...],
    measurements: QualificationMeasurements,
) -> dict[str, object]:
    """Build qualification evidence while refusing to grant any capability."""
    bundle = verify_engineering_bundle(bundle_root, expected_bundle_id)
    validate_collection_terminal(collection_terminal, collection_chains)
    if set(collection_chains) != set(PAIRS):
        raise ValueError("qualification requires all eight collection chains")
    expected_calendar = expected_decision_boundaries(
        config.qualification_start, config.prospective_start
    )
    if expected_boundaries != expected_calendar:
        raise ValueError("qualification boundaries must cover the frozen interval")
    if coverage.expected_sections != len(expected_boundaries):
        raise ValueError("coverage does not reconcile with expected boundaries")
    expected_months = {boundary.strftime("%Y-%m") for boundary in expected_boundaries}
    if (
        not isinstance(coverage.expected_sections, int)
        or isinstance(coverage.expected_sections, bool)
        or not isinstance(coverage.complete_sections, int)
        or isinstance(coverage.complete_sections, bool)
        or not isinstance(coverage.complete_ratio, float)
        or not isinstance(coverage.maximum_consecutive_missing, int)
        or isinstance(coverage.maximum_consecutive_missing, bool)
        or set(coverage.monthly_complete_ratio) != expected_months
        or any(
            not isinstance(ratio, float)
            or not math.isfinite(ratio)
            or not 0.0 <= ratio <= 1.0
            for ratio in coverage.monthly_complete_ratio.values()
        )
    ):
        raise ValueError("coverage report types or months are incompatible")
    monthly_expected: dict[str, int] = {}
    for boundary in expected_boundaries:
        month = boundary.strftime("%Y-%m")
        monthly_expected[month] = monthly_expected.get(month, 0) + 1
    monthly_complete = {
        month: round(coverage.monthly_complete_ratio[month] * expected)
        for month, expected in monthly_expected.items()
    }
    monthly_counts_reconcile = all(
        math.isclose(
            coverage.monthly_complete_ratio[month] * expected,
            monthly_complete[month],
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for month, expected in monthly_expected.items()
    ) and sum(monthly_complete.values()) == coverage.complete_sections
    total_missing = coverage.expected_sections - coverage.complete_sections
    if (
        not 0 <= coverage.complete_sections <= coverage.expected_sections
        or not math.isclose(
            coverage.complete_ratio,
            coverage.complete_sections / coverage.expected_sections,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        or coverage.maximum_consecutive_missing < 0
        or coverage.maximum_consecutive_missing > total_missing
        or (
            coverage.complete_sections < coverage.expected_sections
            and coverage.maximum_consecutive_missing == 0
        )
        or not monthly_counts_reconcile
    ):
        raise ValueError("coverage report is internally incompatible")
    digests = (
        measurements.first_determinism_sha256,
        measurements.second_determinism_sha256,
        measurements.opportunity_ledger_sha256,
    )
    if any(SHA256_PATTERN.fullmatch(value) is None for value in digests):
        raise ValueError("qualification measurements require SHA-256 digests")
    if (
        not isinstance(measurements.peak_rss_bytes, int)
        or isinstance(measurements.peak_rss_bytes, bool)
        or measurements.peak_rss_bytes < 0
        or not math.isfinite(measurements.feature_build_seconds_per_boundary)
        or measurements.feature_build_seconds_per_boundary < 0.0
    ):
        raise ValueError("qualification runtime measurements are invalid")
    terminal_chains = collection_terminal.get("chains")
    if not isinstance(terminal_chains, list):
        raise ValueError("collection terminal chains must be a list")
    critical_violations = sum(
        int(record["critical_violations"])
        for record in terminal_chains
        if isinstance(record, dict)
    )
    collection_rows = sum(
        int(record["rows"])
        for record in terminal_chains
        if isinstance(record, dict)
    )
    minimum_collection_rows = len(expected_boundaries) * len(PAIRS)
    collection_interval_valid = _collection_within_qualification(
        config, collection_chains
    )
    calendar_sha256 = hashlib.sha256(
        canonical_json(
            [
                boundary.astimezone(UTC).isoformat().replace("+00:00", "Z")
                for boundary in expected_boundaries
            ]
        )
    ).hexdigest()
    checks = [
        _check(
            "engineering_bundle_scope",
            bundle.get("bundle_scope") == ENGINEERING_BUNDLE_SCOPE
            and bundle.get("scoring_authorized") is False,
            bundle.get("bundle_scope"),
            "==",
            ENGINEERING_BUNDLE_SCOPE,
        ),
        _check(
            "schema_contracts",
            measurements.schema_valid,
            measurements.schema_valid,
            "==",
            True,
        ),
        _check(
            "collection_interval",
            collection_interval_valid,
            collection_interval_valid,
            "==",
            True,
        ),
        _check(
            "collection_quality",
            critical_violations == 0,
            critical_violations,
            "==",
            0,
        ),
        _check(
            "collection_row_floor",
            collection_rows >= minimum_collection_rows,
            collection_rows,
            ">=",
            minimum_collection_rows,
        ),
        _check(
            "coverage_global",
            coverage.complete_ratio >= config.minimum_complete_ratio,
            coverage.complete_ratio,
            ">=",
            config.minimum_complete_ratio,
        ),
        _check(
            "coverage_monthly",
            all(
                ratio >= config.minimum_complete_ratio
                for ratio in coverage.monthly_complete_ratio.values()
            ),
            coverage.monthly_complete_ratio,
            "all >=",
            config.minimum_complete_ratio,
        ),
        _check(
            "maximum_consecutive_missing",
            coverage.maximum_consecutive_missing
            <= config.maximum_consecutive_missing_bars,
            coverage.maximum_consecutive_missing,
            "<=",
            config.maximum_consecutive_missing_bars,
        ),
        _check(
            "determinism",
            measurements.first_determinism_sha256
            == measurements.second_determinism_sha256,
            {
                "first": measurements.first_determinism_sha256,
                "second": measurements.second_determinism_sha256,
            },
            "digests_equal",
            True,
        ),
        _check(
            "feature_runtime",
            measurements.feature_build_seconds_per_boundary
            <= config.maximum_feature_build_seconds_per_boundary,
            measurements.feature_build_seconds_per_boundary,
            "<=",
            config.maximum_feature_build_seconds_per_boundary,
        ),
        _check(
            "peak_memory",
            measurements.peak_rss_bytes <= config.maximum_peak_rss_bytes,
            measurements.peak_rss_bytes,
            "<=",
            config.maximum_peak_rss_bytes,
        ),
    ]
    checks_passed = all(check["status"] == "pass" for check in checks)
    core: dict[str, object] = {
        "format_version": 1,
        "qualification_set": QUALIFICATION_SET_ID,
        "campaign_id": CAMPAIGN_ID,
        "qualification_scope": QUALIFICATION_SCOPE,
        "bundle_id": bundle["bundle_id"],
        "collection_terminal_id": collection_terminal["terminal_id"],
        "expected_calendar_sha256": calendar_sha256,
        "opportunity_ledger_sha256": measurements.opportunity_ledger_sha256,
        "coverage": {
            "expected_sections": coverage.expected_sections,
            "complete_sections": coverage.complete_sections,
            "complete_ratio": coverage.complete_ratio,
            "maximum_consecutive_missing": coverage.maximum_consecutive_missing,
            "monthly_expected_sections": monthly_expected,
            "monthly_complete_sections": monthly_complete,
            "monthly_complete_ratio": coverage.monthly_complete_ratio,
        },
        "checks": checks,
        "engineering_checks_passed": checks_passed,
        "qualification_complete": False,
        "external_attestation_required": True,
        "authorization_granted": False,
        "collection_authorized": False,
        "scoring_authorized": False,
        "evaluation_authorized": False,
    }
    return {**core, "qualification_id": content_id(core)}


def validate_qualification_envelope(envelope: Mapping[str, object]) -> None:
    """Validate a qualification result without converting it to authorization."""
    expected = {
        "format_version",
        "qualification_set",
        "campaign_id",
        "qualification_scope",
        "bundle_id",
        "collection_terminal_id",
        "expected_calendar_sha256",
        "opportunity_ledger_sha256",
        "coverage",
        "checks",
        "engineering_checks_passed",
        "qualification_complete",
        "external_attestation_required",
        "authorization_granted",
        "collection_authorized",
        "scoring_authorized",
        "evaluation_authorized",
        "qualification_id",
    }
    if set(envelope) != expected:
        raise ValueError("qualification envelope fields are incompatible")
    if (
        envelope["format_version"] != 1
        or envelope["qualification_set"] != QUALIFICATION_SET_ID
        or envelope["campaign_id"] != CAMPAIGN_ID
        or envelope["qualification_scope"] != QUALIFICATION_SCOPE
        or envelope["qualification_complete"] is not False
        or envelope["external_attestation_required"] is not True
        or envelope["authorization_granted"] is not False
        or envelope["collection_authorized"] is not False
        or envelope["scoring_authorized"] is not False
        or envelope["evaluation_authorized"] is not False
    ):
        raise ValueError("qualification authorization boundary is incompatible")
    for field in ("bundle_id", "collection_terminal_id", "qualification_id"):
        value = envelope[field]
        if not isinstance(value, str) or CONTENT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError(f"qualification {field} is invalid")
    for field in ("expected_calendar_sha256", "opportunity_ledger_sha256"):
        value = envelope[field]
        if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
            raise ValueError(f"qualification {field} is invalid")
    checks = envelope["checks"]
    if not isinstance(checks, list) or len(checks) != len(_CHECK_OPERATORS):
        raise ValueError("qualification checks must match the frozen set")
    if any(
        not isinstance(check, dict)
        or set(check) != {"id", "status", "observed", "operator", "threshold"}
        or check["status"] not in {"pass", "fail"}
        for check in checks
    ):
        raise ValueError("qualification checks are incompatible")
    if tuple(check["id"] for check in checks) != tuple(_CHECK_OPERATORS):
        raise ValueError("qualification check IDs are incompatible")
    coverage = envelope["coverage"]
    if not isinstance(coverage, dict) or set(coverage) != {
        "expected_sections",
        "complete_sections",
        "complete_ratio",
        "maximum_consecutive_missing",
        "monthly_expected_sections",
        "monthly_complete_sections",
        "monthly_complete_ratio",
    }:
        raise ValueError("qualification coverage fields are incompatible")
    _validate_envelope_coverage(coverage)
    by_id = {str(check["id"]): check for check in checks}
    if (
        by_id["coverage_global"]["observed"] != coverage["complete_ratio"]
        or type(by_id["coverage_global"]["threshold"]) is not float
        or by_id["coverage_global"]["threshold"] != MINIMUM_COMPLETE_RATIO
        or by_id["coverage_monthly"]["observed"]
        != coverage["monthly_complete_ratio"]
        or type(by_id["coverage_monthly"]["threshold"]) is not float
        or by_id["coverage_monthly"]["threshold"] != MINIMUM_COMPLETE_RATIO
        or by_id["maximum_consecutive_missing"]["observed"]
        != coverage["maximum_consecutive_missing"]
        or type(by_id["maximum_consecutive_missing"]["threshold"]) is not int
        or by_id["maximum_consecutive_missing"]["threshold"]
        != MAXIMUM_CONSECUTIVE_MISSING_BARS
        or type(by_id["feature_runtime"]["threshold"]) is not float
        or by_id["feature_runtime"]["threshold"]
        != MAXIMUM_FEATURE_BUILD_SECONDS_PER_BOUNDARY
        or type(by_id["peak_memory"]["threshold"]) is not int
        or by_id["peak_memory"]["threshold"] != MAXIMUM_PEAK_RSS_BYTES
    ):
        raise ValueError("qualification checks do not bind frozen evidence")
    if any(
        check["operator"] != _CHECK_OPERATORS[str(check["id"])]
        or (check["status"] == "pass") is not _recompute_check(check)
        for check in checks
    ):
        raise ValueError("qualification check semantics are incompatible")
    checks_passed = all(check["status"] == "pass" for check in checks)
    if type(envelope["engineering_checks_passed"]) is not bool or (
        envelope["engineering_checks_passed"] is not checks_passed
    ):
        raise ValueError("qualification summary does not reconcile with checks")
    core = {key: value for key, value in envelope.items() if key != "qualification_id"}
    if envelope["qualification_id"] != content_id(core):
        raise ValueError("qualification content ID mismatch")


def _recompute_check(check: Mapping[str, object]) -> bool:
    operator = check["operator"]
    observed = check["observed"]
    threshold = check["threshold"]
    if operator == "==":
        return type(observed) is type(threshold) and observed == threshold
    if operator in {">=", "<="}:
        if (
            not isinstance(observed, (int, float))
            or isinstance(observed, bool)
            or not isinstance(threshold, (int, float))
            or isinstance(threshold, bool)
        ):
            return False
        return observed >= threshold if operator == ">=" else observed <= threshold
    if operator == "all >=":
        return (
            isinstance(observed, dict)
            and bool(observed)
            and isinstance(threshold, (int, float))
            and not isinstance(threshold, bool)
            and all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and value >= threshold
                for value in observed.values()
            )
        )
    if operator == "digests_equal":
        return (
            isinstance(observed, dict)
            and set(observed) == {"first", "second"}
            and observed["first"] == observed["second"]
            and threshold is True
        )
    return False


def _validate_envelope_coverage(coverage: Mapping[str, object]) -> None:
    integer_fields = (
        "expected_sections",
        "complete_sections",
        "maximum_consecutive_missing",
    )
    if any(
        not isinstance(coverage[field], int) or isinstance(coverage[field], bool)
        for field in integer_fields
    ):
        raise ValueError("qualification coverage counts must be integers")
    expected = cast(int, coverage["expected_sections"])
    complete = cast(int, coverage["complete_sections"])
    maximum_missing = cast(int, coverage["maximum_consecutive_missing"])
    ratio = coverage["complete_ratio"]
    monthly_expected = coverage["monthly_expected_sections"]
    monthly_complete = coverage["monthly_complete_sections"]
    monthly_ratio = coverage["monthly_complete_ratio"]
    if (
        expected <= 0
        or not 0 <= complete <= expected
        or not isinstance(ratio, float)
        or not math.isclose(ratio, complete / expected, rel_tol=0.0, abs_tol=1e-15)
        or not 0 <= maximum_missing <= expected - complete
        or (complete < expected and maximum_missing == 0)
        or not isinstance(monthly_expected, dict)
        or not isinstance(monthly_complete, dict)
        or not isinstance(monthly_ratio, dict)
        or not monthly_expected
        or set(monthly_expected) != set(monthly_complete)
        or set(monthly_expected) != set(monthly_ratio)
    ):
        raise ValueError("qualification coverage summary is incompatible")
    for month in monthly_expected:
        month_expected = monthly_expected[month]
        month_complete = monthly_complete[month]
        month_ratio = monthly_ratio[month]
        if (
            not isinstance(month, str)
            or not isinstance(month_expected, int)
            or isinstance(month_expected, bool)
            or not isinstance(month_complete, int)
            or isinstance(month_complete, bool)
            or not isinstance(month_ratio, float)
            or month_expected <= 0
            or not 0 <= month_complete <= month_expected
            or not math.isclose(
                month_ratio,
                month_complete / month_expected,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("qualification monthly coverage is incompatible")
    if sum(monthly_expected.values()) != expected or sum(
        monthly_complete.values()
    ) != complete:
        raise ValueError("qualification monthly coverage does not reconcile")


def publish_qualification_envelope(path: Path, envelope: Mapping[str, object]) -> None:
    """Publish one immutable qualification result after strict validation."""
    validate_qualification_envelope(envelope)
    write_immutable_json(path, dict(envelope))

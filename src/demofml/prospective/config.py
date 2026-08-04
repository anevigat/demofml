"""Strict engineering-only Campaign 2 configuration loader."""

import hashlib
import tomllib
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from demofml.calendars.prospective_fx import (
    CALENDAR_ID,
    TIMEZONE_NAME,
    TZDATA_VERSION,
)
from demofml.data.prospective_ticks import MAX_PROVIDER_CLOCK_LEAD_NS
from demofml.features.cross_pair import (
    CROSS_PAIR_COLUMNS,
    CROSS_PAIR_FEATURE_SET_ID,
    CURRENCIES,
    PAIRS,
)
from demofml.prospective.opportunities import CAMPAIGN_ID, HORIZONS_MINUTES

ENGINEERING_STATUS = "engineering_only"
ENGINEERING_AUTHORIZED_ON = date(2026, 8, 4)
MINIMUM_COMPLETE_RATIO = 0.95
MAXIMUM_CONSECUTIVE_MISSING_BARS = 36
MAXIMUM_FEATURE_BUILD_SECONDS_PER_BOUNDARY = 1.0
MAXIMUM_PEAK_RSS_BYTES = 1_073_741_824
_EXPECTED_TIMESTAMPS = {
    "historical_fit_start": datetime(2018, 1, 1, tzinfo=UTC),
    "historical_fit_end_exclusive": datetime(2025, 1, 1, tzinfo=UTC),
    "forbidden_start": datetime(2025, 1, 1, tzinfo=UTC),
    "forbidden_end_exclusive": datetime(2026, 3, 11, tzinfo=UTC),
    "qualification_start": datetime(2026, 3, 11, tzinfo=UTC),
    "context_start": datetime(2026, 8, 31, 18, tzinfo=UTC),
    "prospective_start": datetime(2026, 9, 1, tzinfo=UTC),
    "decision_end_exclusive": datetime(2027, 8, 31, 22, 55, tzinfo=UTC),
    "prospective_end_exclusive": datetime(2027, 9, 1, tzinfo=UTC),
}


@dataclass(frozen=True)
class Campaign2EngineeringConfig:
    """Validated Campaign 2 contract with every unsafe authorization disabled."""

    path: Path
    project_root: Path
    protocol_path: Path
    bar_config_path: Path
    feature_config_path: Path
    label_contract_path: Path
    historical_fit_start: datetime
    historical_fit_end_exclusive: datetime
    forbidden_start: datetime
    forbidden_end_exclusive: datetime
    qualification_start: datetime
    context_start: datetime
    prospective_start: datetime
    decision_end_exclusive: datetime
    prospective_end_exclusive: datetime
    minimum_complete_ratio: float
    maximum_consecutive_missing_bars: int
    maximum_feature_build_seconds_per_boundary: float
    maximum_peak_rss_bytes: int

    @property
    def contract_paths(self) -> tuple[Path, ...]:
        """Return role-ordered files that define the engineering contract."""
        return (
            self.path,
            self.protocol_path,
            self.bar_config_path,
            self.feature_config_path,
            self.label_contract_path,
        )

    @property
    def contract_sha256(self) -> dict[str, str]:
        """Return role-keyed hashes without loading any research data."""
        roles = ("engineering", "protocol", "bars", "features", "labels")
        return {
            role: hashlib.sha256(path.read_bytes()).hexdigest()
            for role, path in zip(roles, self.contract_paths, strict=True)
        }


def _load_toml(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be a regular non-symlink file: {path}")
    try:
        with path.open("rb") as source:
            values = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError(f"{name} is invalid: {path}") from error
    return values


def _exact_fields(values: dict[str, Any], expected: set[str], name: str) -> None:
    if set(values) != expected:
        raise ValueError(f"{name} fields are incompatible")


def _string(values: dict[str, Any], key: str) -> str:
    value = values.get(key)
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string")
    return value


def _string_tuple(values: dict[str, Any], key: str) -> tuple[str, ...]:
    value = values.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} must be an array of strings")
    return tuple(value)


def _integer(values: dict[str, Any], key: str) -> int:
    value = values.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _integer_tuple(values: dict[str, Any], key: str) -> tuple[int, ...]:
    value = values.get(key)
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array of integers")
    return tuple(_integer({"value": item}, "value") for item in value)


def _parse_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a canonical UTC timestamp ending in Z")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} is not a valid UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")
    canonical = parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if canonical != value:
        raise ValueError(f"{name} is not canonically encoded")
    return parsed.astimezone(UTC)


def _project_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    raise ValueError("Campaign 2 config must reside inside the project root")


def _resolve_reference(root: Path, base: Path, raw: object, role: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{role} reference must be a non-empty relative path")
    candidate = Path(raw)
    if candidate.is_absolute():
        raise ValueError(f"{role} reference cannot be absolute")
    unresolved = base / candidate
    if unresolved.is_symlink():
        raise ValueError(f"{role} reference cannot be a symlink")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{role} reference escapes the project root") from error
    if not resolved.is_file() or resolved.is_symlink():
        raise ValueError(f"{role} reference must be a regular file")
    return resolved


def _validate_bar_contract(path: Path) -> None:
    values = _load_toml(path, "prospective bar config")
    _exact_fields(
        values,
        {
            "id",
            "source",
            "interval_minutes",
            "interval_semantics",
            "finalization",
            "late_message_policy",
            "compute_allowance_seconds",
            "finish_policy",
            "persist_watermark_provider_time",
            "persist_watermark_ingest_sequence",
            "receipt_clock",
        },
        "prospective bar config",
    )
    expected_clock = {
        "timestamp_type": "utc_nanoseconds",
        "attestation_required": True,
        "maximum_provider_clock_lead_ns": MAX_PROVIDER_CLOCK_LEAD_NS,
        "sequence_type": "strictly_increasing_uint64",
    }
    receipt_clock = values["receipt_clock"]
    if (
        values["id"] != "prospective-quote-bars-v1"
        or values["source"] != "prospective-ticks-v1"
        or _integer(values, "interval_minutes") != 5
        or values["interval_semantics"] != "half_open_provider_time"
        or values["finalization"]
        != "next_received_quote_provider_time_gte_bar_end"
        or values["late_message_policy"]
        != "quarantine_invalidate_boundary_and_fail_stream"
        or _integer(values, "compute_allowance_seconds") != 1
        or values["finish_policy"] != "discard_unwatermarked_open_bar"
        or values["persist_watermark_provider_time"] is not True
        or values["persist_watermark_ingest_sequence"] is not True
        or not isinstance(receipt_clock, dict)
        or set(receipt_clock) != set(expected_clock)
        or any(
            type(receipt_clock[key]) is not type(expected)
            for key, expected in expected_clock.items()
        )
        or receipt_clock != expected_clock
    ):
        raise ValueError("prospective bar config values are incompatible")


def _validate_feature_contract(path: Path) -> None:
    values = _load_toml(path, "cross-pair feature config")
    _exact_fields(
        values,
        {
            "id",
            "control",
            "source",
            "pairs",
            "currencies",
            "usd_strength",
            "solver",
            "residual_dispersion",
            "windows_bars",
            "gap_policy",
            "missing_pair_policy",
            "forward_fill",
            "features",
        },
        "cross-pair feature config",
    )
    if (
        values["id"] != CROSS_PAIR_FEATURE_SET_ID
        or values["control"] != "causal-v1-prospective-control-v1"
        or values["source"] != "prospective-quote-bars-v1"
        or _string_tuple(values, "pairs") != PAIRS
        or _string_tuple(values, "currencies") != CURRENCIES
        or not isinstance(values["usd_strength"], float)
        or values["usd_strength"] != 0.0
        or values["solver"] != "closed_form_float64_v1"
        or values["residual_dispersion"] != "two_pass_population_std"
        or not isinstance(values["windows_bars"], list)
        or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in values["windows_bars"]
        )
        or values["windows_bars"] != [1, 3, 12]
        or values["gap_policy"] != "reset_cross_pair_windows"
        or values["missing_pair_policy"] != "mark_both_arms_missing"
        or values["forward_fill"] is not False
        or _string_tuple(values, "features") != CROSS_PAIR_COLUMNS
    ):
        raise ValueError("cross-pair feature config values are incompatible")


def _validate_label_contract(path: Path) -> None:
    values = _load_toml(path, "prospective label contract")
    _exact_fields(
        values,
        {
            "id",
            "status",
            "source",
            "horizons_minutes",
            "decision_publication_deadline_minutes",
            "max_entry_wait_minutes",
            "max_exit_wait_minutes",
            "entry",
            "exit",
            "missing_policy",
            "returns",
        },
        "prospective label contract",
    )
    if (
        values["id"] != "prospective-executable-v1"
        or values["status"] != "contract_only_scoring_not_authorized"
        or values["source"] != "prospective-ticks-v1"
        or _integer_tuple(values, "horizons_minutes") != HORIZONS_MINUTES
        or _integer(values, "decision_publication_deadline_minutes") != 5
        or _integer(values, "max_entry_wait_minutes") != 5
        or _integer(values, "max_exit_wait_minutes") != 5
        or values["entry"]
        != "first_received_executable_quote_after_published_at"
        or values["exit"]
        != "first_received_executable_quote_at_or_after_decision_plus_horizon"
        or values["missing_policy"] != "paired_flat_expected_opportunity"
        or values["returns"]
        != {
            "long": "exit_bid / entry_ask - 1",
            "short": "1 - exit_ask / entry_bid",
        }
    ):
        raise ValueError("prospective label contract values are incompatible")


def load_campaign2_engineering_config(path: Path) -> Campaign2EngineeringConfig:
    """Load the exact engineering contract and reject every unsafe capability."""
    requested = path.expanduser().absolute()
    if requested.is_symlink():
        raise ValueError("Campaign 2 engineering config cannot be a symlink")
    resolved = requested.resolve()
    root = _project_root(resolved)
    values = _load_toml(resolved, "Campaign 2 engineering config")
    _exact_fields(
        values,
        {
            "format_version",
            "id",
            "status",
            "engineering_authorized_on",
            "protocol",
            "bar_config",
            "feature_config",
            "label_contract",
            "calendar_id",
            "timezone",
            "tzdata_version",
            "symbols",
            "horizons_minutes",
            "historical_fit_start",
            "historical_fit_end_exclusive",
            "forbidden_start",
            "forbidden_end_exclusive",
            "qualification_start",
            "context_start",
            "prospective_start",
            "decision_end_exclusive",
            "prospective_end_exclusive",
            "coverage_month_timezone",
            "minimum_complete_ratio",
            "maximum_consecutive_missing_bars",
            "maximum_feature_build_seconds_per_boundary",
            "maximum_peak_rss_bytes",
            "authorization",
        },
        "Campaign 2 engineering config",
    )
    authorization = values["authorization"]
    if not isinstance(authorization, dict):
        raise ValueError("authorization must be a table")
    expected_authorization = {
        "engineering": True,
        "fitting": False,
        "scoring": False,
        "collection": False,
        "evaluation": False,
        "raw_prospective_access": False,
    }
    if any(type(value) is not bool for value in authorization.values()) or (
        authorization != expected_authorization
    ):
        raise ValueError("Campaign 2 authorization must remain engineering-only")

    protocol = _resolve_reference(root, resolved.parent, values["protocol"], "protocol")
    bar_config = _resolve_reference(
        root, resolved.parent, values["bar_config"], "bar config"
    )
    feature_config = _resolve_reference(
        root, resolved.parent, values["feature_config"], "feature config"
    )
    label_contract = _resolve_reference(
        root, resolved.parent, values["label_contract"], "label contract"
    )
    if protocol != root / "docs/research/campaign-2-prospective-factor-plan.md":
        raise ValueError("Campaign 2 protocol reference is incompatible")
    if not protocol.read_text(encoding="utf-8").startswith(
        "# Research Campaign 2: Prospective Cross-Pair Factors\n"
    ):
        raise ValueError("Campaign 2 protocol identity is incompatible")
    _validate_bar_contract(bar_config)
    _validate_feature_contract(feature_config)
    _validate_label_contract(label_contract)

    timestamps = {
        key: _parse_utc(values[key], key)
        for key in (
            "historical_fit_start",
            "historical_fit_end_exclusive",
            "forbidden_start",
            "forbidden_end_exclusive",
            "qualification_start",
            "context_start",
            "prospective_start",
            "decision_end_exclusive",
            "prospective_end_exclusive",
        )
    }
    if (
        _integer(values, "format_version") != 1
        or values["id"] != CAMPAIGN_ID
        or values["status"] != ENGINEERING_STATUS
        or _string(values, "engineering_authorized_on")
        != ENGINEERING_AUTHORIZED_ON.isoformat()
        or values["calendar_id"] != CALENDAR_ID
        or values["timezone"] != TIMEZONE_NAME
        or values["tzdata_version"] != TZDATA_VERSION
        or _string_tuple(values, "symbols") != PAIRS
        or _integer_tuple(values, "horizons_minutes") != HORIZONS_MINUTES
        or values["coverage_month_timezone"] != "UTC"
        or not isinstance(values["minimum_complete_ratio"], float)
        or values["minimum_complete_ratio"] != MINIMUM_COMPLETE_RATIO
        or _integer(values, "maximum_consecutive_missing_bars")
        != MAXIMUM_CONSECUTIVE_MISSING_BARS
        or not isinstance(
            values["maximum_feature_build_seconds_per_boundary"], float
        )
        or values["maximum_feature_build_seconds_per_boundary"]
        != MAXIMUM_FEATURE_BUILD_SECONDS_PER_BOUNDARY
        or _integer(values, "maximum_peak_rss_bytes") != MAXIMUM_PEAK_RSS_BYTES
    ):
        raise ValueError("Campaign 2 engineering values are incompatible")
    if not (
        timestamps["historical_fit_start"]
        < timestamps["historical_fit_end_exclusive"]
        == timestamps["forbidden_start"]
        < timestamps["forbidden_end_exclusive"]
        == timestamps["qualification_start"]
        < timestamps["context_start"]
        < timestamps["prospective_start"]
        < timestamps["decision_end_exclusive"]
        < timestamps["prospective_end_exclusive"]
    ):
        raise ValueError("Campaign 2 temporal boundaries are incompatible")
    if timestamps != _EXPECTED_TIMESTAMPS:
        raise ValueError("Campaign 2 temporal boundaries differ from the protocol")
    if timestamps["context_start"] != timestamps["prospective_start"] - timedelta(
        hours=6
    ):
        raise ValueError("Campaign 2 context must be exactly six hours")
    if timestamps["decision_end_exclusive"] != timestamps[
        "prospective_end_exclusive"
    ] - timedelta(minutes=65):
        raise ValueError("decision interval must preserve the 65-minute window")

    return Campaign2EngineeringConfig(
        resolved,
        root,
        protocol,
        bar_config,
        feature_config,
        label_contract,
        timestamps["historical_fit_start"],
        timestamps["historical_fit_end_exclusive"],
        timestamps["forbidden_start"],
        timestamps["forbidden_end_exclusive"],
        timestamps["qualification_start"],
        timestamps["context_start"],
        timestamps["prospective_start"],
        timestamps["decision_end_exclusive"],
        timestamps["prospective_end_exclusive"],
        float(values["minimum_complete_ratio"]),
        int(values["maximum_consecutive_missing_bars"]),
        float(values["maximum_feature_build_seconds_per_boundary"]),
        int(values["maximum_peak_rss_bytes"]),
    )

"""Append-only custody records without raw-data or scoring capabilities."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

from demofml.data.prospective_ticks import (
    PROSPECTIVE_TICK_SCHEMA,
    ProspectiveTickQualityReport,
)
from demofml.features.cross_pair import PAIRS
from demofml.prospective.campaigns import CAMPAIGN_V1, CampaignSpec, campaign_spec
from demofml.prospective.records import (
    CONTENT_ID_PATTERN,
    SHA256_PATTERN,
    content_id,
)

COLLECTION_MANIFEST_SET_ID = CAMPAIGN_V1.collection_manifest_set_id
COLLECTION_TERMINAL_SET_ID = CAMPAIGN_V1.collection_terminal_set_id
_UINT64_MAX = 2**64 - 1
PROSPECTIVE_TICK_SCHEMA_SHA256 = hashlib.sha256(
    PROSPECTIVE_TICK_SCHEMA.serialize().to_pybytes()
).hexdigest()
_SEGMENT_AUTHORIZATION = "engineering_record_only_collection_not_authorized"
_QUALITY_FIELDS = (
    "rows",
    "null_values",
    "non_finite_values",
    "non_positive_bid",
    "non_positive_ask",
    "crossed_quotes",
    "inconsistent_mid",
    "inconsistent_spread",
    "provider_out_of_order",
    "receipt_out_of_order",
    "sequence_not_increasing",
    "clock_lead_violations",
    "mixed_symbols",
    "max_delivery_latency_ns",
    "critical_violations",
    "invalidated_boundaries",
)


@dataclass(frozen=True)
class CollectionObjectClaim:
    """Metadata claim for one externally immutable raw tick object."""

    object_key: str
    object_version_id: str
    size_bytes: int
    sha256: str
    rows: int
    first_ingest_sequence: int
    last_ingest_sequence: int
    provider_start: datetime
    provider_end_exclusive: datetime
    received_start: datetime
    received_end_exclusive: datetime
    object_last_modified: datetime


@dataclass(frozen=True)
class SegmentQuality:
    """Outcome-free quality counters bound into a collection segment."""

    rows: int
    null_values: int
    non_finite_values: int
    non_positive_bid: int
    non_positive_ask: int
    crossed_quotes: int
    inconsistent_mid: int
    inconsistent_spread: int
    provider_out_of_order: int
    receipt_out_of_order: int
    sequence_not_increasing: int
    clock_lead_violations: int
    mixed_symbols: int
    max_delivery_latency_ns: int
    invalidated_boundaries: tuple[datetime, ...] = ()

    @classmethod
    def from_report(
        cls,
        report: ProspectiveTickQualityReport,
        invalidated_boundaries: tuple[datetime, ...] = (),
    ) -> "SegmentQuality":
        """Capture public counters without private streaming state."""
        return cls(
            report.rows,
            report.null_values,
            report.non_finite_values,
            report.non_positive_bid,
            report.non_positive_ask,
            report.crossed_quotes,
            report.inconsistent_mid,
            report.inconsistent_spread,
            report.provider_out_of_order,
            report.receipt_out_of_order,
            report.sequence_not_increasing,
            report.clock_lead_violations,
            report.mixed_symbols,
            report.max_delivery_latency_ns,
            invalidated_boundaries,
        )

    @property
    def critical_violations(self) -> int:
        return (
            self.null_values
            + self.non_finite_values
            + self.non_positive_bid
            + self.non_positive_ask
            + self.crossed_quotes
            + self.inconsistent_mid
            + self.inconsistent_spread
            + self.provider_out_of_order
            + self.receipt_out_of_order
            + self.sequence_not_increasing
            + self.clock_lead_violations
            + self.mixed_symbols
            + len(self.invalidated_boundaries)
        )

    def as_record(self) -> dict[str, object]:
        """Return the exact quality payload used by the chain hash."""
        return {
            "rows": self.rows,
            "null_values": self.null_values,
            "non_finite_values": self.non_finite_values,
            "non_positive_bid": self.non_positive_bid,
            "non_positive_ask": self.non_positive_ask,
            "crossed_quotes": self.crossed_quotes,
            "inconsistent_mid": self.inconsistent_mid,
            "inconsistent_spread": self.inconsistent_spread,
            "provider_out_of_order": self.provider_out_of_order,
            "receipt_out_of_order": self.receipt_out_of_order,
            "sequence_not_increasing": self.sequence_not_increasing,
            "clock_lead_violations": self.clock_lead_violations,
            "mixed_symbols": self.mixed_symbols,
            "max_delivery_latency_ns": self.max_delivery_latency_ns,
            "critical_violations": self.critical_violations,
            "invalidated_boundaries": [
                _format_utc(value) for value in self.invalidated_boundaries
            ],
        }


@dataclass(frozen=True)
class CollectionChainSummary:
    """Verified terminal state for one symbol's ordered segment chain."""

    symbol: str
    segments: int
    rows: int
    critical_violations: int
    terminal_segment_id: str

    @property
    def qualified(self) -> bool:
        return self.critical_violations == 0


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("custody timestamps must be timezone-aware UTC")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} is not a valid UTC timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be UTC")
    if _format_utc(parsed) != value:
        raise ValueError(f"{name} is not canonically encoded")
    return parsed.astimezone(UTC)


def _exact_int(
    value: object,
    name: str,
    *,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        suffix = "" if maximum is None else f" and <= {maximum}"
        raise ValueError(f"{name} must be an integer >= {minimum}{suffix}")
    return value


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _content_id(value: object, name: str) -> str:
    if not isinstance(value, str) or CONTENT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a content ID")
    return value


def _validate_object_key(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or "//" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("object_key must be a safe relative object key")


def build_collection_segment(
    *,
    campaign: CampaignSpec,
    sequence: int,
    previous_segment_id: str | None,
    symbol: str,
    claim: CollectionObjectClaim,
    quality: SegmentQuality,
    collector_attestation_id: str,
    collector_attestation_sha256: str,
    recorded_at: datetime,
) -> dict[str, object]:
    """Build a content-addressed metadata record without reading the raw object."""
    campaign.require_artifact_creation()
    if symbol not in PAIRS:
        raise ValueError(f"unsupported Campaign 2 symbol {symbol}")
    _validate_object_key(claim.object_key)
    if not claim.object_version_id or claim.object_version_id == "null":
        raise ValueError("object_version_id cannot be empty or null")
    _digest(claim.sha256, "object sha256")
    _content_id(collector_attestation_id, "collector_attestation_id")
    _digest(collector_attestation_sha256, "collector_attestation_sha256")
    _exact_int(sequence, "sequence")
    _exact_int(claim.size_bytes, "size_bytes", minimum=1)
    _exact_int(claim.rows, "rows", minimum=1)
    _exact_int(
        claim.first_ingest_sequence,
        "first_ingest_sequence",
        maximum=_UINT64_MAX,
    )
    _exact_int(
        claim.last_ingest_sequence,
        "last_ingest_sequence",
        maximum=_UINT64_MAX,
    )
    if claim.last_ingest_sequence < claim.first_ingest_sequence:
        raise ValueError("object ingest sequence range is reversed")
    if claim.rows > claim.last_ingest_sequence - claim.first_ingest_sequence + 1:
        raise ValueError("object rows exceed the strictly increasing sequence range")
    if quality.rows != claim.rows:
        raise ValueError("quality rows must equal object rows")
    quality_record = quality.as_record()
    _validate_quality(quality_record)
    provider_start = _format_utc(claim.provider_start)
    provider_end = _format_utc(claim.provider_end_exclusive)
    received_start = _format_utc(claim.received_start)
    received_end = _format_utc(claim.received_end_exclusive)
    last_modified = _format_utc(claim.object_last_modified)
    recorded = _format_utc(recorded_at)
    if not (
        claim.provider_start < claim.provider_end_exclusive
        and claim.received_start < claim.received_end_exclusive
        and claim.received_end_exclusive <= claim.object_last_modified
        and claim.object_last_modified <= recorded_at
        and recorded_at >= claim.received_end_exclusive
    ):
        raise ValueError("collection object time ranges are incompatible")
    if not (
        campaign.qualification_start <= claim.provider_start
        and claim.provider_end_exclusive <= campaign.prospective_end_exclusive
        and campaign.qualification_start <= claim.received_start
        and claim.received_end_exclusive <= campaign.prospective_end_exclusive
    ):
        raise ValueError("collection object falls outside the campaign interval")
    if any(
        not claim.provider_start < boundary <= claim.provider_end_exclusive
        for boundary in quality.invalidated_boundaries
    ):
        raise ValueError("invalidated boundary falls outside the provider interval")
    if sequence == 0:
        if previous_segment_id is not None:
            raise ValueError("first segment cannot have a previous segment")
    else:
        _content_id(previous_segment_id, "previous_segment_id")
    core: dict[str, object] = {
        "format_version": 1,
        "manifest_set": campaign.collection_manifest_set_id,
        "campaign_id": campaign.campaign_id,
        "authorization": _SEGMENT_AUTHORIZATION,
        "scoring_authorized": False,
        "sequence": sequence,
        "previous_segment_id": previous_segment_id,
        "symbol": symbol,
        "object": {
            "key": claim.object_key,
            "version_id": claim.object_version_id,
            "size_bytes": claim.size_bytes,
            "sha256": claim.sha256,
            "rows": claim.rows,
            "first_ingest_sequence": claim.first_ingest_sequence,
            "last_ingest_sequence": claim.last_ingest_sequence,
            "provider_start": provider_start,
            "provider_end_exclusive": provider_end,
            "received_start": received_start,
            "received_end_exclusive": received_end,
            "last_modified": last_modified,
        },
        "tick_schema_sha256": PROSPECTIVE_TICK_SCHEMA_SHA256,
        "collector_attestation": {
            "id": collector_attestation_id,
            "sha256": collector_attestation_sha256,
        },
        "quality": quality_record,
        "recorded_at": recorded,
    }
    return {**core, "segment_id": content_id(core)}


def _validate_quality(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != set(_QUALITY_FIELDS):
        raise ValueError("segment quality fields are incompatible")
    counters = {
        key: _exact_int(value[key], f"quality.{key}")
        for key in _QUALITY_FIELDS
        if key not in {"invalidated_boundaries"}
    }
    invalidated = value["invalidated_boundaries"]
    if not isinstance(invalidated, list):
        raise ValueError("invalidated_boundaries must be a list")
    expected_critical = sum(
        counters[key]
        for key in (
            "null_values",
            "non_finite_values",
            "non_positive_bid",
            "non_positive_ask",
            "crossed_quotes",
            "inconsistent_mid",
            "inconsistent_spread",
            "provider_out_of_order",
            "receipt_out_of_order",
            "sequence_not_increasing",
            "clock_lead_violations",
            "mixed_symbols",
        )
    ) + len(invalidated)
    if counters["critical_violations"] != expected_critical:
        raise ValueError("quality critical_violations does not reconcile")
    parsed = [_parse_utc(item, "invalidated boundary") for item in invalidated]
    if parsed != sorted(set(parsed)):
        raise ValueError("invalidated boundaries must be unique and ordered")
    if counters["rows"] <= 0:
        raise ValueError("quality rows must be positive")
    return value


def validate_collection_segment(segment: Mapping[str, object]) -> None:
    """Validate exact fields and recompute one segment's content identity."""
    expected = {
        "format_version",
        "manifest_set",
        "campaign_id",
        "authorization",
        "scoring_authorized",
        "sequence",
        "previous_segment_id",
        "symbol",
        "object",
        "tick_schema_sha256",
        "collector_attestation",
        "quality",
        "recorded_at",
        "segment_id",
    }
    if set(segment) != expected:
        raise ValueError("collection segment fields are incompatible")
    campaign = campaign_spec(segment["campaign_id"])
    if (
        segment["format_version"] != 1
        or segment["manifest_set"] != campaign.collection_manifest_set_id
        or segment["authorization"] != _SEGMENT_AUTHORIZATION
        or segment["scoring_authorized"] is not False
        or segment["symbol"] not in PAIRS
        or segment["tick_schema_sha256"] != PROSPECTIVE_TICK_SCHEMA_SHA256
    ):
        raise ValueError("collection segment identity is incompatible")
    sequence = _exact_int(segment["sequence"], "sequence")
    previous = segment["previous_segment_id"]
    if sequence == 0:
        if previous is not None:
            raise ValueError("first segment cannot bind a predecessor")
    else:
        _content_id(previous, "previous_segment_id")
    _content_id(segment["segment_id"], "segment_id")
    object_record = segment["object"]
    if not isinstance(object_record, dict) or set(object_record) != {
        "key",
        "version_id",
        "size_bytes",
        "sha256",
        "rows",
        "first_ingest_sequence",
        "last_ingest_sequence",
        "provider_start",
        "provider_end_exclusive",
        "received_start",
        "received_end_exclusive",
        "last_modified",
    }:
        raise ValueError("collection object fields are incompatible")
    key = object_record["key"]
    version = object_record["version_id"]
    if (
        not isinstance(key, str)
        or not isinstance(version, str)
        or not version
        or version == "null"
    ):
        raise ValueError("collection object key and version must be strings")
    _validate_object_key(key)
    _exact_int(object_record["size_bytes"], "size_bytes", minimum=1)
    rows = _exact_int(object_record["rows"], "rows", minimum=1)
    _digest(object_record["sha256"], "object sha256")
    first_sequence = _exact_int(
        object_record["first_ingest_sequence"],
        "first_ingest_sequence",
        maximum=_UINT64_MAX,
    )
    last_sequence = _exact_int(
        object_record["last_ingest_sequence"],
        "last_ingest_sequence",
        maximum=_UINT64_MAX,
    )
    if last_sequence < first_sequence:
        raise ValueError("object ingest sequence range is reversed")
    if rows > last_sequence - first_sequence + 1:
        raise ValueError("object rows exceed the strictly increasing sequence range")
    provider_start = _parse_utc(object_record["provider_start"], "provider_start")
    provider_end = _parse_utc(
        object_record["provider_end_exclusive"], "provider_end_exclusive"
    )
    received_start = _parse_utc(object_record["received_start"], "received_start")
    received_end = _parse_utc(
        object_record["received_end_exclusive"], "received_end_exclusive"
    )
    last_modified = _parse_utc(object_record["last_modified"], "last_modified")
    recorded_at = _parse_utc(segment["recorded_at"], "recorded_at")
    if not (
        provider_start < provider_end
        and received_start < received_end
        and received_end <= last_modified <= recorded_at
        and recorded_at >= received_end
    ):
        raise ValueError("collection object time ranges are incompatible")
    if not (
        campaign.qualification_start <= provider_start
        and provider_end <= campaign.prospective_end_exclusive
        and campaign.qualification_start <= received_start
        and received_end <= campaign.prospective_end_exclusive
    ):
        raise ValueError("collection object falls outside the campaign interval")
    attestation = segment["collector_attestation"]
    if not isinstance(attestation, dict) or set(attestation) != {"id", "sha256"}:
        raise ValueError("collector attestation fields are incompatible")
    _content_id(attestation["id"], "collector attestation ID")
    _digest(attestation["sha256"], "collector attestation sha256")
    quality = _validate_quality(segment["quality"])
    if quality["rows"] != rows:
        raise ValueError("quality rows do not reconcile with object rows")
    invalidated = quality["invalidated_boundaries"]
    if not isinstance(invalidated, list):
        raise AssertionError("validated invalidated_boundaries must be a list")
    if any(
        not provider_start < _parse_utc(value, "invalidated boundary") <= provider_end
        for value in invalidated
    ):
        raise ValueError("invalidated boundary falls outside the provider interval")
    core = {key: value for key, value in segment.items() if key != "segment_id"}
    if segment["segment_id"] != content_id(core):
        raise ValueError("collection segment content ID mismatch")


def validate_collection_chain(
    segments: Sequence[Mapping[str, object]],
    *,
    expected_symbol: str | None = None,
    expected_campaign: CampaignSpec | None = None,
) -> CollectionChainSummary:
    """Verify predecessor hashes and monotone ranges for one symbol chain."""
    if not segments:
        raise ValueError("collection chain cannot be empty")
    previous: Mapping[str, object] | None = None
    seen_objects: set[object] = set()
    total_rows = 0
    total_critical = 0
    symbol: str | None = None
    attestation_id: object | None = None
    campaign: CampaignSpec | None = expected_campaign
    for index, segment in enumerate(segments):
        validate_collection_segment(segment)
        current_campaign = campaign_spec(segment["campaign_id"])
        if campaign is None:
            campaign = current_campaign
        elif current_campaign != campaign:
            raise ValueError("collection campaign changed within a chain")
        current_symbol = str(segment["symbol"])
        if symbol is None:
            symbol = current_symbol
            attestation = segment["collector_attestation"]
            if isinstance(attestation, dict):
                attestation_id = attestation["id"]
        if current_symbol != symbol or segment["sequence"] != index:
            raise ValueError("collection chain symbol or sequence changed")
        current_attestation = segment["collector_attestation"]
        if (
            not isinstance(current_attestation, dict)
            or current_attestation["id"] != attestation_id
        ):
            raise ValueError("collector attestation changed within a chain")
        if previous is None:
            if segment["previous_segment_id"] is not None:
                raise ValueError("first chain segment binds a predecessor")
        else:
            if segment["previous_segment_id"] != previous["segment_id"]:
                raise ValueError("collection predecessor hash mismatch")
            _validate_monotone_segments(previous, segment)
        object_record = segment["object"]
        quality = segment["quality"]
        if not isinstance(object_record, dict) or not isinstance(quality, dict):
            raise AssertionError("validated records must be objects")
        object_identity = object_record["key"]
        if object_identity in seen_objects:
            raise ValueError("collection chain reuses an immutable object key")
        seen_objects.add(object_identity)
        total_rows += int(object_record["rows"])
        total_critical += int(quality["critical_violations"])
        previous = segment
    if symbol is None or previous is None:
        raise AssertionError("non-empty chain must have a terminal segment")
    if expected_symbol is not None and symbol != expected_symbol:
        raise ValueError("collection chain does not match expected symbol")
    return CollectionChainSummary(
        symbol,
        len(segments),
        total_rows,
        total_critical,
        str(previous["segment_id"]),
    )


def _validate_monotone_segments(
    previous: Mapping[str, object], current: Mapping[str, object]
) -> None:
    previous_object = previous["object"]
    current_object = current["object"]
    if not isinstance(previous_object, dict) or not isinstance(current_object, dict):
        raise AssertionError("validated collection objects must be mappings")
    if int(current_object["first_ingest_sequence"]) <= int(
        previous_object["last_ingest_sequence"]
    ):
        raise ValueError("collection ingest sequence ranges overlap")
    for previous_key, current_key in (
        ("provider_end_exclusive", "provider_start"),
        ("received_end_exclusive", "received_start"),
    ):
        if _parse_utc(current_object[current_key], current_key) < _parse_utc(
            previous_object[previous_key], previous_key
        ):
            raise ValueError("collection timestamp ranges overlap")
    if _parse_utc(current["recorded_at"], "recorded_at") < _parse_utc(
        previous["recorded_at"], "recorded_at"
    ):
        raise ValueError("collection record time regressed")


def build_collection_terminal(
    chains: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    campaign: CampaignSpec,
    recorded_at: datetime,
) -> dict[str, object]:
    """Bind all eight verified symbol chains into one outcome-free terminal."""
    campaign.require_artifact_creation()
    return _build_collection_terminal(
        chains,
        campaign=campaign,
        recorded_at=recorded_at,
    )


def _build_collection_terminal(
    chains: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    campaign: CampaignSpec,
    recorded_at: datetime,
) -> dict[str, object]:
    if set(chains) != set(PAIRS):
        raise ValueError("terminal manifest requires all eight symbol chains")
    object_identities: set[object] = set()
    for chain in chains.values():
        for segment in chain:
            object_record = segment.get("object")
            if not isinstance(object_record, dict):
                raise ValueError("collection segment object must be a mapping")
            identity = object_record.get("key")
            if identity in object_identities:
                raise ValueError("terminal reuses an object key across symbols")
            object_identities.add(identity)
    summaries = [
        validate_collection_chain(
            chains[symbol],
            expected_symbol=symbol,
            expected_campaign=campaign,
        )
        for symbol in PAIRS
    ]
    latest_recorded = max(
        _parse_utc(chain[-1]["recorded_at"], "recorded_at")
        for chain in chains.values()
    )
    if recorded_at < latest_recorded:
        raise ValueError("terminal record cannot predate a symbol chain")
    core: dict[str, object] = {
        "format_version": 1,
        "terminal_set": campaign.collection_terminal_set_id,
        "campaign_id": campaign.campaign_id,
        "authorization": _SEGMENT_AUTHORIZATION,
        "scoring_authorized": False,
        "chains": [
            {
                "symbol": summary.symbol,
                "segments": summary.segments,
                "rows": summary.rows,
                "critical_violations": summary.critical_violations,
                "terminal_segment_id": summary.terminal_segment_id,
                "qualified": summary.qualified,
            }
            for summary in summaries
        ],
        "all_chains_qualified": all(summary.qualified for summary in summaries),
        "recorded_at": _format_utc(recorded_at),
    }
    return {**core, "terminal_id": content_id(core)}


def validate_collection_terminal(
    terminal: Mapping[str, object],
    chains: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    """Rebuild and exactly reconcile a terminal against all bound chains."""
    if set(terminal) != {
        "format_version",
        "terminal_set",
        "campaign_id",
        "authorization",
        "scoring_authorized",
        "chains",
        "all_chains_qualified",
        "recorded_at",
        "terminal_id",
    }:
        raise ValueError("collection terminal fields are incompatible")
    campaign = campaign_spec(terminal["campaign_id"])
    if terminal["terminal_set"] != campaign.collection_terminal_set_id:
        raise ValueError("collection terminal identity is incompatible")
    recorded_at = _parse_utc(terminal["recorded_at"], "recorded_at")
    expected = _build_collection_terminal(
        chains,
        campaign=campaign,
        recorded_at=recorded_at,
    )
    if dict(terminal) != expected:
        raise ValueError("collection terminal does not reconcile with its chains")

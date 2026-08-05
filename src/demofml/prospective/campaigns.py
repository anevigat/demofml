"""Immutable identities and temporal boundaries for Campaign 2 attempts."""

from dataclasses import dataclass
from datetime import UTC, date, datetime
from types import MappingProxyType
from typing import Final


@dataclass(frozen=True, slots=True)
class CampaignSpec:
    """Frozen identity family for one prospectively declared campaign attempt."""

    campaign_id: str
    engineering_authorized_on: date
    engineering_config_relative_path: str
    protocol_relative_path: str
    protocol_heading: str
    historical_fit_start: datetime
    historical_fit_end_exclusive: datetime
    forbidden_start: datetime
    forbidden_end_exclusive: datetime
    qualification_start: datetime
    context_start: datetime
    prospective_start: datetime
    decision_end_exclusive: datetime
    prospective_end_exclusive: datetime
    opportunity_ledger_id: str
    collection_manifest_set_id: str
    collection_terminal_set_id: str
    qualification_set_id: str
    engineering_bundle_set_id: str
    engineering_verify_set_id: str
    artifact_creation_open: bool

    @property
    def timestamps(self) -> dict[str, datetime]:
        """Return the exact timestamp fields expected in the engineering config."""
        return {
            "historical_fit_start": self.historical_fit_start,
            "historical_fit_end_exclusive": self.historical_fit_end_exclusive,
            "forbidden_start": self.forbidden_start,
            "forbidden_end_exclusive": self.forbidden_end_exclusive,
            "qualification_start": self.qualification_start,
            "context_start": self.context_start,
            "prospective_start": self.prospective_start,
            "decision_end_exclusive": self.decision_end_exclusive,
            "prospective_end_exclusive": self.prospective_end_exclusive,
        }

    def require_artifact_creation(self) -> None:
        """Reject new artifacts for attempts that have already closed."""
        if CAMPAIGN_SPECS.get(self.campaign_id) is not self:
            raise ValueError("artifact creation requires a canonical campaign spec")
        if not self.artifact_creation_open:
            raise ValueError(
                f"artifact creation is closed for Campaign 2 attempt {self.campaign_id}"
            )


CAMPAIGN_V1: Final = CampaignSpec(
    campaign_id="prospective-cross-pair-factor-v1",
    engineering_authorized_on=date(2026, 8, 4),
    engineering_config_relative_path=(
        "configs/prospective/campaign-2-engineering-v1.toml"
    ),
    protocol_relative_path="docs/research/campaign-2-prospective-factor-plan.md",
    protocol_heading="# Research Campaign 2: Prospective Cross-Pair Factors\n",
    historical_fit_start=datetime(2018, 1, 1, tzinfo=UTC),
    historical_fit_end_exclusive=datetime(2025, 1, 1, tzinfo=UTC),
    forbidden_start=datetime(2025, 1, 1, tzinfo=UTC),
    forbidden_end_exclusive=datetime(2026, 3, 11, tzinfo=UTC),
    qualification_start=datetime(2026, 3, 11, tzinfo=UTC),
    context_start=datetime(2026, 8, 31, 18, tzinfo=UTC),
    prospective_start=datetime(2026, 9, 1, tzinfo=UTC),
    decision_end_exclusive=datetime(2027, 8, 31, 22, 55, tzinfo=UTC),
    prospective_end_exclusive=datetime(2027, 9, 1, tzinfo=UTC),
    opportunity_ledger_id="prospective-opportunities-v1",
    collection_manifest_set_id="prospective-collection-segment-v1",
    collection_terminal_set_id="prospective-collection-terminal-v1",
    qualification_set_id="campaign-2-engineering-qualification-v1",
    engineering_bundle_set_id="campaign-2-engineering-bundle-v1",
    engineering_verify_set_id="campaign-2-onprem-engineering-verify-v1",
    artifact_creation_open=False,
)

CAMPAIGN_V2: Final = CampaignSpec(
    campaign_id="prospective-cross-pair-factor-v2",
    engineering_authorized_on=date(2026, 8, 5),
    engineering_config_relative_path=(
        "configs/prospective/campaign-2-engineering-v2.toml"
    ),
    protocol_relative_path="docs/research/campaign-2-prospective-factor-v2.md",
    protocol_heading="# Research Campaign 2 v2: Prospective Cross-Pair Factors\n",
    historical_fit_start=datetime(2018, 1, 1, tzinfo=UTC),
    historical_fit_end_exclusive=datetime(2025, 1, 1, tzinfo=UTC),
    forbidden_start=datetime(2025, 1, 1, tzinfo=UTC),
    forbidden_end_exclusive=datetime(2026, 3, 11, tzinfo=UTC),
    qualification_start=datetime(2026, 9, 1, tzinfo=UTC),
    context_start=datetime(2027, 2, 28, 18, tzinfo=UTC),
    prospective_start=datetime(2027, 3, 1, tzinfo=UTC),
    decision_end_exclusive=datetime(2028, 2, 29, 22, 55, tzinfo=UTC),
    prospective_end_exclusive=datetime(2028, 3, 1, tzinfo=UTC),
    opportunity_ledger_id="prospective-opportunities-v2",
    collection_manifest_set_id="prospective-collection-segment-v2",
    collection_terminal_set_id="prospective-collection-terminal-v2",
    qualification_set_id="campaign-2-engineering-qualification-v2",
    engineering_bundle_set_id="campaign-2-engineering-bundle-v2",
    engineering_verify_set_id="campaign-2-onprem-engineering-verify-v2",
    artifact_creation_open=True,
)

CAMPAIGN_SPECS = MappingProxyType(
    {spec.campaign_id: spec for spec in (CAMPAIGN_V1, CAMPAIGN_V2)}
)


def campaign_spec(campaign_id: object) -> CampaignSpec:
    """Resolve an exact campaign identity and reject unknown attempts."""
    if not isinstance(campaign_id, str) or campaign_id not in CAMPAIGN_SPECS:
        raise ValueError(f"unsupported Campaign 2 identity: {campaign_id!r}")
    return CAMPAIGN_SPECS[campaign_id]


def campaign_spec_for(field: str, value: object) -> CampaignSpec:
    """Resolve a campaign from one versioned artifact-set identity."""
    if field not in CampaignSpec.__dataclass_fields__:
        raise ValueError(f"unsupported Campaign 2 identity field: {field}")
    matches = [
        spec for spec in CAMPAIGN_SPECS.values() if getattr(spec, field) == value
    ]
    if len(matches) != 1:
        raise ValueError(f"unsupported Campaign 2 {field}: {value!r}")
    return matches[0]

"""Outcome-free expected-opportunity ledger for Campaign 2 qualification."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import product
from typing import TYPE_CHECKING

import pyarrow as pa  # type: ignore[import-untyped]

from demofml.calendars.prospective_fx import (
    CALENDAR_ID,
    INTERVAL,
    expected_decision_boundaries,
)
from demofml.features.cross_pair import (
    CANDIDATE_FEATURE_SCHEMA,
    CONTROL_FEATURE_SCHEMA,
    CROSS_PAIR_FEATURE_SET_ID,
    PAIRS,
    PairedFeatureBatch,
)

if TYPE_CHECKING:
    from demofml.prospective.config import Campaign2EngineeringConfig

CAMPAIGN_ID = "prospective-cross-pair-factor-v1"
OPPORTUNITY_LEDGER_ID = "prospective-opportunities-v1"
ARMS = ("control", "candidate")
HORIZONS_MINUTES = (15, 30, 60)
CONTROL_FEATURE_SET_ID = "causal-v1-prospective-control-v1"
OPPORTUNITY_STATUSES = (
    "ready",
    "missing_input",
    "factor_warmup",
    "late_feature",
)
OPPORTUNITY_SCHEMA = pa.schema(
    [
        pa.field("campaign_id", pa.string(), nullable=False),
        pa.field("ledger_id", pa.string(), nullable=False),
        pa.field("arm", pa.string(), nullable=False),
        pa.field("feature_set_id", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("horizon_minutes", pa.int16(), nullable=False),
        pa.field("decision_time", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field(
            "publication_deadline", pa.timestamp("ns", tz="UTC"), nullable=False
        ),
        pa.field("feature_available_at", pa.timestamp("ns", tz="UTC")),
        pa.field("status", pa.string(), nullable=False),
        pa.field("missing_reason", pa.string()),
    ],
    metadata={
        b"demofml.ledger": OPPORTUNITY_LEDGER_ID.encode(),
        b"demofml.calendar": CALENDAR_ID.encode(),
        b"demofml.authorization": b"engineering_only_no_scores_or_outcomes",
        b"demofml.expected_rows_per_boundary": b"48",
    },
)
_FORBIDDEN_COLUMN_PARTS = (
    "score",
    "action",
    "label",
    "return",
    "pnl",
    "profit",
    "drawdown",
    "entry_price",
    "exit_price",
    "outcome",
    "target",
)


def empty_opportunities() -> pa.Table:
    """Return an empty table with the qualification-safe ledger schema."""
    return pa.Table.from_batches([], schema=OPPORTUNITY_SCHEMA)


def assert_outcome_free_schema(schema: pa.Schema) -> None:
    """Reject any engineering artifact carrying scores or outcome information."""
    names: list[str] = []

    def collect(field: pa.Field) -> None:
        names.append(field.name)
        if pa.types.is_struct(field.type):
            for child in field.type:
                collect(child)

    for field in schema:
        collect(field)
    forbidden = [
        name
        for name in names
        if any(part in name.lower() for part in _FORBIDDEN_COLUMN_PARTS)
    ]
    if forbidden:
        raise ValueError(f"engineering ledger contains forbidden columns: {forbidden}")


def materialize_expected_opportunities(batch: PairedFeatureBatch) -> pa.Table:
    """Create both arms for every symbol/horizon without scores or outcomes."""
    _validate_feature_batch(batch)
    deadline = batch.decision_time + timedelta(minutes=5)
    if batch.missing_symbols:
        status = "missing_input"
        reason = "missing_symbols:" + ",".join(batch.missing_symbols)
    elif not batch.ready:
        status = "factor_warmup"
        reason = "cross_pair_return_unavailable"
    elif batch.feature_available_at is None:
        raise ValueError("ready feature batch lacks feature_available_at")
    elif batch.feature_available_at > deadline:
        status = "late_feature"
        reason = "feature_available_after_publication_deadline"
    else:
        status = "ready"
        reason = None

    feature_sets = {
        "control": CONTROL_FEATURE_SET_ID,
        "candidate": CROSS_PAIR_FEATURE_SET_ID,
    }
    rows = [
        {
            "campaign_id": CAMPAIGN_ID,
            "ledger_id": OPPORTUNITY_LEDGER_ID,
            "arm": arm,
            "feature_set_id": feature_sets[arm],
            "symbol": symbol,
            "horizon_minutes": horizon,
            "decision_time": batch.decision_time,
            "publication_deadline": deadline,
            "feature_available_at": batch.feature_available_at,
            "status": status,
            "missing_reason": reason,
        }
        for arm, symbol, horizon in product(ARMS, PAIRS, HORIZONS_MINUTES)
    ]
    return pa.Table.from_pylist(rows, schema=OPPORTUNITY_SCHEMA)


def _validate_feature_batch(batch: PairedFeatureBatch) -> None:
    if batch.control.schema != CONTROL_FEATURE_SCHEMA:
        raise ValueError("control feature schema mismatch")
    if batch.candidate.schema != CANDIDATE_FEATURE_SCHEMA:
        raise ValueError("candidate feature schema mismatch")
    canonical_missing = tuple(
        symbol for symbol in PAIRS if symbol in set(batch.missing_symbols)
    )
    if batch.missing_symbols and batch.missing_symbols != canonical_missing:
        raise ValueError("missing_symbols must be unique canonical Campaign 2 symbols")
    if batch.missing_symbols:
        if batch.control.num_rows or batch.candidate.num_rows:
            raise ValueError("missing feature batch must not carry partial rows")
        if batch.feature_available_at is not None or batch.ready:
            raise ValueError("missing feature batch cannot be ready or available")
        return
    if batch.control.num_rows != len(PAIRS) or batch.candidate.num_rows != len(PAIRS):
        raise ValueError("complete feature batch requires eight rows per arm")
    if batch.feature_available_at is None:
        raise ValueError("complete feature batch lacks feature availability")
    control = {str(row["symbol"]): row for row in batch.control.to_pylist()}
    candidate = {str(row["symbol"]): row for row in batch.candidate.to_pylist()}
    if set(control) != set(PAIRS) or set(candidate) != set(PAIRS):
        raise ValueError("feature batch symbols do not match Campaign 2")
    for symbol in PAIRS:
        for row in (control[symbol], candidate[symbol]):
            if row["decision_time"] != batch.decision_time:
                raise ValueError("feature decision time does not match batch")
            if row["feature_available_at"] != batch.feature_available_at:
                raise ValueError("feature availability does not match batch")
    if batch.ready:
        required_current = (
            "base_strength_1",
            "quote_strength_1",
            "pair_factor_residual_1",
            "cross_pair_residual_dispersion_1",
        )
        if any(
            candidate[symbol][name] is None
            for symbol in PAIRS
            for name in required_current
        ):
            raise ValueError("ready candidate lacks current cross-pair features")


def validate_opportunity_ledger(
    ledger: pa.Table,
    expected_boundaries: tuple[datetime, ...] | None = None,
) -> None:
    """Validate exact keys and explicit status for every expected opportunity."""
    if ledger.schema != OPPORTUNITY_SCHEMA:
        raise ValueError("opportunity ledger schema or metadata mismatch")
    assert_outcome_free_schema(ledger.schema)
    rows = ledger.to_pylist()
    keys: set[tuple[object, ...]] = set()
    by_boundary: dict[datetime, list[dict[str, object]]] = {}
    expected_cells = set(product(ARMS, PAIRS, HORIZONS_MINUTES))
    feature_sets = {
        "control": CONTROL_FEATURE_SET_ID,
        "candidate": CROSS_PAIR_FEATURE_SET_ID,
    }
    for row in rows:
        decision_time = row["decision_time"]
        if not isinstance(decision_time, datetime):
            raise ValueError("decision_time cannot be null")
        key = (
            row["arm"],
            row["symbol"],
            row["horizon_minutes"],
            decision_time,
        )
        if key in keys:
            raise ValueError("duplicate expected-opportunity key")
        keys.add(key)
        arm = str(row["arm"])
        if row["campaign_id"] != CAMPAIGN_ID:
            raise ValueError("unexpected campaign_id")
        if row["ledger_id"] != OPPORTUNITY_LEDGER_ID:
            raise ValueError("unexpected ledger_id")
        if arm not in feature_sets or row["feature_set_id"] != feature_sets[arm]:
            raise ValueError("arm and feature_set_id do not match")
        if row["publication_deadline"] != decision_time + timedelta(minutes=5):
            raise ValueError(
                "publication deadline must be decision_time plus five minutes"
            )
        status = str(row["status"])
        reason = row["missing_reason"]
        if status not in OPPORTUNITY_STATUSES:
            raise ValueError(f"unsupported opportunity status {status}")
        available = row["feature_available_at"]
        if status == "ready" and reason is not None:
            raise ValueError("ready rows must omit missing_reason")
        if status == "missing_input":
            if not isinstance(reason, str) or not reason.startswith("missing_symbols:"):
                raise ValueError("missing_input reason is incompatible")
            symbols = tuple(reason.removeprefix("missing_symbols:").split(","))
            canonical = tuple(symbol for symbol in PAIRS if symbol in set(symbols))
            if not symbols or symbols != canonical or available is not None:
                raise ValueError("missing_input payload is incompatible")
        elif status == "factor_warmup":
            if reason != "cross_pair_return_unavailable" or available is None:
                raise ValueError("factor_warmup payload is incompatible")
        elif status == "late_feature":
            if (
                reason != "feature_available_after_publication_deadline"
                or available is None
                or available <= row["publication_deadline"]
            ):
                raise ValueError("late_feature payload is incompatible")
        if status == "ready":
            if available is None:
                raise ValueError("ready feature lacks availability time")
            if available > row["publication_deadline"]:
                raise ValueError("ready feature misses publication deadline")
        by_boundary.setdefault(decision_time, []).append(row)

    expected = (
        tuple(by_boundary) if expected_boundaries is None else expected_boundaries
    )
    if set(by_boundary) != set(expected):
        raise ValueError("ledger boundaries do not match the expected calendar")
    for boundary in expected:
        boundary_rows = by_boundary[boundary]
        cells = {
            (row["arm"], row["symbol"], row["horizon_minutes"])
            for row in boundary_rows
        }
        if cells != expected_cells:
            raise ValueError(f"incomplete opportunity cells at {boundary.isoformat()}")
        statuses = {row["status"] for row in boundary_rows}
        if len(statuses) != 1:
            raise ValueError("paired arms must share one boundary status")
        if len({row["missing_reason"] for row in boundary_rows}) != 1:
            raise ValueError("paired arms must share one missing reason")
        if len({row["feature_available_at"] for row in boundary_rows}) != 1:
            raise ValueError("paired arms must share one feature availability time")


@dataclass(frozen=True)
class CoverageReport:
    """Qualification metrics derived only from expected and ready boundaries."""

    expected_sections: int
    complete_sections: int
    complete_ratio: float
    maximum_consecutive_missing: int
    monthly_complete_ratio: dict[str, float]

    def passes(self, config: "Campaign2EngineeringConfig") -> bool:
        """Apply the frozen global, monthly, and consecutive-missing gates."""
        return (
            self.complete_ratio >= config.minimum_complete_ratio
            and all(
                value >= config.minimum_complete_ratio
                for value in self.monthly_complete_ratio.values()
            )
            and self.maximum_consecutive_missing
            <= config.maximum_consecutive_missing_bars
        )


def summarize_coverage(
    ledger: pa.Table,
    expected_boundaries: tuple[datetime, ...],
) -> CoverageReport:
    """Compute missingness-only qualification metrics in UTC calendar months."""
    validate_opportunity_ledger(ledger)
    if any(
        right <= left
        for left, right in zip(
            expected_boundaries, expected_boundaries[1:], strict=False
        )
    ):
        raise ValueError("expected boundaries must be strictly increasing")
    if expected_boundaries:
        regenerated = expected_decision_boundaries(
            expected_boundaries[0], expected_boundaries[-1] + INTERVAL
        )
        if regenerated != expected_boundaries:
            raise ValueError("expected boundaries do not match the frozen calendar")
    represented: dict[datetime, str] = {}
    for row in ledger.select(["decision_time", "status"]).to_pylist():
        decision_time = row["decision_time"]
        if isinstance(decision_time, datetime):
            represented[decision_time] = str(row["status"])
    unexpected = set(represented).difference(expected_boundaries)
    if unexpected:
        raise ValueError("ledger contains boundaries outside the expected calendar")
    ordered = [
        (boundary, represented.get(boundary, "absent"))
        for boundary in expected_boundaries
    ]
    expected = len(ordered)
    complete = sum(status == "ready" for _, status in ordered)
    longest = 0
    current = 0
    monthly: dict[str, list[bool]] = {}
    for decision_time, status in ordered:
        ready = status == "ready"
        if ready:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
        month = decision_time.strftime("%Y-%m")
        monthly.setdefault(month, []).append(ready)
    monthly_ratios = {
        month: sum(values) / len(values) for month, values in monthly.items()
    }
    return CoverageReport(
        expected,
        complete,
        complete / expected if expected else 0.0,
        longest,
        monthly_ratios,
    )

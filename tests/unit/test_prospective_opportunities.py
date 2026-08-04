import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

from demofml.features.cross_pair import (
    CANDIDATE_FEATURE_SCHEMA,
    CONTROL_FEATURE_SCHEMA,
    PairedFeatureBatch,
)
from demofml.prospective.config import load_campaign2_engineering_config
from demofml.prospective.opportunities import (
    assert_outcome_free_schema,
    materialize_expected_opportunities,
    summarize_coverage,
    validate_opportunity_ledger,
)

PROJECT_ROOT = Path(__file__).parents[2]


def _batch(
    decision_time: datetime,
    *,
    ready: bool,
    missing_symbols: tuple[str, ...] = (),
    available_delay: timedelta = timedelta(seconds=2),
) -> PairedFeatureBatch:
    available = None if missing_symbols else decision_time + available_delay
    if missing_symbols:
        control = pa.Table.from_batches([], schema=CONTROL_FEATURE_SCHEMA)
        candidate = pa.Table.from_batches([], schema=CANDIDATE_FEATURE_SCHEMA)
    else:
        control = _feature_table(
            CONTROL_FEATURE_SCHEMA, decision_time, available, ready=ready
        )
        candidate = _feature_table(
            CANDIDATE_FEATURE_SCHEMA, decision_time, available, ready=ready
        )
    return PairedFeatureBatch(
        decision_time,
        control,
        candidate,
        missing_symbols,
        available,
        ready,
    )


def _feature_table(
    schema: pa.Schema,
    decision_time: datetime,
    available: datetime | None,
    *,
    ready: bool,
) -> pa.Table:
    current_cross = {
        "base_strength_1",
        "quote_strength_1",
        "pair_factor_residual_1",
        "cross_pair_residual_dispersion_1",
    }
    rows: list[dict[str, object]] = []
    for symbol in (
        "AUDUSD",
        "EURCHF",
        "EURJPY",
        "EURUSD",
        "GBPJPY",
        "GBPUSD",
        "USDCAD",
        "USDJPY",
    ):
        row: dict[str, object] = {}
        for field in schema:
            if field.name == "symbol":
                row[field.name] = symbol
            elif field.name == "decision_time":
                row[field.name] = decision_time
            elif field.name == "feature_available_at":
                row[field.name] = available
            elif field.name in current_cross and ready:
                row[field.name] = 0.0
            elif field.nullable:
                row[field.name] = None
            elif pa.types.is_floating(field.type):
                row[field.name] = 0.0
            else:
                raise AssertionError(f"unhandled non-null field {field.name}")
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=schema)


def test_opportunity_ledger_has_48_paired_outcome_free_rows() -> None:
    decision = datetime(2026, 9, 1, tzinfo=UTC)
    ledger = materialize_expected_opportunities(_batch(decision, ready=True))

    validate_opportunity_ledger(ledger, (decision,))
    assert ledger.num_rows == 48
    assert set(ledger.column("arm").to_pylist()) == {"control", "candidate"}
    assert set(ledger.column("status").to_pylist()) == {"ready"}
    assert not any(
        forbidden in name
        for name in ledger.column_names
        for forbidden in ("score", "action", "return", "label", "pnl")
    )


def test_coverage_counts_explicit_missing_and_late_boundaries() -> None:
    first = datetime(2026, 9, 1, tzinfo=UTC)
    ledgers = [
        materialize_expected_opportunities(_batch(first, ready=True)),
        materialize_expected_opportunities(
            _batch(
                first + timedelta(minutes=5),
                ready=False,
                missing_symbols=("EURUSD",),
            )
        ),
        materialize_expected_opportunities(
            _batch(
                first + timedelta(minutes=10),
                ready=True,
                available_delay=timedelta(minutes=6),
            )
        ),
    ]
    ledger = pa.concat_tables(ledgers)
    expected = tuple(first + timedelta(minutes=5 * index) for index in range(4))
    report = summarize_coverage(ledger, expected)

    assert report.expected_sections == 4
    assert report.complete_sections == 1
    assert report.complete_ratio == pytest.approx(1 / 4)
    assert report.maximum_consecutive_missing == 3
    config = load_campaign2_engineering_config(
        PROJECT_ROOT / "configs/prospective/campaign-2-engineering-v1.toml"
    )
    assert not report.passes(config)


def test_engineering_schema_rejects_score_and_outcome_columns() -> None:
    forbidden = pa.schema([pa.field("return_bps", pa.float64())])
    with pytest.raises(ValueError, match="forbidden columns"):
        assert_outcome_free_schema(forbidden)

    decision = datetime(2026, 9, 1, tzinfo=UTC)
    malicious = _batch(
        decision,
        ready=False,
        missing_symbols=("return_bps:12",),
    )
    with pytest.raises(ValueError, match="canonical Campaign 2 symbols"):
        materialize_expected_opportunities(malicious)


def test_campaign_config_authorizes_engineering_only() -> None:
    path = PROJECT_ROOT / "configs/prospective/campaign-2-engineering-v1.toml"
    loaded = load_campaign2_engineering_config(path)
    with path.open("rb") as source:
        config = tomllib.load(source)

    assert loaded.minimum_complete_ratio == 0.95
    assert loaded.maximum_consecutive_missing_bars == 36
    assert loaded.maximum_feature_build_seconds_per_boundary == 1.0
    assert loaded.maximum_peak_rss_bytes == 1_073_741_824
    assert len(loaded.contract_sha256) == 5
    assert config["status"] == "engineering_only"
    assert config["tzdata_version"] == "2025.2"
    assert config["authorization"] == {
        "engineering": True,
        "fitting": False,
        "scoring": False,
        "collection": False,
        "evaluation": False,
        "raw_prospective_access": False,
    }

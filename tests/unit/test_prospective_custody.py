import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from demofml.calendars.prospective_fx import expected_decision_boundaries
from demofml.features.cross_pair import PAIRS
from demofml.prospective.bundle import (
    ENGINEERING_BUNDLE_SCOPE,
    freeze_engineering_bundle,
    verify_engineering_bundle,
)
from demofml.prospective.config import load_campaign2_engineering_config
from demofml.prospective.custody import (
    CollectionObjectClaim,
    SegmentQuality,
    build_collection_segment,
    build_collection_terminal,
    validate_collection_chain,
    validate_collection_segment,
    validate_collection_terminal,
)
from demofml.prospective.opportunities import CoverageReport
from demofml.prospective.qualification import (
    QualificationMeasurements,
    build_qualification_envelope,
    validate_qualification_envelope,
)
from demofml.prospective.records import (
    read_strict_json,
    write_immutable_json,
)

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs/prospective/campaign-2-engineering-v1.toml"
IMAGE_DIGEST = "sha256:" + "a" * 64
ATTESTATION_ID = "sha256-" + "b" * 64
ATTESTATION_SHA256 = "c" * 64


def _quality(rows: int = 2) -> SegmentQuality:
    return SegmentQuality(
        rows=rows,
        null_values=0,
        non_finite_values=0,
        non_positive_bid=0,
        non_positive_ask=0,
        crossed_quotes=0,
        inconsistent_mid=0,
        inconsistent_spread=0,
        provider_out_of_order=0,
        receipt_out_of_order=0,
        sequence_not_increasing=0,
        clock_lead_violations=0,
        mixed_symbols=0,
        max_delivery_latency_ns=10_000_000,
    )


def _segment(
    symbol: str,
    start: datetime,
    end: datetime,
    *,
    sequence: int = 0,
    previous: str | None = None,
    rows: int = 2,
    quality: SegmentQuality | None = None,
    object_key: str | None = None,
    object_version_id: str | None = None,
) -> dict[str, object]:
    first_sequence = sequence * 10 + 1
    claim = CollectionObjectClaim(
        object_key=object_key or f"qualification/{symbol}/{sequence:04d}.parquet",
        object_version_id=object_version_id or f"version-{symbol}-{sequence}",
        size_bytes=max(1024, rows * 64),
        sha256="d" * 64,
        rows=rows,
        first_ingest_sequence=first_sequence,
        last_ingest_sequence=first_sequence + rows - 1,
        provider_start=start,
        provider_end_exclusive=end,
        received_start=start,
        received_end_exclusive=end,
        object_last_modified=end,
    )
    return build_collection_segment(
        sequence=sequence,
        previous_segment_id=previous,
        symbol=symbol,
        claim=claim,
        quality=quality or _quality(rows),
        collector_attestation_id=ATTESTATION_ID,
        collector_attestation_sha256=ATTESTATION_SHA256,
        recorded_at=end,
    )


def _qualification_chains() -> dict[str, list[dict[str, object]]]:
    config = load_campaign2_engineering_config(CONFIG_PATH)
    expected_rows = len(
        expected_decision_boundaries(
            config.qualification_start, config.prospective_start
        )
    )
    return {
        symbol: [
            _segment(
                symbol,
                config.qualification_start,
                config.prospective_start,
                rows=expected_rows,
            )
        ]
        for symbol in PAIRS
    }


def test_collection_chain_binds_predecessor_ranges_and_terminal() -> None:
    start = datetime(2026, 3, 11, tzinfo=UTC)
    middle = start + timedelta(hours=1)
    end = middle + timedelta(hours=1)
    first = _segment("EURUSD", start, middle)
    second = _segment(
        "EURUSD",
        middle,
        end,
        sequence=1,
        previous=str(first["segment_id"]),
    )

    validate_collection_segment(first)
    summary = validate_collection_chain([first, second], expected_symbol="EURUSD")
    assert summary.rows == 4
    assert summary.segments == 2
    assert summary.qualified

    invalidated = _segment(
        "AUDUSD",
        start,
        middle,
        quality=replace(_quality(), invalidated_boundaries=(middle,)),
    )
    invalidated_summary = validate_collection_chain([invalidated])
    assert invalidated_summary.critical_violations == 1
    assert not invalidated_summary.qualified

    tampered = {**second, "previous_segment_id": "sha256-" + "e" * 64}
    with pytest.raises(ValueError, match="content ID mismatch"):
        validate_collection_segment(tampered)

    chains = _qualification_chains()
    terminal = build_collection_terminal(
        chains,
        recorded_at=load_campaign2_engineering_config(
            CONFIG_PATH
        ).prospective_start,
    )
    validate_collection_terminal(terminal, chains)
    assert terminal["all_chains_qualified"] is True
    assert terminal["scoring_authorized"] is False

    reused = {
        symbol: [
            _segment(
                symbol,
                start,
                middle,
                object_key="qualification/shared.parquet",
                object_version_id="shared-version",
            )
        ]
        for symbol in PAIRS
    }
    with pytest.raises(ValueError, match="reuses an object key"):
        build_collection_terminal(reused, recorded_at=middle)


def test_strict_records_are_immutable_and_reject_duplicate_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "record.json"
    write_immutable_json(path, {"format_version": 1})
    assert read_strict_json(path, "test record") == {"format_version": 1}
    with pytest.raises(RuntimeError, match="already exists"):
        write_immutable_json(path, {"format_version": 1})

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"id":1,"id":2}\n', encoding="ascii")
    with pytest.raises(ValueError, match="invalid"):
        read_strict_json(duplicate, "duplicate record")


def test_config_loader_rejects_scoring_authorization(tmp_path: Path) -> None:
    root = tmp_path / "project"
    (root / "configs/prospective").mkdir(parents=True)
    (root / "configs/bars").mkdir(parents=True)
    (root / "configs/features").mkdir(parents=True)
    (root / "configs/experiments").mkdir(parents=True)
    (root / "docs/research").mkdir(parents=True)
    shutil.copyfile(PROJECT_ROOT / "pyproject.toml", root / "pyproject.toml")
    for relative in (
        "configs/bars/prospective-quote-bars-v1.toml",
        "configs/features/cross-pair-v1.toml",
        "configs/experiments/prospective-executable-v1.toml",
        "docs/research/campaign-2-prospective-factor-plan.md",
    ):
        shutil.copyfile(PROJECT_ROOT / relative, root / relative)
    unsafe = CONFIG_PATH.read_text(encoding="utf-8").replace(
        "scoring = false", "scoring = true"
    )
    copied = root / "configs/prospective/campaign-2-engineering-v1.toml"
    copied.write_text(unsafe, encoding="utf-8")

    with pytest.raises(ValueError, match="engineering-only"):
        load_campaign2_engineering_config(copied)

    moved_boundary = CONFIG_PATH.read_text(encoding="utf-8").replace(
        'prospective_start = "2026-09-01T00:00:00Z"',
        'prospective_start = "2026-09-02T00:00:00Z"',
    )
    copied.write_text(moved_boundary, encoding="utf-8")
    with pytest.raises(ValueError, match="differ from the protocol"):
        load_campaign2_engineering_config(copied)


def test_engineering_bundle_is_standalone_exact_and_not_scoring_authorized(
    tmp_path: Path,
) -> None:
    output = tmp_path / "bundle"
    manifest = freeze_engineering_bundle(CONFIG_PATH, output, IMAGE_DIGEST)

    assert manifest["bundle_scope"] == ENGINEERING_BUNDLE_SCOPE
    assert manifest["scoring_authorized"] is False
    bundle_id = str(manifest["bundle_id"])
    assert verify_engineering_bundle(output, bundle_id) == manifest
    with pytest.raises(ValueError, match="identity is incompatible"):
        verify_engineering_bundle(output, "sha256-" + "0" * 64)
    inventory = manifest["inventory"]
    assert isinstance(inventory, list)
    assert all(isinstance(record, dict) for record in inventory)
    assert not any(
        "models/" in str(record["path"])
        for record in inventory
        if isinstance(record, dict)
    )

    first_record = inventory[0]
    assert isinstance(first_record, dict)
    artifact = output / str(first_record["path"])
    artifact.write_bytes(artifact.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="bundle artifact differs"):
        verify_engineering_bundle(output, bundle_id)


def test_qualification_envelope_passes_checks_but_never_authorizes(
    tmp_path: Path,
) -> None:
    config = load_campaign2_engineering_config(CONFIG_PATH)
    bundle_root = tmp_path / "bundle"
    bundle = freeze_engineering_bundle(CONFIG_PATH, bundle_root, IMAGE_DIGEST)
    chains = _qualification_chains()
    terminal = build_collection_terminal(
        chains, recorded_at=config.prospective_start
    )
    boundaries = expected_decision_boundaries(
        config.qualification_start, config.prospective_start
    )
    months = {boundary.strftime("%Y-%m"): 1.0 for boundary in boundaries}
    coverage = CoverageReport(len(boundaries), len(boundaries), 1.0, 0, months)
    measurements = QualificationMeasurements(
        schema_valid=True,
        first_determinism_sha256="e" * 64,
        second_determinism_sha256="e" * 64,
        feature_build_seconds_per_boundary=0.01,
        peak_rss_bytes=100_000_000,
        opportunity_ledger_sha256="f" * 64,
    )
    envelope = build_qualification_envelope(
        config=config,
        bundle_root=bundle_root,
        expected_bundle_id=str(bundle["bundle_id"]),
        collection_terminal=terminal,
        collection_chains=chains,
        coverage=coverage,
        expected_boundaries=boundaries,
        measurements=measurements,
    )

    validate_qualification_envelope(envelope)
    assert envelope["engineering_checks_passed"] is True
    assert envelope["qualification_complete"] is False
    assert envelope["external_attestation_required"] is True
    assert envelope["authorization_granted"] is False
    assert envelope["collection_authorized"] is False
    assert envelope["scoring_authorized"] is False
    assert envelope["evaluation_authorized"] is False

    inconsistent_months = dict(months)
    inconsistent_months[next(iter(inconsistent_months))] = 0.95
    contradictory = CoverageReport(
        len(boundaries),
        len(boundaries),
        1.0,
        0,
        inconsistent_months,
    )
    with pytest.raises(ValueError, match="internally incompatible"):
        build_qualification_envelope(
            config=config,
            bundle_root=bundle_root,
            expected_bundle_id=str(bundle["bundle_id"]),
            collection_terminal=terminal,
            collection_chains=chains,
            coverage=contradictory,
            expected_boundaries=boundaries,
            measurements=measurements,
        )

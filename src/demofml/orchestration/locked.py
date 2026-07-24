"""Frozen-candidate and one-shot locked-test orchestration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import tomllib
import uuid
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.bars.build import build_quote_bars
from demofml.bars.quotes import QUOTE_BAR_SCHEMA, validate_quote_bar_schema
from demofml.data.remote import (
    LOCKED_DATASET_SET_ID,
    LockedTestDataset,
    load_locked_test_dataset,
    load_published_manifest,
    materialize_locked_test_file,
    s3_client,
    verify_materialized_inventory,
)
from demofml.evaluation.portfolio import (
    PORTFOLIO_HORIZONS,
    PORTFOLIO_SET_ID,
    PORTFOLIO_SYMBOLS,
    load_portfolio_config,
    simulate_locked_portfolio,
)
from demofml.evaluation.signals import evaluate_locked_predictions
from demofml.features.build import build_features
from demofml.features.causal import FEATURE_SET_ID
from demofml.labels.build import build_labels
from demofml.labels.executable import LABEL_SET_ID, MAX_QUOTE_LATENCY_MINUTES
from demofml.models.baseline import (
    MODEL_SET_ID,
    align_research_tables,
    load_baseline_config,
)
from demofml.models.frozen import (
    FROZEN_MODEL_SET_ID,
    fit_frozen_ridge,
    load_frozen_ridge,
    write_frozen_ridge,
)
from demofml.models.locked import attach_locked_outcomes, score_locked_features
from demofml.orchestration.development import load_pipeline_config
from demofml.reporting.acceptance import (
    ACCEPTANCE_SET_ID,
    PIPELINE_SET_ID,
    evaluate_development_run,
    load_acceptance_config,
)
from demofml.reporting.portfolio import portfolio_report
from demofml.validation.splits import VALIDATION_SET_ID, load_validation_plan

LOCKED_TEST_SET_ID = "locked-test-evaluation-v1"
LOCKED_PREDICTION_SET_ID = "locked-test-predictions-v1"
LOCKED_EVALUATION_SET_ID = "locked-test-signal-metrics-v1"
FINAL_FIT_POLICY = "all_resolved_development_rows_before_decision_end"
UNRESOLVED_EXECUTION_POLICY = "flat_zero_return_after_blind_scoring"
ONE_SHOT_POLICY = "consume_before_remote_access_no_retry"
_CODE_REFERENCE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_CONTENT_ID_PATTERN = re.compile(r"^sha256-[0-9a-f]{64}$")
_HASH_BLOCK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class LockedTestConfig:
    """Immutable protocol fixed before a candidate can see locked data."""

    id: str
    source_pipeline_set: str
    source_acceptance_set: str
    locked_dataset_set: str
    model_artifact_set: str
    prediction_set: str
    evaluation_set: str
    final_fit_policy: str
    unresolved_execution_policy: str
    one_shot_policy: str
    feature_context_bars: int
    locked_test_start: datetime
    locked_test_end_exclusive: datetime
    decision_interval_minutes: int
    symbols: tuple[str, ...]
    horizons_minutes: tuple[int, ...]
    acceptance_config: Path
    source_pipeline_config: Path
    feature_config: Path
    label_config: Path
    validation_config: Path
    model_config: Path
    portfolio_config: Path
    minimum_positive_symbols_per_horizon: int
    minimum_trades_per_symbol_horizon: int
    minimum_mean_executable_return_bps_exclusive: float
    maximum_unresolved_execution_fraction: float
    minimum_total_return_exclusive: float
    maximum_drawdown_exclusive: float
    minimum_realized_annual_volatility: float
    maximum_realized_annual_volatility: float
    maximum_gross_leverage: float
    require_no_drawdown_halt: bool
    reconciliation_tolerance_usd: float

    @property
    def referenced_configs(self) -> tuple[Path, ...]:
        return (
            self.acceptance_config,
            self.source_pipeline_config,
            self.feature_config,
            self.label_config,
            self.validation_config,
            self.model_config,
            self.portfolio_config,
        )


@dataclass(frozen=True)
class CandidateFreezeResult:
    """Identity and location of one immutable candidate package."""

    candidate_id: str
    output: Path


@dataclass(frozen=True)
class LockedTestGrant:
    """Custodian authorization bound to every immutable test input."""

    grant_id: str
    candidate_id: str
    candidate_manifest_sha256: str
    protocol_config_sha256: str
    locked_dataset_config_sha256: str
    code_reference: str
    authorized_at: datetime


@dataclass(frozen=True)
class LockedTestRunResult:
    """Terminal one-shot result, including a valid scientific rejection."""

    grant_id: str
    candidate_id: str
    accepted: bool
    output: Path


@dataclass(frozen=True)
class _AcceptedSource:
    """One source artifact and its pre-acceptance stage attestation."""

    symbol: str
    role: str
    path: Path
    relative_path: str
    size_bytes: int
    sha256: str
    marker: Path
    marker_sha256: str


def _parse_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be an ISO-8601 UTC string")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 UTC string") from error
    if parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return parsed.astimezone(UTC)


def _config_path(parent: Path, value: object, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty path")
    path = (parent / value).resolve()
    if not path.is_file():
        raise RuntimeError(f"Locked protocol referenced config is not a file: {path}")
    return path


def load_locked_test_config(path: Path) -> LockedTestConfig:
    """Load and cross-check the static Phase 13 protocol."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Locked test config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    expected = {
        "format_version",
        "id",
        "source_pipeline_set",
        "source_acceptance_set",
        "locked_dataset_set",
        "model_artifact_set",
        "prediction_set",
        "evaluation_set",
        "final_fit_policy",
        "unresolved_execution_policy",
        "one_shot_policy",
        "feature_context_bars",
        "locked_test_start",
        "locked_test_end_exclusive",
        "decision_interval_minutes",
        "symbols",
        "horizons_minutes",
        "acceptance_config",
        "source_pipeline_config",
        "feature_config",
        "label_config",
        "validation_config",
        "model_config",
        "portfolio_config",
        "model",
        "portfolio",
    }
    if set(values) != expected:
        raise ValueError("locked test config fields are incompatible")
    try:
        if int(values["format_version"]) != 1:
            raise ValueError("locked test format_version must be 1")
        model = values["model"]
        portfolio = values["portfolio"]
        if not isinstance(model, dict) or not isinstance(portfolio, dict):
            raise ValueError("locked test criteria must be tables")
        if set(model) != {
            "minimum_positive_symbols_per_horizon",
            "minimum_trades_per_symbol_horizon",
            "minimum_mean_executable_return_bps_exclusive",
            "maximum_unresolved_execution_fraction",
        } or set(portfolio) != {
            "minimum_total_return_exclusive",
            "maximum_drawdown_exclusive",
            "minimum_realized_annual_volatility",
            "maximum_realized_annual_volatility",
            "maximum_gross_leverage",
            "require_no_drawdown_halt",
            "reconciliation_tolerance_usd",
        }:
            raise ValueError("locked test criteria fields are incompatible")
        config = LockedTestConfig(
            id=str(values["id"]),
            source_pipeline_set=str(values["source_pipeline_set"]),
            source_acceptance_set=str(values["source_acceptance_set"]),
            locked_dataset_set=str(values["locked_dataset_set"]),
            model_artifact_set=str(values["model_artifact_set"]),
            prediction_set=str(values["prediction_set"]),
            evaluation_set=str(values["evaluation_set"]),
            final_fit_policy=str(values["final_fit_policy"]),
            unresolved_execution_policy=str(values["unresolved_execution_policy"]),
            one_shot_policy=str(values["one_shot_policy"]),
            feature_context_bars=int(values["feature_context_bars"]),
            locked_test_start=_parse_utc(
                values["locked_test_start"], "locked_test_start"
            ),
            locked_test_end_exclusive=_parse_utc(
                values["locked_test_end_exclusive"], "locked_test_end_exclusive"
            ),
            decision_interval_minutes=int(values["decision_interval_minutes"]),
            symbols=tuple(str(value) for value in values["symbols"]),
            horizons_minutes=tuple(int(value) for value in values["horizons_minutes"]),
            acceptance_config=_config_path(
                path.parent, values["acceptance_config"], "acceptance_config"
            ),
            source_pipeline_config=_config_path(
                path.parent,
                values["source_pipeline_config"],
                "source_pipeline_config",
            ),
            feature_config=_config_path(
                path.parent, values["feature_config"], "feature_config"
            ),
            label_config=_config_path(
                path.parent, values["label_config"], "label_config"
            ),
            validation_config=_config_path(
                path.parent, values["validation_config"], "validation_config"
            ),
            model_config=_config_path(
                path.parent, values["model_config"], "model_config"
            ),
            portfolio_config=_config_path(
                path.parent, values["portfolio_config"], "portfolio_config"
            ),
            minimum_positive_symbols_per_horizon=int(
                model["minimum_positive_symbols_per_horizon"]
            ),
            minimum_trades_per_symbol_horizon=int(
                model["minimum_trades_per_symbol_horizon"]
            ),
            minimum_mean_executable_return_bps_exclusive=float(
                model["minimum_mean_executable_return_bps_exclusive"]
            ),
            maximum_unresolved_execution_fraction=float(
                model["maximum_unresolved_execution_fraction"]
            ),
            minimum_total_return_exclusive=float(
                portfolio["minimum_total_return_exclusive"]
            ),
            maximum_drawdown_exclusive=float(portfolio["maximum_drawdown_exclusive"]),
            minimum_realized_annual_volatility=float(
                portfolio["minimum_realized_annual_volatility"]
            ),
            maximum_realized_annual_volatility=float(
                portfolio["maximum_realized_annual_volatility"]
            ),
            maximum_gross_leverage=float(portfolio["maximum_gross_leverage"]),
            require_no_drawdown_halt=bool(portfolio["require_no_drawdown_halt"]),
            reconciliation_tolerance_usd=float(
                portfolio["reconciliation_tolerance_usd"]
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid locked test config field: {error}") from error
    if (
        config.id != LOCKED_TEST_SET_ID
        or config.source_pipeline_set != PIPELINE_SET_ID
        or config.source_acceptance_set != ACCEPTANCE_SET_ID
        or config.locked_dataset_set != LOCKED_DATASET_SET_ID
        or config.model_artifact_set != FROZEN_MODEL_SET_ID
        or config.prediction_set != LOCKED_PREDICTION_SET_ID
        or config.evaluation_set != LOCKED_EVALUATION_SET_ID
        or config.final_fit_policy != FINAL_FIT_POLICY
        or config.unresolved_execution_policy != UNRESOLVED_EXECUTION_POLICY
        or config.one_shot_policy != ONE_SHOT_POLICY
    ):
        raise ValueError("locked test protocol identity or policy is incompatible")
    if (
        config.symbols != PORTFOLIO_SYMBOLS
        or config.horizons_minutes != PORTFOLIO_HORIZONS
        or config.decision_interval_minutes != 5
        or config.feature_context_bars != 73
        or not config.locked_test_start < config.locked_test_end_exclusive
    ):
        raise ValueError("locked test universe or interval is incompatible")
    counts = (
        config.minimum_positive_symbols_per_horizon,
        config.minimum_trades_per_symbol_horizon,
    )
    thresholds = (
        config.minimum_mean_executable_return_bps_exclusive,
        config.maximum_unresolved_execution_fraction,
        config.minimum_total_return_exclusive,
        config.maximum_drawdown_exclusive,
        config.minimum_realized_annual_volatility,
        config.maximum_realized_annual_volatility,
        config.maximum_gross_leverage,
        config.reconciliation_tolerance_usd,
    )
    if any(value <= 0 for value in counts) or not all(
        math.isfinite(value) for value in thresholds
    ):
        raise ValueError("locked test criteria must be finite and positive")
    if not (
        config.minimum_positive_symbols_per_horizon <= len(config.symbols)
        and 0.0 <= config.maximum_unresolved_execution_fraction <= 1.0
        and 0.0 < config.maximum_drawdown_exclusive < 1.0
        and 0.0
        <= config.minimum_realized_annual_volatility
        < config.maximum_realized_annual_volatility
        and config.maximum_gross_leverage > 0.0
        and config.reconciliation_tolerance_usd > 0.0
    ):
        raise ValueError("locked test risk criteria are invalid")
    _validate_protocol_contracts(config)
    return config


def _validate_protocol_contracts(config: LockedTestConfig) -> None:
    acceptance = load_acceptance_config(config.acceptance_config)
    pipeline = load_pipeline_config(config.source_pipeline_config)
    plan = load_validation_plan(config.validation_config)
    model = load_baseline_config(config.model_config)
    portfolio = load_portfolio_config(config.portfolio_config)
    if (
        acceptance.id != config.source_acceptance_set
        or pipeline.id != config.source_pipeline_set
        or pipeline.acceptance_config != config.acceptance_config
        or pipeline.feature_config != config.feature_config
        or pipeline.label_config != config.label_config
        or pipeline.validation_config != config.validation_config
        or pipeline.model_config != config.model_config
        or pipeline.portfolio_config != config.portfolio_config
        or pipeline.symbols != config.symbols
        or plan.id != VALIDATION_SET_ID
        or model.id != MODEL_SET_ID
        or portfolio.id != PORTFOLIO_SET_ID
        or plan.feature_set != FEATURE_SET_ID
        or plan.label_set != LABEL_SET_ID
        or config.locked_test_start != plan.locked_test_start
        or config.locked_test_end_exclusive != plan.locked_test_end_exclusive
        or model.horizons_minutes != config.horizons_minutes
        or portfolio.symbols != config.symbols
        or portfolio.horizons_minutes != config.horizons_minutes
    ):
        raise ValueError("locked test referenced contracts differ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _development_run_id_from_contracts(
    pipeline_path: Path,
    referenced_configs: Sequence[Path],
    payloads: dict[Path, bytes],
    code_reference: str,
) -> str:
    roles = [
        "dataset",
        "features",
        "labels",
        "validation",
        "model",
        "portfolio",
        "acceptance",
    ]
    if len(referenced_configs) != len(roles):
        raise RuntimeError("Source pipeline contract count is incompatible")
    identity = {
        "pipeline_config_sha256": _sha256_bytes(payloads[pipeline_path]),
        "referenced_configs": [
            {"role": role, "sha256": _sha256_bytes(payloads[path])}
            for role, path in zip(roles, referenced_configs, strict=True)
        ],
        "code_reference": code_reference,
    }
    payload = _canonical_json(identity)[:-1]
    return f"sha256-{hashlib.sha256(payload).hexdigest()}"


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{name} is not a file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{name} is invalid: {path}") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{name} must be an object: {path}")
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with partial.open("xb") as target:
            target.write(_canonical_json(value))
            target.flush()
            os.fsync(target.fileno())
        os.link(partial, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except FileExistsError as error:
        raise RuntimeError(f"Immutable record already exists: {path}") from error
    finally:
        partial.unlink(missing_ok=True)


def _tail_context(path: Path, symbol: str, boundary: datetime, rows: int) -> pa.Table:
    parquet = pq.ParquetFile(path)
    validate_quote_bar_schema(parquet.schema_arrow)
    context = pa.Table.from_batches([], schema=QUOTE_BAR_SCHEMA)
    for batch in parquet.iter_batches(batch_size=10_000):
        table = pa.Table.from_batches([batch], schema=parquet.schema_arrow)
        mask = pc.and_(
            pc.less(table.column("bar_start"), pa.scalar(boundary)),
            pc.less_equal(table.column("bar_end"), pa.scalar(boundary)),
        )
        selected = table.filter(mask)
        if selected.num_rows:
            context = pa.concat_tables([context, selected])
            if context.num_rows > rows:
                context = context.slice(context.num_rows - rows)
    if context.num_rows != rows:
        raise RuntimeError(f"{symbol} has insufficient pre-lock feature context")
    if set(context.column("symbol").to_pylist()) != {symbol}:
        raise RuntimeError(f"{symbol} feature context has incompatible symbols")
    times = context.column("bar_end").to_pylist()
    if any(
        not isinstance(value, datetime) or (index and value <= times[index - 1])
        for index, value in enumerate(times)
    ):
        raise RuntimeError(f"{symbol} feature context is not strictly ordered")
    return context


def _artifact_records(root: Path, paths: Sequence[Path]) -> list[dict[str, object]]:
    records = []
    for path in sorted(paths):
        if not path.is_file():
            raise RuntimeError(f"Candidate artifact is missing: {path}")
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return records


def _capture_accepted_sources(
    root: Path, symbols: Sequence[str]
) -> tuple[_AcceptedSource, ...]:
    captured: list[_AcceptedSource] = []
    for symbol in symbols:
        symbol_root = root / "symbols" / symbol
        specs = (
            ("bars", symbol_root / "bars.stage.json", symbol_root / "bars.parquet"),
            (
                "features",
                symbol_root / "development.stage.json",
                symbol_root / "development" / "features.parquet",
            ),
            (
                "labels",
                symbol_root / "development.stage.json",
                symbol_root / "development" / "labels.parquet",
            ),
        )
        for role, marker, path in specs:
            record = _read_json(marker, "Accepted source stage marker")
            outputs = record.get("outputs")
            relative = path.relative_to(root).as_posix()
            matches = (
                [
                    output
                    for output in outputs
                    if isinstance(output, dict) and output.get("path") == relative
                ]
                if isinstance(outputs, list)
                else []
            )
            if record.get("format_version") != 1 or len(matches) != 1:
                raise RuntimeError("Accepted source stage marker is incompatible")
            output = matches[0]
            size = output.get("size_bytes")
            digest = output.get("sha256")
            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise RuntimeError("Accepted source output record is incompatible")
            captured.append(
                _AcceptedSource(
                    symbol,
                    role,
                    path,
                    relative,
                    size,
                    digest,
                    marker,
                    _sha256(marker),
                )
            )
    return tuple(captured)


def _snapshot_accepted_source(source: _AcceptedSource, destination: Path) -> Path:
    if _sha256(source.marker) != source.marker_sha256:
        raise RuntimeError("Accepted source stage marker changed during freeze")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source.path, destination)
    if (
        destination.stat().st_size != source.size_bytes
        or _sha256(destination) != source.sha256
    ):
        raise RuntimeError("Accepted source changed during candidate freeze")
    return destination


def _candidate_artifact_paths(root: Path) -> tuple[Path, ...]:
    ignored = {root / "candidate.json", root / "_FROZEN"}
    paths = tuple(
        path for path in root.rglob("*") if path.is_file() and path not in ignored
    )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise RuntimeError("Candidate package cannot contain symlinks")
    return paths


def freeze_candidate(
    development_run_root: Path,
    protocol_config_path: Path,
    output: Path,
    code_reference: str,
    *,
    acceptance_evaluator: Callable[[Path, Path], dict[str, Any]] = (
        evaluate_development_run
    ),
) -> CandidateFreezeResult:
    """Freeze one accepted development run without any remote-data access."""
    if not _CODE_REFERENCE_PATTERN.fullmatch(code_reference):
        raise ValueError("code reference must be an immutable sha256 image digest")
    source = development_run_root.expanduser().resolve()
    protocol_path = protocol_config_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace frozen candidate: {output}")
    config = load_locked_test_config(protocol_path)
    pipeline = load_pipeline_config(config.source_pipeline_config)
    contract_paths = tuple(
        dict.fromkeys(
            (
                protocol_path,
                *config.referenced_configs,
                *pipeline.referenced_configs,
            )
        )
    )
    if len({path.name for path in contract_paths}) != len(contract_paths):
        raise RuntimeError("Candidate contract basenames must be unique")
    contract_payloads = {path: path.read_bytes() for path in contract_paths}
    run_path = source / "run.json"
    validation_path = source / "validation" / "manifest.json"
    run_payload = run_path.read_bytes()
    validation_payload = validation_path.read_bytes()
    run = _read_json(run_path, "Development run record")
    run_id = str(run.get("run_id", ""))
    expected_run_id = _development_run_id_from_contracts(
        config.source_pipeline_config,
        pipeline.referenced_configs,
        contract_payloads,
        str(run.get("code_reference", "")),
    )
    accepted_sources = _capture_accepted_sources(source, config.symbols)
    recomputed = acceptance_evaluator(source, config.acceptance_config)
    acceptance_path = source / "acceptance" / f"{config.source_acceptance_set}.json"
    recorded = _read_json(acceptance_path, "Development acceptance report")
    if (
        recomputed != recorded
        or recomputed.get("summary", {}).get("accepted") is not True
    ):
        raise RuntimeError("Development run is not immutably accepted")
    success = _read_json(source / "_SUCCESS", "Development success marker")
    if (
        run_path.read_bytes() != run_payload
        or validation_path.read_bytes() != validation_payload
        or any(path.read_bytes() != contract_payloads[path] for path in contract_paths)
        or run.get("pipeline_set") != config.source_pipeline_set
        or run.get("acceptance_set") != config.source_acceptance_set
        or run.get("development_only") is not True
        or tuple(run.get("symbols", [])) != config.symbols
        or source.name != run_id
        or not _CONTENT_ID_PATTERN.fullmatch(run_id)
        or run_id != expected_run_id
        or success.get("run_id") != run_id
    ):
        raise RuntimeError("Development run provenance is incompatible")

    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    partial.mkdir(parents=True)
    try:
        copied_contracts: list[Path] = []
        for contract in contract_paths:
            target = partial / "contracts" / contract.name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(contract_payloads[contract])
            copied_contracts.append(target)
        plan = load_validation_plan(
            partial / "contracts" / config.validation_config.name
        )
        model_config = load_baseline_config(
            partial / "contracts" / config.model_config.name
        )
        evidence_payloads = (
            ("run.json", run_payload),
            (acceptance_path.name, acceptance_path.read_bytes()),
            ("manifest.json", validation_payload),
        )
        copied_evidence: list[Path] = []
        for name, payload in evidence_payloads:
            target = partial / "evidence" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
            copied_evidence.append(target)
        copied_markers: dict[Path, Path] = {}
        for accepted_source in accepted_sources:
            marker = accepted_source.marker
            if marker in copied_markers:
                continue
            if _sha256(marker) != accepted_source.marker_sha256:
                raise RuntimeError(
                    "Accepted source stage marker changed during acceptance"
                )
            target = (
                partial / "evidence" / "stages" / accepted_source.symbol / marker.name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(marker, target)
            if _sha256(target) != accepted_source.marker_sha256:
                raise RuntimeError("Accepted source stage marker copy differs")
            copied_markers[marker] = target
            copied_evidence.append(target)

        model_paths: list[Path] = []
        context_paths: list[Path] = []
        training_rows: dict[str, dict[str, int]] = {}
        sources_by_symbol = {
            symbol: {
                item.role: item for item in accepted_sources if item.symbol == symbol
            }
            for symbol in config.symbols
        }
        for symbol in config.symbols:
            source_snapshot = partial / ".source" / symbol
            symbol_sources = sources_by_symbol[symbol]
            bars_snapshot = _snapshot_accepted_source(
                symbol_sources["bars"], source_snapshot / "bars.parquet"
            )
            features_snapshot = _snapshot_accepted_source(
                symbol_sources["features"], source_snapshot / "features.parquet"
            )
            labels_snapshot = _snapshot_accepted_source(
                symbol_sources["labels"], source_snapshot / "labels.parquet"
            )
            features = pq.read_table(features_snapshot)
            labels = pq.read_table(labels_snapshot)
            aligned = align_research_tables(features, labels, plan, model_config)
            if any(
                decision >= plan.development_decision_end
                for decision in aligned.decision_times
            ):
                raise RuntimeError("Candidate training reaches the information purge")
            training_rows[symbol] = {}
            for horizon in config.horizons_minutes:
                mask = np.isfinite(aligned.long_targets[horizon]) & np.isfinite(
                    aligned.short_targets[horizon]
                )
                indices = np.flatnonzero(mask)
                if indices.size < model_config.minimum_training_rows:
                    raise RuntimeError(
                        f"{symbol} {horizon}m has insufficient final-fit rows"
                    )
                for index in indices:
                    decision = aligned.decision_times[int(index)]
                    entry = aligned.entry_times[int(index)]
                    exit_time = aligned.exit_times[horizon][int(index)]
                    if (
                        not isinstance(entry, datetime)
                        or not isinstance(exit_time, datetime)
                        or not decision
                        <= entry
                        <= decision + timedelta(minutes=MAX_QUOTE_LATENCY_MINUTES)
                        or not decision + timedelta(minutes=horizon)
                        <= exit_time
                        <= decision
                        + timedelta(minutes=horizon + MAX_QUOTE_LATENCY_MINUTES)
                        or exit_time >= plan.locked_test_start
                    ):
                        raise RuntimeError(
                            "Candidate training execution times are unsafe"
                        )
                targets = np.column_stack(
                    (
                        aligned.long_targets[horizon][indices],
                        aligned.short_targets[horizon][indices],
                    )
                )
                frozen = fit_frozen_ridge(
                    aligned.features[indices],
                    targets,
                    model_config,
                    symbol,
                    horizon,
                    tuple(aligned.decision_times[int(index)] for index in indices),
                )
                model_path = partial / "models" / symbol / f"{horizon}m.json"
                write_frozen_ridge(frozen, model_path)
                model_paths.append(model_path)
                training_rows[symbol][str(horizon)] = frozen.training_rows

            context = _tail_context(
                bars_snapshot,
                symbol,
                plan.locked_test_start,
                config.feature_context_bars,
            )
            context_path = partial / "feature-context" / f"{symbol}.parquet"
            context_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(context, context_path, compression="zstd")
            context_paths.append(context_path)
            shutil.rmtree(source_snapshot)
        (partial / ".source").rmdir()

        source_artifacts = [
            {
                "symbol": item.symbol,
                "role": item.role,
                "path": item.relative_path,
                "size_bytes": item.size_bytes,
                "sha256": item.sha256,
                "stage_marker_sha256": item.marker_sha256,
            }
            for item in accepted_sources
        ]

        artifacts = _artifact_records(
            partial,
            (*copied_contracts, *copied_evidence, *model_paths, *context_paths),
        )
        core = {
            "format_version": 1,
            "candidate_set": FROZEN_MODEL_SET_ID,
            "protocol_set": config.id,
            "source_pipeline_set": config.source_pipeline_set,
            "source_run_id": run_id,
            "source_code_reference": str(run["code_reference"]),
            "freeze_code_reference": code_reference,
            "dataset_set": str(run["dataset_set"]),
            "dataset_version": str(run["dataset_version"]),
            "symbols": list(config.symbols),
            "horizons_minutes": list(config.horizons_minutes),
            "final_fit_policy": config.final_fit_policy,
            "feature_context_bars": config.feature_context_bars,
            "training_decision_end_exclusive": (
                plan.development_decision_end.isoformat()
            ),
            "locked_test_start": plan.locked_test_start.isoformat(),
            "locked_test_end_exclusive": plan.locked_test_end_exclusive.isoformat(),
            "locked_data_accessed": False,
            "training_rows": training_rows,
            "source_artifacts": source_artifacts,
            "artifacts": artifacts,
        }
        candidate_id = f"sha256-{hashlib.sha256(_canonical_json(core)).hexdigest()}"
        manifest = {**core, "candidate_id": candidate_id}
        _write_json(partial / "candidate.json", manifest)
        _write_json(
            partial / "_FROZEN",
            {
                "format_version": 1,
                "candidate_id": candidate_id,
                "candidate_manifest_sha256": _sha256(partial / "candidate.json"),
            },
        )
        if output.exists():
            raise RuntimeError(f"Frozen candidate appeared during build: {output}")
        os.rename(partial, output)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    verify_candidate(output, protocol_path)
    return CandidateFreezeResult(candidate_id, output)


def verify_candidate(
    candidate_root: Path, protocol_config_path: Path
) -> dict[str, Any]:
    """Verify every candidate byte and referenced model before authorization."""
    expanded_root = candidate_root.expanduser()
    if expanded_root.is_symlink():
        raise RuntimeError("Candidate package root cannot be a symlink")
    root = expanded_root.resolve()
    config = load_locked_test_config(protocol_config_path)
    manifest = _read_json(root / "candidate.json", "Candidate manifest")
    frozen = _read_json(root / "_FROZEN", "Candidate frozen marker")
    expected_fields = {
        "format_version",
        "candidate_set",
        "protocol_set",
        "source_pipeline_set",
        "source_run_id",
        "source_code_reference",
        "freeze_code_reference",
        "dataset_set",
        "dataset_version",
        "symbols",
        "horizons_minutes",
        "final_fit_policy",
        "feature_context_bars",
        "training_decision_end_exclusive",
        "locked_test_start",
        "locked_test_end_exclusive",
        "locked_data_accessed",
        "training_rows",
        "source_artifacts",
        "artifacts",
        "candidate_id",
    }
    if set(manifest) != expected_fields:
        raise RuntimeError("Candidate manifest fields are incompatible")
    candidate_id = str(manifest.get("candidate_id", ""))
    core = {key: value for key, value in manifest.items() if key != "candidate_id"}
    expected_id = f"sha256-{hashlib.sha256(_canonical_json(core)).hexdigest()}"
    plan = load_validation_plan(config.validation_config)
    if (
        candidate_id != expected_id
        or manifest.get("format_version") != 1
        or manifest.get("candidate_set") != config.model_artifact_set
        or manifest.get("protocol_set") != config.id
        or manifest.get("source_pipeline_set") != config.source_pipeline_set
        or not _CONTENT_ID_PATTERN.fullmatch(str(manifest.get("source_run_id", "")))
        or not _CODE_REFERENCE_PATTERN.fullmatch(
            str(manifest.get("source_code_reference", ""))
        )
        or not _CODE_REFERENCE_PATTERN.fullmatch(
            str(manifest.get("freeze_code_reference", ""))
        )
        or not _CONTENT_ID_PATTERN.fullmatch(str(manifest.get("dataset_version", "")))
        or tuple(manifest.get("symbols", [])) != config.symbols
        or tuple(manifest.get("horizons_minutes", [])) != config.horizons_minutes
        or manifest.get("final_fit_policy") != config.final_fit_policy
        or manifest.get("feature_context_bars") != config.feature_context_bars
        or manifest.get("training_decision_end_exclusive")
        != plan.development_decision_end.isoformat()
        or manifest.get("locked_test_start") != plan.locked_test_start.isoformat()
        or manifest.get("locked_test_end_exclusive")
        != plan.locked_test_end_exclusive.isoformat()
        or manifest.get("locked_data_accessed") is not False
        or frozen
        != {
            "format_version": 1,
            "candidate_id": candidate_id,
            "candidate_manifest_sha256": _sha256(root / "candidate.json"),
        }
    ):
        raise RuntimeError("Candidate identity or frozen marker differs")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise RuntimeError("Candidate artifact manifest is invalid")
    expected_records = _artifact_records(root, _candidate_artifact_paths(root))
    if artifacts != expected_records:
        raise RuntimeError("Candidate artifact inventory or hashes differ")
    pipeline = load_pipeline_config(config.source_pipeline_config)
    contract_sources = tuple(
        dict.fromkeys(
            (
                protocol_config_path.expanduser().resolve(),
                *config.referenced_configs,
                *pipeline.referenced_configs,
            )
        )
    )
    expected_paths = {
        *(f"contracts/{path.name}" for path in contract_sources),
        "evidence/run.json",
        f"evidence/{config.source_acceptance_set}.json",
        "evidence/manifest.json",
        *(
            f"evidence/stages/{symbol}/{marker}"
            for symbol in config.symbols
            for marker in ("bars.stage.json", "development.stage.json")
        ),
        *(
            f"models/{symbol}/{horizon}m.json"
            for symbol in config.symbols
            for horizon in config.horizons_minutes
        ),
        *(f"feature-context/{symbol}.parquet" for symbol in config.symbols),
    }
    observed_paths = {str(record.get("path", "")) for record in artifacts}
    if observed_paths != expected_paths:
        raise RuntimeError("Candidate package layout is incompatible")
    for source in contract_sources:
        if _sha256(root / "contracts" / source.name) != _sha256(source):
            raise RuntimeError("Candidate contract differs from the frozen source")
    bundled_payloads = {
        source: (root / "contracts" / source.name).read_bytes()
        for source in contract_sources
    }
    source_run = _read_json(root / "evidence" / "run.json", "Candidate run evidence")
    source_acceptance = _read_json(
        root / "evidence" / f"{config.source_acceptance_set}.json",
        "Candidate acceptance evidence",
    )
    validation = _read_json(
        root / "evidence" / "manifest.json", "Candidate validation evidence"
    )
    if (
        source_run.get("format_version") != 1
        or source_run.get("pipeline_set") != config.source_pipeline_set
        or source_run.get("run_id") != manifest.get("source_run_id")
        or source_run.get("code_reference") != manifest.get("source_code_reference")
        or source_run.get("dataset_set") != manifest.get("dataset_set")
        or source_run.get("dataset_version") != manifest.get("dataset_version")
        or source_run.get("development_only") is not True
        or tuple(source_run.get("symbols", [])) != config.symbols
        or source_run.get("acceptance_set") != config.source_acceptance_set
        or source_run.get("run_id")
        != _development_run_id_from_contracts(
            config.source_pipeline_config,
            pipeline.referenced_configs,
            bundled_payloads,
            str(source_run.get("code_reference", "")),
        )
        or source_acceptance.get("format_version") != 1
        or source_acceptance.get("acceptance_set") != config.source_acceptance_set
        or source_acceptance.get("development_only") is not True
        or source_acceptance.get("run_id") != manifest.get("source_run_id")
        or source_acceptance.get("summary", {}).get("accepted") is not True
        or validation.get("id") != VALIDATION_SET_ID
        or validation.get("locked_test", {}).get("start")
        != config.locked_test_start.isoformat().replace("+00:00", "Z")
    ):
        raise RuntimeError("Candidate accepted-development evidence is incompatible")
    training_rows = manifest.get("training_rows")
    if not isinstance(training_rows, dict):
        raise RuntimeError("Candidate training-row evidence is invalid")
    bundled_model_config = load_baseline_config(
        root / "contracts" / config.model_config.name
    )
    source_artifacts = manifest.get("source_artifacts")
    if not isinstance(source_artifacts, list) or len(source_artifacts) != (
        len(config.symbols) * 3
    ):
        raise RuntimeError("Candidate source-artifact evidence is invalid")
    expected_source_keys = {
        (symbol, role)
        for symbol in config.symbols
        for role in ("bars", "features", "labels")
    }
    observed_source_keys: set[tuple[str, str]] = set()
    for record in source_artifacts:
        if not isinstance(record, dict):
            raise RuntimeError("Candidate source-artifact evidence is invalid")
        symbol = str(record.get("symbol", ""))
        role = str(record.get("role", ""))
        key = (symbol, role)
        expected_relative = {
            "bars": f"symbols/{symbol}/bars.parquet",
            "features": f"symbols/{symbol}/development/features.parquet",
            "labels": f"symbols/{symbol}/development/labels.parquet",
        }.get(role)
        marker_name = "bars.stage.json" if role == "bars" else "development.stage.json"
        marker_path = root / "evidence" / "stages" / symbol / marker_name
        marker = _read_json(marker_path, "Candidate source stage evidence")
        outputs = marker.get("outputs")
        matching = (
            [
                output
                for output in outputs
                if isinstance(output, dict) and output.get("path") == record.get("path")
            ]
            if isinstance(outputs, list)
            else []
        )
        if (
            key in observed_source_keys
            or key not in expected_source_keys
            or record.get("path") != expected_relative
            or set(record)
            != {
                "symbol",
                "role",
                "path",
                "size_bytes",
                "sha256",
                "stage_marker_sha256",
            }
            or len(matching) != 1
            or matching[0].get("size_bytes") != record.get("size_bytes")
            or matching[0].get("sha256") != record.get("sha256")
            or _sha256(marker_path) != record.get("stage_marker_sha256")
        ):
            raise RuntimeError("Candidate source-artifact evidence differs")
        observed_source_keys.add(key)
    if observed_source_keys != expected_source_keys:
        raise RuntimeError("Candidate source-artifact evidence is incomplete")
    for symbol in config.symbols:
        context = pq.read_table(root / "feature-context" / f"{symbol}.parquet")
        validate_quote_bar_schema(context.schema)
        context_rows = context.to_pylist()
        if (
            context.num_rows != config.feature_context_bars
            or any(row["symbol"] != symbol for row in context_rows)
            or any(
                not isinstance(row["bar_start"], datetime)
                or not isinstance(row["bar_end"], datetime)
                or row["bar_start"] >= config.locked_test_start
                or row["bar_end"] > config.locked_test_start
                for row in context_rows
            )
        ):
            raise RuntimeError("Candidate feature context is incompatible")
        previous_end: datetime | None = None
        for row in context_rows:
            bar_end = row["bar_end"]
            if not isinstance(bar_end, datetime) or (
                previous_end is not None and bar_end <= previous_end
            ):
                raise RuntimeError("Candidate feature context is not ordered")
            previous_end = bar_end
        symbol_rows = training_rows.get(symbol)
        if not isinstance(symbol_rows, dict):
            raise RuntimeError("Candidate training-row evidence is invalid")
        for horizon in config.horizons_minutes:
            model = load_frozen_ridge(root / "models" / symbol / f"{horizon}m.json")
            if (
                model.symbol != symbol
                or model.horizon_minutes != horizon
                or model.training_end >= plan.development_decision_end
                or symbol_rows.get(str(horizon)) != model.training_rows
                or model.alpha != bundled_model_config.alpha
                or model.features != bundled_model_config.features
            ):
                raise RuntimeError("Candidate frozen model identity differs")
    return manifest


def _grant_core(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if key != "grant_id"}


def load_locked_test_grant(path: Path) -> LockedTestGrant:
    """Load a content-addressed grant issued outside the research workflow."""
    values = _read_json(path.expanduser().resolve(), "Locked test grant")
    expected = {
        "format_version",
        "test_set",
        "grant_id",
        "candidate_id",
        "candidate_manifest_sha256",
        "protocol_config_sha256",
        "locked_dataset_config_sha256",
        "code_reference",
        "one_shot_policy",
        "authorized_at",
    }
    if set(values) != expected:
        raise ValueError("locked test grant fields are incompatible")
    digest = hashlib.sha256(_canonical_json(_grant_core(values))).hexdigest()
    expected_id = f"sha256-{digest}"
    authorized_at = _parse_utc(values["authorized_at"], "authorized_at")
    grant = LockedTestGrant(
        grant_id=str(values["grant_id"]),
        candidate_id=str(values["candidate_id"]),
        candidate_manifest_sha256=str(values["candidate_manifest_sha256"]),
        protocol_config_sha256=str(values["protocol_config_sha256"]),
        locked_dataset_config_sha256=str(values["locked_dataset_config_sha256"]),
        code_reference=str(values["code_reference"]),
        authorized_at=authorized_at,
    )
    if (
        values.get("format_version") != 1
        or values.get("test_set") != LOCKED_TEST_SET_ID
        or values.get("one_shot_policy") != ONE_SHOT_POLICY
        or grant.grant_id != expected_id
        or not _CONTENT_ID_PATTERN.fullmatch(grant.grant_id)
        or not _CONTENT_ID_PATTERN.fullmatch(grant.candidate_id)
        or not _CODE_REFERENCE_PATTERN.fullmatch(grant.code_reference)
        or any(
            re.fullmatch(r"[0-9a-f]{64}", value) is None
            for value in (
                grant.candidate_manifest_sha256,
                grant.protocol_config_sha256,
                grant.locked_dataset_config_sha256,
            )
        )
    ):
        raise ValueError("locked test grant identity is incompatible")
    return grant


def _validate_grant(
    grant: LockedTestGrant,
    candidate_root: Path,
    candidate: dict[str, Any],
    protocol_path: Path,
    dataset_path: Path,
    code_reference: str,
) -> None:
    if (
        grant.candidate_id != candidate.get("candidate_id")
        or grant.candidate_manifest_sha256 != _sha256(candidate_root / "candidate.json")
        or grant.protocol_config_sha256 != _sha256(protocol_path)
        or grant.locked_dataset_config_sha256 != _sha256(dataset_path)
        or grant.code_reference != code_reference
    ):
        raise RuntimeError("Locked test grant does not bind the requested inputs")


def _write_parquet(table: pa.Table, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def _candidate_contract(candidate_root: Path, source: Path) -> Path:
    path = candidate_root / "contracts" / source.name
    if not path.is_file():
        raise RuntimeError(f"Candidate contract is missing: {path}")
    return path


def _snapshot_candidate(source: Path, destination: Path, protocol_path: Path) -> Path:
    if destination.is_symlink():
        raise RuntimeError("Locked test candidate snapshot cannot be a symlink")
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError("Locked test candidate snapshot is not a directory")
        source_manifest = _sha256(source / "candidate.json")
        verify_candidate(destination, protocol_path)
        if _sha256(destination / "candidate.json") != source_manifest:
            raise RuntimeError("Locked test candidate snapshot identity differs")
        return destination
    partial = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.partial")
    try:
        shutil.copytree(source, partial, symlinks=True)
        verify_candidate(partial, protocol_path)
        os.rename(partial, destination)
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return destination


def _combine_locked_bars(
    context_path: Path, locked_path: Path, output: Path, symbol: str
) -> None:
    context = pq.read_table(context_path)
    locked = pq.read_table(locked_path)
    validate_quote_bar_schema(context.schema)
    validate_quote_bar_schema(locked.schema)
    if context.num_rows == 0 or locked.num_rows == 0:
        raise RuntimeError(f"{symbol} context and locked bars must be non-empty")
    context_last = context.column("bar_end")[-1].as_py()
    locked_first = locked.column("bar_end")[0].as_py()
    if (
        not isinstance(context_last, datetime)
        or not isinstance(locked_first, datetime)
        or context_last >= locked_first
    ):
        raise RuntimeError(f"{symbol} context overlaps locked bars")
    _write_parquet(pa.concat_tables([context, locked]), output)


def _period_returns_table(
    values: Sequence[float], config_id: str, candidate_id: str
) -> pa.Table:
    return pa.table(
        {
            "period_index": pa.array(range(len(values)), type=pa.int64()),
            "portfolio_return": pa.array(values, type=pa.float64()),
        }
    ).replace_schema_metadata(
        {
            b"demofml.portfolio_set": config_id.encode(),
            b"demofml.candidate_id": candidate_id.encode(),
            b"demofml.return_frequency": b"observed_five_minute_period",
        }
    )


def _locked_report(
    signal: dict[str, Any],
    portfolio: dict[str, Any],
    config: LockedTestConfig,
    candidate_id: str,
    grant_id: str,
    score_rows: int,
    evaluated_rows: int,
    unresolved_executions: int,
    portfolio_recomputed: bool,
) -> dict[str, Any]:
    aggregate = {int(row["horizon_minutes"]): row for row in signal["aggregate"]}
    symbols = signal["symbols"]
    expected_cells = {
        (symbol, horizon)
        for symbol in config.symbols
        for horizon in config.horizons_minutes
    }
    observed_cells = {
        (str(row["symbol"]), int(row["horizon_minutes"])) for row in symbols
    }
    positive_symbols = {
        horizon: sum(
            float(row["mean_executable_return_bps"])
            > config.minimum_mean_executable_return_bps_exclusive
            for row in symbols
            if int(row["horizon_minutes"]) == horizon
        )
        for horizon in config.horizons_minutes
    }
    trades_by_cell = {
        (str(row["symbol"]), int(row["horizon_minutes"])): int(row["trades"])
        for row in symbols
    }
    minimum_trades = min(trades_by_cell.get(cell, 0) for cell in expected_cells)
    unresolved_fraction = unresolved_executions / score_rows if score_rows else 1.0
    checks: list[dict[str, Any]] = [
        {
            "id": "inference.all_keys_scored",
            "status": "pass"
            if score_rows == evaluated_rows and evaluated_rows > 0
            else "fail",
            "observed": {
                "scores": score_rows,
                "evaluated": evaluated_rows,
                "unresolved": unresolved_executions,
            },
            "operator": "scores_eq_evaluated",
            "threshold": True,
        },
        {
            "id": "outcomes.complete_execution_cells",
            "status": "pass"
            if observed_cells == expected_cells
            and unresolved_fraction <= config.maximum_unresolved_execution_fraction
            else "fail",
            "observed": {
                "cells": len(observed_cells),
                "expected_cells": len(expected_cells),
                "unresolved_fraction": unresolved_fraction,
            },
            "operator": "exact_cells_and_lte_unresolved_fraction",
            "threshold": config.maximum_unresolved_execution_fraction,
        },
        {
            "id": "model.positive_horizon_means",
            "status": "pass"
            if all(
                float(aggregate[horizon]["mean_executable_return_bps"])
                > config.minimum_mean_executable_return_bps_exclusive
                for horizon in config.horizons_minutes
            )
            else "fail",
            "observed": {
                str(horizon): float(aggregate[horizon]["mean_executable_return_bps"])
                for horizon in config.horizons_minutes
            },
            "operator": "gt_each",
            "threshold": config.minimum_mean_executable_return_bps_exclusive,
        },
        {
            "id": "model.positive_symbols",
            "status": "pass"
            if all(
                value >= config.minimum_positive_symbols_per_horizon
                for value in positive_symbols.values()
            )
            else "fail",
            "observed": {str(key): value for key, value in positive_symbols.items()},
            "operator": "gte_each",
            "threshold": config.minimum_positive_symbols_per_horizon,
        },
        {
            "id": "model.minimum_trades",
            "status": "pass"
            if minimum_trades >= config.minimum_trades_per_symbol_horizon
            else "fail",
            "observed": minimum_trades,
            "operator": "gte",
            "threshold": config.minimum_trades_per_symbol_horizon,
        },
        {
            "id": "portfolio.total_return",
            "status": "pass"
            if float(portfolio["total_return"]) > config.minimum_total_return_exclusive
            else "fail",
            "observed": float(portfolio["total_return"]),
            "operator": "gt",
            "threshold": config.minimum_total_return_exclusive,
        },
        {
            "id": "portfolio.maximum_drawdown",
            "status": "pass"
            if float(portfolio["maximum_drawdown"]) < config.maximum_drawdown_exclusive
            else "fail",
            "observed": float(portfolio["maximum_drawdown"]),
            "operator": "lt",
            "threshold": config.maximum_drawdown_exclusive,
        },
        {
            "id": "portfolio.realized_volatility",
            "status": "pass"
            if config.minimum_realized_annual_volatility
            <= float(portfolio["realized_annual_volatility"])
            <= config.maximum_realized_annual_volatility
            else "fail",
            "observed": float(portfolio["realized_annual_volatility"]),
            "operator": "between_inclusive",
            "threshold": [
                config.minimum_realized_annual_volatility,
                config.maximum_realized_annual_volatility,
            ],
        },
        {
            "id": "portfolio.maximum_gross_leverage",
            "status": "pass"
            if float(portfolio["maximum_gross_leverage"])
            <= config.maximum_gross_leverage
            else "fail",
            "observed": float(portfolio["maximum_gross_leverage"]),
            "operator": "lte",
            "threshold": config.maximum_gross_leverage,
        },
        {
            "id": "portfolio.drawdown_halt",
            "status": "pass"
            if not config.require_no_drawdown_halt
            or portfolio["drawdown_halt_triggered"] is False
            else "fail",
            "observed": bool(portfolio["drawdown_halt_triggered"]),
            "operator": "eq",
            "threshold": False,
        },
        {
            "id": "portfolio.full_recomputation",
            "status": "pass" if portfolio_recomputed else "fail",
            "observed": portfolio_recomputed,
            "operator": "eq",
            "threshold": True,
        },
    ]
    failed = sum(check["status"] == "fail" for check in checks)
    return {
        "format_version": 1,
        "test_set": config.id,
        "candidate_id": candidate_id,
        "grant_id": grant_id,
        "development_only": False,
        "locked_test": True,
        "checks": checks,
        "summary": {
            "pass": len(checks) - failed,
            "fail": failed,
            "accepted": failed == 0,
        },
    }


def _run_locked_compute(
    root: Path,
    candidate_root: Path,
    candidate: dict[str, Any],
    config: LockedTestConfig,
    dataset: LockedTestDataset,
    grant: LockedTestGrant,
    code_reference: str,
    bucket: str,
    client: Any,
) -> LockedTestRunResult:
    load_published_manifest(client, bucket, dataset)
    private = root / "private"
    results = root / "results"
    if results.exists():
        raise RuntimeError("Locked test results already exist")
    partial = root / f".results.{uuid.uuid4().hex}.partial"
    partial.mkdir(parents=True)
    prediction_tables: list[pa.Table] = []
    score_tables: list[pa.Table] = []
    unresolved_total = 0
    symbol_state: dict[str, tuple[Path, Path]] = {}
    try:
        model_config = load_baseline_config(
            _candidate_contract(candidate_root, config.model_config)
        )
        plan = load_validation_plan(
            _candidate_contract(candidate_root, config.validation_config)
        )
        for symbol in config.symbols:
            entries = dataset.files_for_symbol(symbol)
            if not entries:
                raise RuntimeError(f"Locked dataset has no files for {symbol}")
            for entry in entries:
                materialize_locked_test_file(
                    client, bucket, dataset, entry, private / "inputs"
                )
            source = private / "inputs" / symbol
            verify_materialized_inventory(source, entries, path_prefix=symbol)
            symbol_private = private / "symbols" / symbol
            locked_bars = symbol_private / "locked-bars.parquet"
            build_quote_bars(source, locked_bars, symbol)
            combined = symbol_private / "combined-bars.parquet"
            _combine_locked_bars(
                candidate_root / "feature-context" / f"{symbol}.parquet",
                locked_bars,
                combined,
                symbol,
            )
            features = symbol_private / "features.parquet"
            build_features(combined, features, symbol)
            scores = score_locked_features(
                pq.read_table(features),
                plan,
                model_config,
                candidate_root,
                str(candidate["candidate_id"]),
                symbol,
            )
            score_tables.append(scores)
            symbol_state[symbol] = (combined, symbol_private / "labels.parquet")
            _write_parquet(scores, partial / "symbols" / symbol / "scores.parquet")

        score_paths = tuple(
            partial / "symbols" / symbol / "scores.parquet" for symbol in config.symbols
        )
        _write_json(
            partial / "_SCORES_FROZEN.json",
            {
                "format_version": 1,
                "candidate_id": str(candidate["candidate_id"]),
                "artifacts": _artifact_records(partial, score_paths),
            },
        )

        score_tables = []
        for symbol in config.symbols:
            combined, labels = symbol_state[symbol]
            build_labels(combined, labels, config.horizons_minutes, 0.0)
            scores = pq.read_table(partial / "symbols" / symbol / "scores.parquet")
            scored = attach_locked_outcomes(
                scores,
                pq.read_table(labels),
                plan,
                model_config,
                str(candidate["candidate_id"]),
                symbol,
            )
            score_tables.append(scores)
            prediction_tables.append(scored.evaluated_predictions)
            unresolved_total += scored.unresolved_executions
            _write_parquet(
                scored.evaluated_predictions,
                partial / "symbols" / symbol / "predictions.parquet",
            )

        all_predictions = pa.concat_tables(prediction_tables)
        all_scores = pa.concat_tables(score_tables)
        signal = evaluate_locked_predictions(
            all_predictions, str(candidate["candidate_id"])
        )
        _write_json(partial / "signal" / "metrics.json", signal)
        portfolio_config = load_portfolio_config(
            _candidate_contract(candidate_root, config.portfolio_config)
        )
        simulation = simulate_locked_portfolio(
            prediction_tables,
            portfolio_config,
            plan.locked_test_start,
            plan.locked_test_decision_end,
            plan.locked_test_end_exclusive,
            config.prediction_set,
            str(candidate["candidate_id"]),
        )
        portfolio = portfolio_report(
            simulation,
            portfolio_config,
            development_only=False,
            prediction_set=config.prediction_set,
            candidate_id=str(candidate["candidate_id"]),
        )
        portfolio_root = partial / "portfolio"
        _write_parquet(simulation.ledger, portfolio_root / "ledger.parquet")
        _write_parquet(simulation.equity, portfolio_root / "equity.parquet")
        _write_parquet(
            _period_returns_table(
                simulation.period_returns,
                portfolio_config.id,
                str(candidate["candidate_id"]),
            ),
            portfolio_root / "period-returns.parquet",
        )
        _write_json(portfolio_root / "metrics.json", portfolio)

        persisted_predictions = [
            pq.read_table(partial / "symbols" / symbol / "predictions.parquet")
            for symbol in config.symbols
        ]
        persisted_scores = [
            pq.read_table(partial / "symbols" / symbol / "scores.parquet")
            for symbol in config.symbols
        ]
        score_frozen = _read_json(
            partial / "_SCORES_FROZEN.json", "Frozen score manifest"
        )
        scores_verified = score_frozen == {
            "format_version": 1,
            "candidate_id": str(candidate["candidate_id"]),
            "artifacts": _artifact_records(partial, score_paths),
        }
        score_by_key = {
            (
                str(row["symbol"]),
                row["decision_time"],
                int(row["horizon_minutes"]),
            ): row
            for table in persisted_scores
            for row in table.to_pylist()
        }
        persisted_prediction_rows = [
            row for table in persisted_predictions for row in table.to_pylist()
        ]
        predictions_match_scores = all(
            (
                key := (
                    str(row["symbol"]),
                    row["decision_time"],
                    int(row["horizon_minutes"]),
                )
            )
            in score_by_key
            and row["predicted_long_return"]
            == score_by_key[key]["predicted_long_return"]
            and row["predicted_short_return"]
            == score_by_key[key]["predicted_short_return"]
            and (
                row["action"] == score_by_key[key]["action"]
                if row["execution_available"]
                else row["action"] == "flat" and row["realized_return"] == 0.0
            )
            for row in persisted_prediction_rows
        )
        persisted_unresolved = sum(
            row["execution_available"] is False for row in persisted_prediction_rows
        )
        replay = simulate_locked_portfolio(
            persisted_predictions,
            portfolio_config,
            plan.locked_test_start,
            plan.locked_test_decision_end,
            plan.locked_test_end_exclusive,
            config.prediction_set,
            str(candidate["candidate_id"]),
        )
        replay_report = portfolio_report(
            replay,
            portfolio_config,
            development_only=False,
            prediction_set=config.prediction_set,
            candidate_id=str(candidate["candidate_id"]),
        )
        persisted_ledger = pq.read_table(portfolio_root / "ledger.parquet")
        persisted_equity = pq.read_table(portfolio_root / "equity.parquet")
        persisted_returns = pq.read_table(portfolio_root / "period-returns.parquet")
        persisted_report = _read_json(
            portfolio_root / "metrics.json", "Locked portfolio report"
        )
        return_indices = persisted_returns.column("period_index").to_pylist()
        return_values = tuple(
            float(value)
            for value in persisted_returns.column("portfolio_return").to_pylist()
        )
        ledger_pnl = sum(
            float(value) for value in persisted_ledger.column("pnl_usd").to_pylist()
        )
        final_equity = float(persisted_equity.column("equity_usd")[-1].as_py())
        reconciled = (
            abs(final_equity - portfolio_config.initial_capital_usd - ledger_pnl)
            <= config.reconciliation_tolerance_usd
        )
        recomputed = (
            scores_verified
            and predictions_match_scores
            and persisted_unresolved == unresolved_total
            and replay.ledger.equals(persisted_ledger)
            and replay.equity.equals(persisted_equity)
            and return_indices == list(range(len(return_indices)))
            and replay.period_returns == return_values
            and replay_report == persisted_report
            and reconciled
        )
        report = _locked_report(
            signal,
            portfolio,
            config,
            str(candidate["candidate_id"]),
            grant.grant_id,
            all_scores.num_rows,
            all_predictions.num_rows,
            unresolved_total,
            recomputed,
        )
        _write_json(partial / "locked-test-report.json", report)
        _write_json(
            partial / "run.json",
            {
                "format_version": 1,
                "test_set": config.id,
                "grant_id": grant.grant_id,
                "candidate_id": str(candidate["candidate_id"]),
                "candidate_manifest_sha256": _sha256(candidate_root / "candidate.json"),
                "code_reference": code_reference,
                "dataset_set": dataset.id,
                "dataset_version": dataset.dataset_version,
                "dataset_config_sha256": grant.locked_dataset_config_sha256,
                "symbols": list(config.symbols),
                "horizons_minutes": list(config.horizons_minutes),
                "development_only": False,
                "locked_test": True,
                "one_shot_policy": config.one_shot_policy,
            },
        )
        artifact_paths = tuple(
            path
            for path in partial.rglob("*")
            if path.is_file() and path.name != "artifact-manifest.json"
        )
        _write_json(
            partial / "artifact-manifest.json",
            {
                "format_version": 1,
                "candidate_id": str(candidate["candidate_id"]),
                "grant_id": grant.grant_id,
                "artifacts": _artifact_records(partial, artifact_paths),
            },
        )
        artifact_manifest_sha256 = _sha256(partial / "artifact-manifest.json")
        os.rename(partial, results)
        shutil.rmtree(private, ignore_errors=True)
        _write_json(
            root / "_SUCCESS",
            {
                "format_version": 1,
                "grant_id": grant.grant_id,
                "candidate_id": str(candidate["candidate_id"]),
                "accepted": bool(report["summary"]["accepted"]),
                "artifact_manifest_sha256": artifact_manifest_sha256,
            },
        )
        return LockedTestRunResult(
            grant.grant_id,
            str(candidate["candidate_id"]),
            bool(report["summary"]["accepted"]),
            results,
        )
    finally:
        if partial.exists():
            shutil.rmtree(partial)


def run_locked_test_once(
    candidate_root: Path,
    protocol_config_path: Path,
    locked_dataset_config_path: Path,
    grant_path: Path,
    workdir: Path,
    code_reference: str,
    bucket: str,
    endpoint_url: str,
    region_name: str,
    *,
    s3_factory: Callable[[str, str], Any] = s3_client,
) -> LockedTestRunResult:
    """Consume one grant before constructing a client or reading remote data."""
    candidate_root = candidate_root.expanduser().resolve()
    protocol_path = protocol_config_path.expanduser().resolve()
    dataset_path = locked_dataset_config_path.expanduser().resolve()
    config = load_locked_test_config(protocol_path)
    candidate = verify_candidate(candidate_root, protocol_path)
    dataset = load_locked_test_dataset(dataset_path)
    if (
        dataset.id != config.locked_dataset_set
        or dataset.symbols != config.symbols
        or dataset.start != config.locked_test_start
        or dataset.end_exclusive != config.locked_test_end_exclusive
    ):
        raise RuntimeError("Locked dataset does not match the Phase 13 protocol")
    grant = load_locked_test_grant(grant_path)
    _validate_grant(
        grant,
        candidate_root,
        candidate,
        protocol_path,
        dataset_path,
        code_reference,
    )
    root = workdir.expanduser().resolve() / config.id
    started = root / "_LOCKED_TEST_STARTED.json"
    if started.exists():
        raise RuntimeError("Locked test one-shot has already been consumed")
    snapshot = _snapshot_candidate(
        candidate_root, root / "candidate-snapshot", protocol_path
    )
    candidate = verify_candidate(snapshot, protocol_path)
    _validate_grant(
        grant,
        snapshot,
        candidate,
        protocol_path,
        dataset_path,
        code_reference,
    )
    _write_json(
        started,
        {
            "format_version": 1,
            "test_set": config.id,
            "grant_id": grant.grant_id,
            "candidate_id": grant.candidate_id,
            "consumed_at": datetime.now(UTC).isoformat(),
            "one_shot_policy": config.one_shot_policy,
        },
    )
    try:
        client = s3_factory(endpoint_url, region_name)
        return _run_locked_compute(
            root,
            snapshot,
            candidate,
            config,
            dataset,
            grant,
            code_reference,
            bucket,
            client,
        )
    except Exception as error:
        with suppress(Exception):
            _write_json(
                root / "_FAILED.json",
                {
                    "format_version": 1,
                    "grant_id": grant.grant_id,
                    "candidate_id": grant.candidate_id,
                    "error_type": type(error).__name__,
                    "terminal": True,
                },
            )
        raise


def freeze_main(argv: Sequence[str] | None = None) -> None:
    """Run the candidate-freeze CLI without accepting remote-data options."""
    parser = argparse.ArgumentParser(
        description="Freeze one accepted development candidate for Phase 13."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--code-reference", default=os.environ.get("DEMOFML_IMAGE_DIGEST")
    )
    arguments = parser.parse_args(argv)
    if not arguments.code_reference:
        parser.error("set DEMOFML_IMAGE_DIGEST or pass --code-reference")
    try:
        result = freeze_candidate(
            arguments.run_root,
            arguments.protocol_config,
            arguments.output,
            arguments.code_reference,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(
        f"candidate frozen: candidate_id={result.candidate_id} output={result.output}"
    )


def evaluate_main(argv: Sequence[str] | None = None) -> None:
    """Run the terminal one-shot locked-test CLI."""
    parser = argparse.ArgumentParser(
        description="Consume one external grant and evaluate the frozen locked test."
    )
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--grant", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--code-reference", default=os.environ.get("DEMOFML_IMAGE_DIGEST")
    )
    parser.add_argument("--bucket", default=os.environ.get("DEMOFML_LOCKED_BUCKET"))
    parser.add_argument(
        "--endpoint-url", default=os.environ.get("DEMOFML_LOCKED_S3_ENDPOINT_URL")
    )
    parser.add_argument(
        "--region-name", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    arguments = parser.parse_args(argv)
    required = {
        "DEMOFML_IMAGE_DIGEST": arguments.code_reference,
        "DEMOFML_LOCKED_BUCKET": arguments.bucket,
        "DEMOFML_LOCKED_S3_ENDPOINT_URL": arguments.endpoint_url,
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        parser.error(f"set {missing[0]} or pass the corresponding argument")
    try:
        result = run_locked_test_once(
            arguments.candidate_root,
            arguments.protocol_config,
            arguments.dataset_config,
            arguments.grant,
            arguments.workdir,
            arguments.code_reference,
            arguments.bucket,
            arguments.endpoint_url,
            arguments.region_name,
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
    print(
        f"locked test complete: grant_id={result.grant_id} "
        f"candidate_id={result.candidate_id} accepted={result.accepted} "
        f"output={result.output}"
    )
    if not result.accepted:
        parser.exit(2, "locked test acceptance rejected\n")

"""Resumable, development-only execution of the complete research pipeline."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import re
import tomllib
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError  # type: ignore[import-untyped]
from mlflow import MlflowClient

from demofml.bars.build import build_quote_bars
from demofml.data.remote import (
    DevelopmentDataset,
    DevelopmentFile,
    load_development_dataset,
    load_published_manifest,
    materialize_development_file,
    s3_client,
)
from demofml.evaluation.portfolio import load_portfolio_config
from demofml.features.build import build_features
from demofml.features.causal import FEATURE_SCHEMA, FEATURE_SET_ID
from demofml.labels.build import build_labels
from demofml.labels.executable import (
    BAR_INTERVAL_MINUTES,
    DEFAULT_HORIZONS_MINUTES,
    LABEL_SET_ID,
    MAX_QUOTE_LATENCY_MINUTES,
)
from demofml.models.baseline import load_baseline_config
from demofml.models.build import run_baseline_experiment
from demofml.reporting.portfolio import run_portfolio_evaluation
from demofml.validation.build import build_validation_manifest
from demofml.validation.development import isolate_development_rows
from demofml.validation.splits import load_validation_plan

PIPELINE_SET_ID = "development-pipeline-v1"
_CODE_REFERENCE_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_HASH_BLOCK_SIZE = 8 * 1024 * 1024


@dataclass(frozen=True)
class PipelineConfig:
    """Resolved immutable configuration for one complete development run."""

    id: str
    dataset_config: Path
    feature_config: Path
    label_config: Path
    validation_config: Path
    model_config: Path
    portfolio_config: Path
    symbols: tuple[str, ...]
    mlflow_experiment: str
    locked_test_policy: str
    resume_policy: str

    @property
    def referenced_configs(self) -> tuple[Path, ...]:
        """Return every file that contributes to the run identity."""
        return (
            self.dataset_config,
            self.feature_config,
            self.label_config,
            self.validation_config,
            self.model_config,
            self.portfolio_config,
        )


@dataclass(frozen=True)
class PipelineRunResult:
    """Stable identifiers and output location for one completed run."""

    run_id: str
    mlflow_run_id: str
    output: Path


def _config_path(parent: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty path")
    path = (parent / value).resolve()
    if not path.is_file():
        raise RuntimeError(f"Pipeline referenced config is not a file: {path}")
    return path


def load_pipeline_config(path: Path) -> PipelineConfig:
    """Load the Phase 11 orchestration contract and resolve its dependencies."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(f"Pipeline config is not a file: {path}")
    with path.open("rb") as source:
        values = tomllib.load(source)
    try:
        if int(values["format_version"]) != 1:
            raise ValueError("pipeline format_version must be 1")
        config = PipelineConfig(
            id=str(values["id"]),
            dataset_config=_config_path(
                path.parent, values["dataset_config"], "dataset_config"
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
            symbols=tuple(str(value) for value in values["symbols"]),
            mlflow_experiment=str(values["mlflow_experiment"]),
            locked_test_policy=str(values["locked_test_policy"]),
            resume_policy=str(values["resume_policy"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid pipeline config field: {error}") from error
    if config.id != PIPELINE_SET_ID:
        raise ValueError(f"pipeline id must be {PIPELINE_SET_ID}")
    if not config.symbols or tuple(sorted(set(config.symbols))) != config.symbols:
        raise ValueError("pipeline symbols must be unique and ordered")
    if not config.mlflow_experiment:
        raise ValueError("pipeline MLflow experiment cannot be empty")
    if config.locked_test_policy != "forbidden":
        raise ValueError("pipeline locked test policy must remain forbidden")
    if config.resume_policy != "verified_stage_fingerprint":
        raise ValueError("pipeline resume policy must verify stage fingerprints")
    return config


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as source:
        return tomllib.load(source)


def _validate_contracts(
    config: PipelineConfig, dataset: DevelopmentDataset
) -> tuple[Any, Any]:
    plan = load_validation_plan(config.validation_config)
    model = load_baseline_config(config.model_config)
    portfolio = load_portfolio_config(config.portfolio_config)
    features = _load_toml(config.feature_config)
    labels = _load_toml(config.label_config)
    if dataset.symbols != config.symbols or portfolio.symbols != config.symbols:
        raise ValueError("pipeline, dataset, and portfolio symbols differ")
    if (
        dataset.start != plan.train_start
        or dataset.end_exclusive != plan.locked_test_start
    ):
        raise ValueError("development dataset does not match validation boundaries")
    expected_features = {
        "id": FEATURE_SET_ID,
        "source": "quote-bars-v1",
        "decision_time": "bar_end",
        "bar_interval_minutes": BAR_INTERVAL_MINUTES,
        "return_lags_bars": [1, 3, 12],
        "realized_volatility_windows_bars": [12, 72],
        "spread_zscore_window_bars": 72,
        "gap_policy": "reset_trailing_state",
        "features": FEATURE_SCHEMA.names[2:],
    }
    if features != expected_features:
        raise ValueError("pipeline feature config is incompatible")
    expected_labels = {
        "id": LABEL_SET_ID,
        "source": "quote-bars-v1",
        "decision_time": "bar_end",
        "entry": "first_quote_at_or_after_decision",
        "exit": "first_quote_at_or_after_horizon",
        "horizons_minutes": list(DEFAULT_HORIZONS_MINUTES),
        "minimum_return_bps": 0.0,
        "source_bar_interval_minutes": BAR_INTERVAL_MINUTES,
        "max_entry_latency_minutes": MAX_QUOTE_LATENCY_MINUTES,
        "max_exit_latency_minutes": MAX_QUOTE_LATENCY_MINUTES,
        "returns": {
            "long": "exit_bid / entry_ask - 1",
            "short": "1 - exit_ask / entry_bid",
        },
        "actions": {
            "positive_best_return": ["long", "short"],
            "otherwise": "flat",
        },
    }
    if labels != expected_labels:
        raise ValueError("pipeline label config is incompatible")
    horizons = tuple(int(value) for value in labels["horizons_minutes"])
    if horizons != DEFAULT_HORIZONS_MINUTES:
        raise ValueError("pipeline label horizons are incompatible")
    if float(labels.get("minimum_return_bps", math.nan)) != 0.0:
        raise ValueError("pipeline label threshold is incompatible")
    if model.horizons_minutes != horizons or portfolio.horizons_minutes != horizons:
        raise ValueError("pipeline model and portfolio horizons differ")
    return plan, portfolio


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(_HASH_BLOCK_SIZE):
            digest.update(block)
    return digest.hexdigest()


def _run_id(config_path: Path, config: PipelineConfig, code_reference: str) -> str:
    if not _CODE_REFERENCE_PATTERN.fullmatch(code_reference):
        raise ValueError("code reference must be an immutable sha256 image digest")
    identity = {
        "pipeline_config_sha256": _file_sha256(config_path),
        "referenced_configs": [
            {"role": role, "sha256": _file_sha256(path)}
            for role, path in zip(
                (
                    "dataset",
                    "features",
                    "labels",
                    "validation",
                    "model",
                    "portfolio",
                ),
                config.referenced_configs,
                strict=True,
            )
        ],
        "code_reference": code_reference,
    }
    payload = json.dumps(identity, separators=(",", ":"), sort_keys=True).encode()
    return f"sha256-{hashlib.sha256(payload).hexdigest()}"


def _output_records(outputs: Sequence[Path], root: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for output in outputs:
        if not output.is_file():
            raise RuntimeError(f"Stage output is not a file: {output}")
        records.append(
            {
                "path": output.relative_to(root).as_posix(),
                "size_bytes": output.stat().st_size,
                "sha256": _file_sha256(output),
            }
        )
    return records


def _read_json_object(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"{description} is not a file: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{description} is invalid: {path}") from error
    if not isinstance(record, dict):
        raise RuntimeError(f"{description} must be an object: {path}")
    return record


def _stage_is_complete(
    marker: Path, fingerprint: str, outputs: Sequence[Path], root: Path
) -> bool:
    if not marker.is_file():
        if marker.exists():
            raise RuntimeError(f"Stage marker is not a file: {marker}")
        return False
    record = _read_json_object(marker, "Stage marker")
    if record.get("fingerprint") != fingerprint:
        raise RuntimeError(f"Stage fingerprint differs: {marker}")
    if record.get("outputs") != _output_records(outputs, root):
        raise RuntimeError(f"Stage output hashes differ: {marker}")
    return True


def _write_json_no_replace(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        partial.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.link(partial, path)
    except FileExistsError as error:
        raise RuntimeError(f"Immutable record appeared during build: {path}") from error
    finally:
        partial.unlink(missing_ok=True)


def _write_json_replace(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        partial.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def _run_stage(
    root: Path,
    marker: Path,
    fingerprint: str,
    outputs: Sequence[Path],
    build: Callable[[], object],
) -> None:
    if _stage_is_complete(marker, fingerprint, outputs, root):
        print(f"verified; skipping stage {marker.stem}", flush=True)
        return
    intent = marker.with_name(f"{marker.name}.intent")
    if intent.exists():
        record = _read_json_object(intent, "Stage intent")
        if record.get("fingerprint") != fingerprint:
            raise RuntimeError(f"Stage intent fingerprint differs: {intent}")
    else:
        _write_json_no_replace(
            intent, {"format_version": 1, "fingerprint": fingerprint}
        )
    existing = [output.exists() for output in outputs]
    if any(existing):
        if not all(existing):
            raise RuntimeError(f"Stage outputs are incomplete: {marker}")
        _write_json_no_replace(
            marker,
            {
                "format_version": 1,
                "fingerprint": fingerprint,
                "outputs": _output_records(outputs, root),
            },
        )
        print(f"recovered stage {marker.stem}", flush=True)
        return
    build()
    _write_json_no_replace(
        marker,
        {
            "format_version": 1,
            "fingerprint": fingerprint,
            "outputs": _output_records(outputs, root),
        },
    )


def _stage_fingerprint(run_id: str, stage: str, symbol: str | None = None) -> str:
    value = f"{run_id}:{stage}:{symbol or 'portfolio'}".encode()
    return hashlib.sha256(value).hexdigest()


def _experiment_id(client: Any, name: str) -> str:
    experiment = client.get_experiment_by_name(name)
    if experiment is not None:
        return str(experiment.experiment_id)
    try:
        return str(client.create_experiment(name))
    except Exception:
        experiment = client.get_experiment_by_name(name)
        if experiment is None:
            raise
        return str(experiment.experiment_id)


def _create_tracking_run(
    client: Any, experiment_id: str, config: PipelineConfig, run_id: str
) -> str:
    tracked = client.create_run(
        experiment_id,
        tags={
            "component": "phase-11",
            "pipeline_set": config.id,
            "pipeline_run_id": run_id,
            "development_only": "true",
        },
    )
    return str(tracked.info.run_id)


def _tracking_status(client: Any, run_id: str) -> str:
    return str(client.get_run(run_id).info.status)


def _tracking_run(
    client: Any,
    experiment_id: str,
    config: PipelineConfig,
    run_id: str,
    record_path: Path,
) -> tuple[str, str]:
    if record_path.exists():
        record = _read_json_object(record_path, "MLflow tracking record")
        if record.get("pipeline_run_id") != run_id:
            raise RuntimeError("MLflow tracking record identity differs")
        mlflow_run_id = str(record.get("mlflow_run_id", ""))
        status = _tracking_status(client, mlflow_run_id)
        if status not in {"FAILED", "KILLED"}:
            return mlflow_run_id, status
    else:
        candidates = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.pipeline_run_id = '{run_id}'",
            max_results=2,
        )
        active = [
            run
            for run in candidates
            if str(run.info.status) not in {"FAILED", "KILLED"}
        ]
        if len(active) > 1:
            raise RuntimeError("Multiple active MLflow runs have the same identity")
        if active:
            mlflow_run_id = str(active[0].info.run_id)
            _write_json_replace(
                record_path,
                {"pipeline_run_id": run_id, "mlflow_run_id": mlflow_run_id},
            )
            return mlflow_run_id, str(active[0].info.status)
    mlflow_run_id = _create_tracking_run(client, experiment_id, config, run_id)
    _write_json_replace(
        record_path,
        {"pipeline_run_id": run_id, "mlflow_run_id": mlflow_run_id},
    )
    return mlflow_run_id, "RUNNING"


def _log_artifacts(
    client: Any,
    mlflow_run_id: str,
    root: Path,
    symbols: Sequence[str],
) -> None:
    client.log_artifact(
        mlflow_run_id, str(root / "validation" / "manifest.json"), "validation"
    )
    for symbol in symbols:
        baseline = root / "symbols" / symbol / "baseline"
        client.log_artifact(
            mlflow_run_id,
            str(baseline / "metrics.json"),
            f"symbols/{symbol}",
        )
        client.log_artifact(
            mlflow_run_id,
            str(baseline / "predictions.parquet"),
            f"symbols/{symbol}",
        )
    for name in ("metrics.json", "ledger.parquet", "equity.parquet"):
        client.log_artifact(
            mlflow_run_id, str(root / "portfolio" / name), "portfolio"
        )
    client.log_artifact(mlflow_run_id, str(root / "run.json"))


def _log_metrics(client: Any, mlflow_run_id: str, root: Path) -> None:
    report = json.loads((root / "portfolio" / "metrics.json").read_text())
    names = (
        "final_equity_usd",
        "total_return",
        "trades",
        "maximum_gross_leverage",
        "realized_annual_volatility",
        "maximum_drawdown",
    )
    for name in names:
        value = float(report[name])
        if not math.isfinite(value):
            raise RuntimeError(f"Portfolio metric is not finite: {name}")
        client.log_metric(mlflow_run_id, name, value)


def _build_symbol_bars(
    client: Any,
    bucket: str,
    dataset: DevelopmentDataset,
    entries: Sequence[DevelopmentFile],
    inputs: Path,
    output: Path,
    symbol: str,
) -> object:
    for entry in entries:
        materialize_development_file(client, bucket, dataset, entry, inputs)
    return build_quote_bars(
        inputs / symbol, output, symbol, BAR_INTERVAL_MINUTES
    )


def _run_development_pipeline(
    pipeline_config_path: Path,
    workdir: Path,
    code_reference: str,
    bucket: str,
    endpoint_url: str,
    region_name: str,
    tracking_uri: str,
    *,
    s3: Any | None = None,
    mlflow: Any | None = None,
) -> PipelineRunResult:
    """Run or safely resume every development stage and track the result."""
    pipeline_config_path = pipeline_config_path.expanduser().resolve()
    config = load_pipeline_config(pipeline_config_path)
    dataset = load_development_dataset(config.dataset_config)
    plan, _ = _validate_contracts(config, dataset)
    run_id = _run_id(pipeline_config_path, config, code_reference)
    root = workdir.expanduser().resolve() / config.id / run_id
    success = root / "_SUCCESS"
    success_record: dict[str, Any] | None = None
    if success.exists() and not success.is_file():
        raise RuntimeError("Pipeline success marker is not a file")
    if success.is_file():
        success_record = _read_json_object(success, "Pipeline success marker")
        if success_record.get("run_id") != run_id:
            raise RuntimeError("Pipeline success marker identity differs")

    s3 = s3 or s3_client(endpoint_url, region_name)
    load_published_manifest(s3, bucket, dataset)
    root.mkdir(parents=True, exist_ok=True)
    mlflow = mlflow or MlflowClient(tracking_uri=tracking_uri)
    experiment_id = _experiment_id(mlflow, config.mlflow_experiment)
    mlflow_run_id, tracking_status = _tracking_run(
        mlflow,
        experiment_id,
        config,
        run_id,
        root / "mlflow-run.json",
    )
    if success_record is not None and (
        success_record.get("mlflow_run_id") != mlflow_run_id
        or tracking_status != "FINISHED"
    ):
        raise RuntimeError("Pipeline success and MLflow state differ")
    try:
        if tracking_status == "RUNNING":
            mlflow.log_param(mlflow_run_id, "code_reference", code_reference)
            mlflow.log_param(mlflow_run_id, "dataset_set", dataset.id)
            mlflow.log_param(
                mlflow_run_id, "dataset_version", dataset.dataset_version
            )

        validation_output = root / "validation" / "manifest.json"
        _run_stage(
            root,
            root / "validation" / "stage.json",
            _stage_fingerprint(run_id, "validation"),
            [validation_output],
            partial(
                build_validation_manifest,
                config.validation_config,
                validation_output,
            ),
        )

        prediction_paths: list[Path] = []
        for symbol in config.symbols:
            entries = dataset.files_for_symbol(symbol)
            if not entries:
                raise RuntimeError(f"Development dataset has no files for {symbol}")
            symbol_root = root / "symbols" / symbol
            bars = symbol_root / "bars.parquet"
            _run_stage(
                root,
                symbol_root / "bars.stage.json",
                _stage_fingerprint(run_id, "bars", symbol),
                [bars],
                partial(
                    _build_symbol_bars,
                    s3,
                    bucket,
                    dataset,
                    entries,
                    root / "inputs",
                    bars,
                    symbol,
                ),
            )
            features = symbol_root / "features-full.parquet"
            _run_stage(
                root,
                symbol_root / "features.stage.json",
                _stage_fingerprint(run_id, "features", symbol),
                [features],
                partial(build_features, bars, features, symbol),
            )
            labels = symbol_root / "labels-full.parquet"
            _run_stage(
                root,
                symbol_root / "labels.stage.json",
                _stage_fingerprint(run_id, "labels", symbol),
                [labels],
                partial(
                    build_labels,
                    bars, labels, DEFAULT_HORIZONS_MINUTES, 0.0
                ),
            )
            development = symbol_root / "development"
            development_features = development / "features.parquet"
            development_labels = development / "labels.parquet"
            _run_stage(
                root,
                symbol_root / "development.stage.json",
                _stage_fingerprint(run_id, "development", symbol),
                [development_features, development_labels],
                partial(
                    isolate_development_rows,
                    features, labels, plan, development
                ),
            )
            baseline = symbol_root / "baseline"
            predictions = baseline / "predictions.parquet"
            _run_stage(
                root,
                symbol_root / "baseline.stage.json",
                _stage_fingerprint(run_id, "baseline", symbol),
                [predictions, baseline / "metrics.json"],
                partial(
                    run_baseline_experiment,
                    development_features,
                    development_labels,
                    config.validation_config,
                    config.model_config,
                    baseline,
                ),
            )
            prediction_paths.append(predictions)

        portfolio = root / "portfolio"
        _run_stage(
            root,
            root / "portfolio.stage.json",
            _stage_fingerprint(run_id, "portfolio"),
            [
                portfolio / "ledger.parquet",
                portfolio / "equity.parquet",
                portfolio / "metrics.json",
            ],
            lambda: run_portfolio_evaluation(
                prediction_paths,
                config.portfolio_config,
                config.validation_config,
                portfolio,
            ),
        )
        run_record = {
            "format_version": 1,
            "pipeline_set": config.id,
            "run_id": run_id,
            "code_reference": code_reference,
            "dataset_set": dataset.id,
            "dataset_version": dataset.dataset_version,
            "symbols": list(config.symbols),
            "development_only": True,
            "locked_test_start": plan.locked_test_start.isoformat(),
        }
        run_record_path = root / "run.json"
        if run_record_path.exists():
            existing = json.loads(run_record_path.read_text(encoding="utf-8"))
            if existing != run_record:
                raise RuntimeError("Pipeline run record differs")
        else:
            _write_json_no_replace(run_record_path, run_record)
        if tracking_status == "RUNNING":
            _log_metrics(mlflow, mlflow_run_id, root)
            _log_artifacts(mlflow, mlflow_run_id, root, config.symbols)
            mlflow.set_terminated(mlflow_run_id, status="FINISHED")
        elif tracking_status != "FINISHED":
            raise RuntimeError(f"MLflow run cannot be resumed from {tracking_status}")
        if success_record is None:
            _write_json_no_replace(
                success, {"run_id": run_id, "mlflow_run_id": mlflow_run_id}
            )
    except Exception:
        try:
            if _tracking_status(mlflow, mlflow_run_id) == "RUNNING":
                mlflow.set_terminated(mlflow_run_id, status="FAILED")
        except Exception:
            pass
        raise
    return PipelineRunResult(run_id, mlflow_run_id, root)


@contextmanager
def _exclusive_run(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".pipeline.lock").open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Pipeline run is already active: {root}") from error
        yield
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()


def run_development_pipeline(
    pipeline_config_path: Path,
    workdir: Path,
    code_reference: str,
    bucket: str,
    endpoint_url: str,
    region_name: str,
    tracking_uri: str,
    *,
    s3: Any | None = None,
    mlflow: Any | None = None,
) -> PipelineRunResult:
    """Run one exclusive pipeline attempt or safely resume a previous attempt."""
    resolved_config = pipeline_config_path.expanduser().resolve()
    config = load_pipeline_config(resolved_config)
    run_id = _run_id(resolved_config, config, code_reference)
    root = workdir.expanduser().resolve() / config.id / run_id
    with _exclusive_run(root):
        return _run_development_pipeline(
            resolved_config,
            workdir,
            code_reference,
            bucket,
            endpoint_url,
            region_name,
            tracking_uri,
            s3=s3,
            mlflow=mlflow,
        )


def _required(value: str | None, name: str, parser: argparse.ArgumentParser) -> str:
    if not value:
        parser.error(f"set {name} or pass the corresponding argument")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    """Run the Phase 11 development pipeline command line interface."""
    parser = argparse.ArgumentParser(
        description="Run the verified, resumable development research pipeline."
    )
    parser.add_argument("--pipeline-config", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--code-reference", default=os.environ.get("DEMOFML_IMAGE_DIGEST")
    )
    parser.add_argument("--bucket", default=os.environ.get("DEMOFML_DATA_BUCKET"))
    parser.add_argument("--endpoint-url", default=os.environ.get("S3_ENDPOINT_URL"))
    parser.add_argument(
        "--region-name", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI"))
    arguments = parser.parse_args(argv)
    code_reference = _required(
        arguments.code_reference, "DEMOFML_IMAGE_DIGEST", parser
    )
    bucket = _required(arguments.bucket, "DEMOFML_DATA_BUCKET", parser)
    endpoint_url = _required(arguments.endpoint_url, "S3_ENDPOINT_URL", parser)
    tracking_uri = _required(arguments.tracking_uri, "MLFLOW_TRACKING_URI", parser)
    try:
        result = run_development_pipeline(
            arguments.pipeline_config,
            arguments.workdir,
            code_reference,
            bucket,
            endpoint_url,
            arguments.region_name,
            tracking_uri,
        )
        print(
            f"development pipeline complete: run_id={result.run_id} "
            f"mlflow_run_id={result.mlflow_run_id} output={result.output}"
        )
    except (ClientError, OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

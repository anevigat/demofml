import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import demofml.orchestration.development as development_module
from demofml.features.causal import FEATURE_SCHEMA
from demofml.labels.executable import label_schema
from demofml.models.baseline import FEATURE_COLUMNS
from demofml.orchestration.development import (
    load_pipeline_config,
    run_development_pipeline,
)
from demofml.validation.development import isolate_development_rows
from demofml.validation.splits import load_validation_plan

PROJECT_ROOT = Path(__file__).parents[2]
PIPELINE_CONFIG = PROJECT_ROOT / "configs/experiments/development-pipeline-v1.toml"
VALIDATION_CONFIG = PROJECT_ROOT / "configs/experiments/purged-walk-forward-v1.toml"


class _Mlflow:
    def __init__(self) -> None:
        self.terminated: list[tuple[str, str]] = []
        self.artifacts: list[str] = []
        self.status = "RUNNING"

    def get_experiment_by_name(self, name: str) -> object | None:
        return None

    def create_experiment(self, name: str) -> str:
        return "experiment-1"

    def create_run(self, experiment_id: str, tags: dict[str, str]) -> object:
        return SimpleNamespace(info=SimpleNamespace(run_id="mlflow-run-1"))

    def search_runs(self, **arguments: Any) -> list[object]:
        return []

    def get_run(self, run_id: str) -> object:
        return SimpleNamespace(info=SimpleNamespace(status=self.status))

    def log_param(self, run_id: str, name: str, value: object) -> None:
        pass

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        pass

    def log_artifact(
        self, run_id: str, local_path: str, artifact_path: str | None = None
    ) -> None:
        self.artifacts.append(local_path)

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated.append((run_id, status))
        self.status = status


def _research_tables() -> tuple[pa.Table, pa.Table]:
    times = [
        datetime(2017, 12, 31, 23, 55, tzinfo=UTC),
        datetime(2018, 1, 1, tzinfo=UTC),
        datetime(2024, 12, 31, 22, 50, tzinfo=UTC),
        datetime(2024, 12, 31, 22, 55, tzinfo=UTC),
    ]
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for index, decision in enumerate(times):
        feature = {"symbol": "EURUSD", "bar_end": decision}
        feature.update({name: float(index) for name in FEATURE_COLUMNS})
        feature_rows.append(feature)
        label: dict[str, object] = {
            "symbol": "EURUSD",
            "decision_time": decision,
            "entry_time": decision,
            "entry_bid": 1.0,
            "entry_ask": 1.1,
        }
        for horizon in (15, 30, 60):
            label[f"exit_time_{horizon}m"] = decision + timedelta(minutes=horizon)
            label[f"long_return_{horizon}m"] = 0.01
            label[f"short_return_{horizon}m"] = -0.01
            label[f"action_{horizon}m"] = "long"
        label_rows.append(label)
    return (
        pa.Table.from_pylist(feature_rows, schema=FEATURE_SCHEMA),
        pa.Table.from_pylist(label_rows, schema=label_schema((15, 30, 60))),
    )


def test_development_slice_applies_information_window(tmp_path: Path) -> None:
    features, labels = _research_tables()
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    pq.write_table(features, features_path)
    pq.write_table(labels, labels_path)
    output = tmp_path / "development"

    result = isolate_development_rows(
        features_path,
        labels_path,
        load_validation_plan(VALIDATION_CONFIG),
        output,
    )

    selected = pq.read_table(output / "features.parquet")
    assert result.input_rows == 4
    assert result.output_rows == 2
    assert selected.column("bar_end").to_pylist()[-1] == datetime(
        2024, 12, 31, 22, 50, tzinfo=UTC
    )


def test_development_slice_rejects_misalignment(tmp_path: Path) -> None:
    features, labels = _research_tables()
    labels = labels.set_column(
        1,
        "decision_time",
        pa.array(
            [
                value + timedelta(minutes=5)
                for value in labels["decision_time"].to_pylist()
            ],
            type=pa.timestamp("ns", tz="UTC"),
        ),
    )
    features_path = tmp_path / "features.parquet"
    labels_path = tmp_path / "labels.parquet"
    pq.write_table(features, features_path)
    pq.write_table(labels, labels_path)

    with pytest.raises(ValueError, match="decision times are not aligned"):
        isolate_development_rows(
            features_path,
            labels_path,
            load_validation_plan(VALIDATION_CONFIG),
            tmp_path / "output",
        )


def test_pipeline_config_binds_all_research_contracts() -> None:
    config = load_pipeline_config(PIPELINE_CONFIG)

    assert config.id == "development-pipeline-v1"
    assert len(config.symbols) == 8
    assert len(config.referenced_configs) == 6
    assert config.locked_test_policy == "forbidden"


def test_stage_recovers_output_published_before_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    marker = tmp_path / "stage.json"

    def interrupted() -> None:
        output.write_bytes(b"complete")
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        development_module._run_stage(
            tmp_path, marker, "fingerprint", [output], interrupted
        )

    development_module._run_stage(
        tmp_path,
        marker,
        "fingerprint",
        [output],
        lambda: pytest.fail("completed output must be recovered"),
    )

    assert marker.is_file()


def test_pipeline_lock_rejects_concurrent_attempt(tmp_path: Path) -> None:
    with (
        development_module._exclusive_run(tmp_path),
        pytest.raises(RuntimeError, match="already active"),
        development_module._exclusive_run(tmp_path),
    ):
        pytest.fail("second attempt must not acquire the lock")


def test_pipeline_runs_all_stages_once_and_then_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def file_builder(name: str) -> Any:
        def build(*arguments: Any) -> None:
            output = [
                argument
                for argument in arguments
                if isinstance(argument, Path) and argument.suffix == ".parquet"
            ][-1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(name.encode())
            calls.append(name)

        return build

    def validation(config: Path, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}\n")
        calls.append("validation")

    def isolated(
        features: Path, labels: Path, plan: object, output: Path
    ) -> None:
        output.mkdir(parents=True)
        (output / "features.parquet").write_bytes(b"features-development")
        (output / "labels.parquet").write_bytes(b"labels-development")
        calls.append("slice")

    def baseline(*arguments: Any) -> None:
        output = arguments[-1]
        output.mkdir(parents=True)
        (output / "predictions.parquet").write_bytes(b"predictions")
        (output / "metrics.json").write_text("{}\n")
        calls.append("baseline")

    def portfolio(*arguments: Any) -> None:
        output = arguments[-1]
        output.mkdir(parents=True)
        (output / "ledger.parquet").write_bytes(b"ledger")
        (output / "equity.parquet").write_bytes(b"equity")
        (output / "metrics.json").write_text(
            json.dumps(
                {
                    "final_equity_usd": 100_001.0,
                    "total_return": 0.00001,
                    "trades": 24,
                    "maximum_gross_leverage": 1.0,
                    "realized_annual_volatility": 0.1,
                    "maximum_drawdown": 0.01,
                }
            )
        )
        calls.append("portfolio")

    monkeypatch.setattr(development_module, "load_published_manifest", lambda *a: {})
    monkeypatch.setattr(
        development_module, "materialize_development_file", lambda *a: tmp_path
    )
    monkeypatch.setattr(development_module, "build_validation_manifest", validation)
    monkeypatch.setattr(development_module, "build_quote_bars", file_builder("bars"))
    monkeypatch.setattr(development_module, "build_features", file_builder("features"))
    monkeypatch.setattr(development_module, "build_labels", file_builder("labels"))
    monkeypatch.setattr(development_module, "isolate_development_rows", isolated)
    monkeypatch.setattr(development_module, "run_baseline_experiment", baseline)
    monkeypatch.setattr(development_module, "run_portfolio_evaluation", portfolio)
    tracking = _Mlflow()

    result = run_development_pipeline(
        PIPELINE_CONFIG,
        tmp_path / "work",
        "sha256:" + "a" * 64,
        "data",
        "https://s3.invalid",
        "us-east-1",
        "https://mlflow.invalid",
        s3=object(),
        mlflow=tracking,
    )
    repeated = run_development_pipeline(
        PIPELINE_CONFIG,
        tmp_path / "work",
        "sha256:" + "a" * 64,
        "data",
        "https://s3.invalid",
        "us-east-1",
        "https://mlflow.invalid",
        s3=object(),
        mlflow=tracking,
    )

    assert result == repeated
    assert calls.count("bars") == 8
    assert calls.count("baseline") == 8
    assert calls.count("portfolio") == 1
    assert tracking.terminated == [("mlflow-run-1", "FINISHED")]
    assert (result.output / "_SUCCESS").is_file()

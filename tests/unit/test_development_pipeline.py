import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import demofml.orchestration.development as development_module
from demofml.evaluation.signals import evaluate_predictions
from demofml.features.causal import FEATURE_SCHEMA
from demofml.labels.executable import label_schema
from demofml.models.baseline import (
    FEATURE_COLUMNS,
    load_baseline_config,
    prediction_schema,
)
from demofml.orchestration.development import (
    load_pipeline_config,
    run_development_pipeline,
)
from demofml.reporting.acceptance import evaluate_development_run
from demofml.reporting.portfolio import (
    run_portfolio_evaluation as run_actual_portfolio_evaluation,
)
from demofml.validation.development import isolate_development_rows
from demofml.validation.splits import load_validation_plan

PROJECT_ROOT = Path(__file__).parents[2]
PIPELINE_CONFIG = PROJECT_ROOT / "configs/experiments/development-pipeline-v2.toml"
VALIDATION_CONFIG = PROJECT_ROOT / "configs/experiments/purged-walk-forward-v1.toml"
MODEL_CONFIG = PROJECT_ROOT / "configs/experiments/baseline-ridge-v1.toml"


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

    assert config.id == "development-pipeline-v2"
    assert len(config.symbols) == 8
    assert len(config.referenced_configs) == 7
    assert config.locked_test_policy == "forbidden"


def test_published_phase_11_pipeline_config_remains_loadable() -> None:
    config = load_pipeline_config(
        PROJECT_ROOT / "configs/experiments/development-pipeline-v1.toml"
    )

    assert config.id == "development-pipeline-v1"
    assert config.acceptance_config is None
    assert len(config.referenced_configs) == 6


def test_stage_recovers_output_published_before_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "output.parquet"
    marker = tmp_path / "stage.json"

    def interrupted() -> None:
        pq.write_table(pa.table({"value": ["complete"]}), output)
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        development_module._run_stage(
            tmp_path, marker, "fingerprint", [output], interrupted
        )

    executions: list[development_module.StageExecution] = []
    development_module._run_stage(
        tmp_path,
        marker,
        "fingerprint",
        [output],
        lambda: pytest.fail("completed output must be recovered"),
        stage="bars",
        symbol="EURUSD",
        executions=executions,
    )

    assert marker.is_file()
    assert executions[0].action == "checkpoint_recovered"
    assert executions[0].resumed is True


def test_stage_profiles_fresh_and_verified_actions(tmp_path: Path) -> None:
    output = tmp_path / "output.json"
    marker = tmp_path / "stage.json"
    fresh: list[development_module.StageExecution] = []
    verified: list[development_module.StageExecution] = []

    development_module._run_stage(
        tmp_path,
        marker,
        "fingerprint",
        [output],
        lambda: output.write_text("{}\n"),
        stage="validation",
        executions=fresh,
    )
    development_module._run_stage(
        tmp_path,
        marker,
        "fingerprint",
        [output],
        lambda: pytest.fail("verified stage must not execute"),
        stage="validation",
        executions=verified,
    )

    assert fresh[0].action == "executed"
    assert fresh[0].resumed is False
    assert fresh[0].build_elapsed_ns is not None
    assert verified[0].action == "verified_skipped"
    assert verified[0].build_elapsed_ns is None


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

    def write_parquet(path: Path, value: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"value": [value]}), path)

    def file_builder(name: str) -> Any:
        def build(*arguments: Any) -> None:
            output = [
                argument
                for argument in arguments
                if isinstance(argument, Path) and argument.suffix == ".parquet"
            ][-1]
            write_parquet(output, name)
            calls.append(name)

        return build

    def validation(config: Path, output: Path) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(
                    {
                        "id": "purged-walk-forward-v1",
                        "purge_minutes": 65,
                        "maximum_information_window_minutes": 65,
                        "locked_test": {"start": "2025-01-01T00:00:00Z"},
                        "folds": [
                        {"id": f"wf-{2022 + index // 12}-{index % 12 + 1:02d}"}
                        for index in range(36)
                    ],
                }
            )
        )
        calls.append("validation")

    def isolated(
        features: Path, labels: Path, plan: object, output: Path
    ) -> None:
        output.mkdir(parents=True)
        write_parquet(output / "features.parquet", "features-development")
        write_parquet(output / "labels.parquet", "labels-development")
        calls.append("slice")

    def baseline(*arguments: Any) -> None:
        output = arguments[-1]
        output.mkdir(parents=True)
        symbol = output.parent.name
        start = datetime(2022, 1, 3, tzinfo=UTC)
        prediction_rows = []
        for fold_index in range(36):
            fold_id = f"wf-{2022 + fold_index // 12}-{fold_index % 12 + 1:02d}"
            for sample in range(4):
                decision = start + timedelta(minutes=5 * (fold_index * 4 + sample))
                for horizon in (15, 30, 60):
                    prediction_rows.append(
                        {
                            "model_set": "baseline-ridge-v1",
                            "validation_set": "purged-walk-forward-v1",
                            "fold_id": fold_id,
                            "symbol": symbol,
                            "decision_time": decision,
                            "entry_time": decision + timedelta(seconds=1),
                            "exit_time": decision
                            + timedelta(minutes=horizon, seconds=1),
                            "horizon_minutes": horizon,
                            "predicted_long_return": 0.0002,
                            "predicted_short_return": -0.0002,
                            "action": "long",
                            "realized_return": 0.0001,
                        }
                    )
        prediction_table = pa.Table.from_pylist(
            prediction_rows,
            schema=prediction_schema(load_baseline_config(MODEL_CONFIG)),
        )
        pq.write_table(prediction_table, output / "predictions.parquet")
        (output / "metrics.json").write_text(
            json.dumps(evaluate_predictions(prediction_table), sort_keys=True)
        )
        calls.append("baseline")

    def portfolio(
        prediction_paths: list[Path],
        portfolio_config: Path,
        validation_config: Path,
        output: Path,
    ) -> None:
        run_actual_portfolio_evaluation(
            prediction_paths, portfolio_config, validation_config, output
        )
        calls.append("portfolio")

    monkeypatch.setattr(development_module, "load_published_manifest", lambda *a: {})
    monkeypatch.setattr(
        development_module, "materialize_development_file", lambda *a: tmp_path
    )
    monkeypatch.setattr(
        development_module, "verify_materialized_inventory", lambda *a, **k: ()
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
    execution = json.loads((result.output / "execution-report.json").read_text())
    acceptance = json.loads(
        (
            result.output
            / "acceptance"
            / "development-acceptance-v1.json"
        ).read_text()
    )
    assert len(execution["stages"]) == 42
    assert {stage["action"] for stage in execution["stages"]} == {"executed"}
    checks = {check["id"]: check["status"] for check in acceptance["checks"]}
    assert checks["execution.profile"] == "pass"
    assert checks["model.metric_cells"] == "pass"
    assert checks["portfolio.full_recomputation"] == "pass"

    execution_path = result.output / "execution-report.json"
    malformed_execution = json.loads(execution_path.read_text())
    malformed_execution["stages"][0]["outputs"][0]["private_path"] = "/secret"
    execution_path.write_text(json.dumps(malformed_execution))
    with pytest.raises(RuntimeError, match="Execution profile is invalid"):
        evaluate_development_run(
            result.output,
            PROJECT_ROOT / "configs/experiments/development-acceptance-v1.toml",
        )
    execution_path.write_text(json.dumps(execution))

    portfolio_metrics = result.output / "portfolio" / "metrics.json"
    rejected_metrics = json.loads(portfolio_metrics.read_text())
    rejected_metrics["final_equity_usd"] = 90_000.0
    rejected_metrics["total_return"] = -0.1
    for dimension in rejected_metrics["attribution"].values():
        for row in dimension:
            row["pnl_usd"] = -10_000.0 / len(dimension)
    portfolio_metrics.write_text(json.dumps(rejected_metrics))
    rejected = evaluate_development_run(
        result.output,
        PROJECT_ROOT / "configs/experiments/development-acceptance-v1.toml",
    )

    assert rejected["summary"]["accepted"] is False
    assert rejected["summary"]["fail"] > acceptance["summary"]["fail"]

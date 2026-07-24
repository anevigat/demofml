import hashlib
import io
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import demofml.orchestration.development as development_module
import demofml.orchestration.locked as locked_module
from demofml.bars.quotes import QUOTE_BAR_SCHEMA
from demofml.data.publisher import build_manifest, manifest_bytes
from demofml.evaluation.signals import evaluate_locked_predictions
from demofml.features.causal import FEATURE_SCHEMA
from demofml.labels.executable import label_schema
from demofml.models.baseline import FEATURE_COLUMNS, load_baseline_config
from demofml.models.locked import (
    LOCKED_PREDICTION_SET_ID,
    LOCKED_SCORE_SET_ID,
    attach_locked_outcomes,
    score_locked_features,
    score_locked_test,
)
from demofml.orchestration.locked import (
    ONE_SHOT_POLICY,
    freeze_candidate,
    load_locked_test_config,
    run_locked_test_once,
    verify_candidate,
)
from demofml.validation.splits import load_validation_plan

PROJECT_ROOT = Path(__file__).parents[2]
PROTOCOL_CONFIG = PROJECT_ROOT / "configs/experiments/locked-test-evaluation-v1.toml"
MODEL_CONFIG = PROJECT_ROOT / "configs/experiments/baseline-ridge-v1.toml"
VALIDATION_CONFIG = PROJECT_ROOT / "configs/experiments/purged-walk-forward-v1.toml"
PIPELINE_CONFIG = PROJECT_ROOT / "configs/experiments/development-pipeline-v2.toml"
CODE_REFERENCE = "sha256:" + "d" * 64
SOURCE_CODE_REFERENCE = "sha256:" + "b" * 64
SOURCE_RUN_ID = development_module._run_id(
    PIPELINE_CONFIG,
    development_module.load_pipeline_config(PIPELINE_CONFIG),
    SOURCE_CODE_REFERENCE,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode()


def _bar(symbol: str, end: datetime, index: int) -> dict[str, object]:
    mid = 1.0 + index / 100_000.0
    spread = 0.0001
    return {
        "symbol": symbol,
        "bar_start": end - timedelta(minutes=5),
        "bar_end": end,
        "first_tick": end - timedelta(minutes=5),
        "last_tick": end - timedelta(seconds=1),
        "bid_open": mid - spread / 2,
        "bid_high": mid + 0.0001 - spread / 2,
        "bid_low": mid - 0.0001 - spread / 2,
        "bid_close": mid - spread / 2,
        "ask_open": mid + spread / 2,
        "ask_high": mid + 0.0001 + spread / 2,
        "ask_low": mid - 0.0001 + spread / 2,
        "ask_close": mid + spread / 2,
        "mid_open": mid,
        "mid_high": mid + 0.0001,
        "mid_low": mid - 0.0001,
        "mid_close": mid,
        "spread_open": spread,
        "spread_high": spread,
        "spread_low": spread,
        "spread_close": spread,
        "spread_mean": spread,
        "quote_count": 10,
        "staleness_ns": 1_000_000_000,
    }


def _development_tables(symbol: str) -> tuple[pa.Table, pa.Table]:
    start = datetime(2024, 12, 30, tzinfo=UTC)
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for index in range(30):
        decision = start + timedelta(minutes=5 * index)
        feature: dict[str, object] = {"symbol": symbol, "bar_end": decision}
        feature.update(
            {name: float(index + offset) for offset, name in enumerate(FEATURE_COLUMNS)}
        )
        feature_rows.append(feature)
        label: dict[str, object] = {
            "symbol": symbol,
            "decision_time": decision,
            "entry_time": decision,
            "entry_bid": 1.0,
            "entry_ask": 1.0001,
        }
        for horizon in (15, 30, 60):
            label[f"exit_time_{horizon}m"] = decision + timedelta(minutes=horizon)
            label[f"long_return_{horizon}m"] = (index + 1) / 100_000.0
            label[f"short_return_{horizon}m"] = -(index + 1) / 100_000.0
            label[f"action_{horizon}m"] = "long"
        label_rows.append(label)
    return (
        pa.Table.from_pylist(feature_rows, schema=FEATURE_SCHEMA),
        pa.Table.from_pylist(label_rows, schema=label_schema((15, 30, 60))),
    )


def _accepted_run(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    protocol = load_locked_test_config(PROTOCOL_CONFIG)
    root = tmp_path / SOURCE_RUN_ID
    acceptance = {
        "format_version": 1,
        "acceptance_set": protocol.source_acceptance_set,
        "development_only": True,
        "run_id": SOURCE_RUN_ID,
        "summary": {"pass": 20, "fail": 0, "blocked": 0, "accepted": True},
    }
    (root / "acceptance").mkdir(parents=True)
    (root / "acceptance" / "development-acceptance-v1.json").write_text(
        json.dumps(acceptance)
    )
    (root / "validation").mkdir()
    (root / "validation" / "manifest.json").write_text(
        json.dumps(
            {
                "id": "purged-walk-forward-v1",
                "locked_test": {"start": "2025-01-01T00:00:00Z"},
            }
        )
    )
    (root / "run.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "pipeline_set": protocol.source_pipeline_set,
                "run_id": SOURCE_RUN_ID,
                "code_reference": SOURCE_CODE_REFERENCE,
                "dataset_set": "cleaned-ticks-development-v1",
                "dataset_version": "sha256-" + "c" * 64,
                "symbols": list(protocol.symbols),
                "development_only": True,
                "acceptance_set": protocol.source_acceptance_set,
            }
        )
    )
    (root / "_SUCCESS").write_text(
        json.dumps({"run_id": SOURCE_RUN_ID, "mlflow_run_id": "run-1"})
    )
    context_end = protocol.locked_test_start
    for symbol in protocol.symbols:
        symbol_root = root / "symbols" / symbol
        development = symbol_root / "development"
        development.mkdir(parents=True)
        features, labels = _development_tables(symbol)
        pq.write_table(features, development / "features.parquet")
        pq.write_table(labels, development / "labels.parquet")
        bars = pa.Table.from_pylist(
            [
                _bar(
                    symbol,
                    context_end - timedelta(minutes=5 * (72 - index)),
                    index,
                )
                for index in range(73)
            ],
            schema=QUOTE_BAR_SCHEMA,
        )
        pq.write_table(bars, symbol_root / "bars.parquet")
        bars_path = symbol_root / "bars.parquet"
        features_path = development / "features.parquet"
        labels_path = development / "labels.parquet"

        def output_record(path: Path) -> dict[str, object]:
            return {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }

        (symbol_root / "bars.stage.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "fingerprint": "synthetic",
                    "outputs": [output_record(bars_path)],
                }
            )
        )
        (symbol_root / "development.stage.json").write_text(
            json.dumps(
                {
                    "format_version": 1,
                    "fingerprint": "synthetic",
                    "outputs": [
                        output_record(features_path),
                        output_record(labels_path),
                    ],
                }
            )
        )
    return root, acceptance


@pytest.fixture
def frozen_candidate(tmp_path: Path) -> Path:
    run, acceptance = _accepted_run(tmp_path)
    output = tmp_path / "candidate"
    freeze_candidate(
        run,
        PROTOCOL_CONFIG,
        output,
        CODE_REFERENCE,
        acceptance_evaluator=lambda run_root, config: acceptance,
    )
    return output


def test_phase_13_protocol_is_frozen_before_locked_access() -> None:
    config = load_locked_test_config(PROTOCOL_CONFIG)

    assert config.feature_context_bars == 73
    assert config.one_shot_policy == ONE_SHOT_POLICY
    assert config.symbols == (
        "AUDUSD",
        "EURCHF",
        "EURJPY",
        "EURUSD",
        "GBPJPY",
        "GBPUSD",
        "USDCAD",
        "USDJPY",
    )
    assert config.locked_test_start == datetime(2025, 1, 1, tzinfo=UTC)
    assert config.locked_test_end_exclusive == datetime(2026, 3, 11, tzinfo=UTC)


def test_freeze_publishes_24_models_and_detects_tampering(
    frozen_candidate: Path,
) -> None:
    manifest = verify_candidate(frozen_candidate, PROTOCOL_CONFIG)

    assert manifest["locked_data_accessed"] is False
    assert len(list((frozen_candidate / "models").rglob("*.json"))) == 24
    assert len(list((frozen_candidate / "feature-context").glob("*.parquet"))) == 8

    model = frozen_candidate / "models" / "EURUSD" / "15m.json"
    model.write_text(model.read_text().replace('"alpha": 1.0', '"alpha": 2.0'))
    with pytest.raises(RuntimeError, match="hashes differ"):
        verify_candidate(frozen_candidate, PROTOCOL_CONFIG)


def test_freeze_rejects_unaccepted_run(tmp_path: Path) -> None:
    run, acceptance = _accepted_run(tmp_path)
    acceptance["summary"]["accepted"] = False
    (run / "acceptance" / "development-acceptance-v1.json").write_text(
        json.dumps(acceptance)
    )

    with pytest.raises(RuntimeError, match="not immutably accepted"):
        freeze_candidate(
            run,
            PROTOCOL_CONFIG,
            tmp_path / "candidate",
            CODE_REFERENCE,
            acceptance_evaluator=lambda run_root, config: acceptance,
        )


def test_freeze_rejects_source_mutation_after_acceptance(tmp_path: Path) -> None:
    run, acceptance = _accepted_run(tmp_path)

    def mutate_after_acceptance(run_root: Path, config: Path) -> dict[str, Any]:
        del config
        source = run_root / "symbols" / "AUDUSD" / "development" / "features.parquet"
        source.write_bytes(source.read_bytes() + b"changed")
        return acceptance

    with pytest.raises(RuntimeError, match="changed during candidate freeze"):
        freeze_candidate(
            run,
            PROTOCOL_CONFIG,
            tmp_path / "candidate",
            CODE_REFERENCE,
            acceptance_evaluator=mutate_after_acceptance,
        )


def _locked_tables() -> tuple[pa.Table, pa.Table]:
    plan = load_validation_plan(VALIDATION_CONFIG)
    times = [
        plan.locked_test_start + timedelta(minutes=5 * index) for index in range(2)
    ]
    feature_rows = []
    label_rows = []
    for index, decision in enumerate(times):
        feature: dict[str, object] = {"symbol": "EURUSD", "bar_end": decision}
        feature.update({name: float(index + 1) for name in FEATURE_COLUMNS})
        feature_rows.append(feature)
        label: dict[str, object] = {
            "symbol": "EURUSD",
            "decision_time": decision,
            "entry_time": decision,
            "entry_bid": 1.0,
            "entry_ask": 1.0001,
        }
        for horizon in (15, 30, 60):
            label[f"exit_time_{horizon}m"] = decision + timedelta(minutes=horizon)
            label[f"long_return_{horizon}m"] = 0.001
            label[f"short_return_{horizon}m"] = -0.001
            label[f"action_{horizon}m"] = "long"
        label_rows.append(label)
    return (
        pa.Table.from_pylist(feature_rows, schema=FEATURE_SCHEMA),
        pa.Table.from_pylist(label_rows, schema=label_schema((15, 30, 60))),
    )


def test_locked_scoring_emits_all_blind_keys_before_outcome_filtering(
    frozen_candidate: Path,
) -> None:
    plan = load_validation_plan(VALIDATION_CONFIG)
    config = load_baseline_config(MODEL_CONFIG)
    features, labels = _locked_tables()
    label_rows = labels.to_pylist()
    label_rows[1]["exit_time_60m"] = None
    label_rows[1]["long_return_60m"] = None
    label_rows[1]["short_return_60m"] = None
    label_rows[1]["action_60m"] = None
    labels = pa.Table.from_pylist(label_rows, schema=label_schema((15, 30, 60)))
    candidate = json.loads((frozen_candidate / "candidate.json").read_text())

    result = score_locked_test(
        features,
        labels,
        plan,
        config,
        frozen_candidate,
        candidate["candidate_id"],
        "EURUSD",
    )

    assert result.scores.num_rows == 6
    assert result.evaluated_predictions.num_rows == 6
    assert result.unresolved_executions == 1
    assert result.scores.schema.metadata[b"demofml.score_set"] == (
        LOCKED_SCORE_SET_ID.encode()
    )
    assert (
        result.evaluated_predictions.schema.metadata[b"demofml.prediction_set"]
        == LOCKED_PREDICTION_SET_ID.encode()
    )
    report = evaluate_locked_predictions(
        result.evaluated_predictions, candidate["candidate_id"]
    )
    assert report["locked_test"] is True
    assert len(report["symbols"]) == 3

    with pytest.raises(ValueError, match="candidate identity"):
        evaluate_locked_predictions(result.evaluated_predictions, "sha256-wrong")
    with pytest.raises(ValueError, match="schema is missing"):
        evaluate_locked_predictions(
            result.evaluated_predictions.drop(["action"]), candidate["candidate_id"]
        )
    with pytest.raises(ValueError, match="locked prediction set"):
        evaluate_locked_predictions(
            result.evaluated_predictions.replace_schema_metadata({}),
            candidate["candidate_id"],
        )
    with pytest.raises(ValueError, match="empty locked predictions"):
        evaluate_locked_predictions(
            result.evaluated_predictions.slice(0, 0), candidate["candidate_id"]
        )


def test_locked_scoring_rejects_invalid_keys_features_and_execution(
    frozen_candidate: Path,
) -> None:
    plan = load_validation_plan(VALIDATION_CONFIG)
    config = load_baseline_config(MODEL_CONFIG)
    candidate = json.loads((frozen_candidate / "candidate.json").read_text())
    features, labels = _locked_tables()

    with pytest.raises(ValueError, match="rows must match"):
        score_locked_test(
            features.slice(0, 1),
            labels,
            plan,
            config,
            frozen_candidate,
            candidate["candidate_id"],
            "EURUSD",
        )
    with pytest.raises(ValueError, match="schema is missing"):
        score_locked_test(
            features.drop(["mid_return_1"]),
            labels,
            plan,
            config,
            frozen_candidate,
            candidate["candidate_id"],
            "EURUSD",
        )
    feature_rows = features.to_pylist()
    feature_rows[0]["mid_return_1"] = float("inf")
    infinite = pa.Table.from_pylist(feature_rows, schema=FEATURE_SCHEMA)
    with pytest.raises(ValueError, match="contains infinity"):
        score_locked_test(
            infinite,
            labels,
            plan,
            config,
            frozen_candidate,
            candidate["candidate_id"],
            "EURUSD",
        )

    label_rows = labels.to_pylist()
    label_rows[0]["symbol"] = "GBPUSD"
    wrong_symbol = pa.Table.from_pylist(label_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="symbols are not aligned"):
        score_locked_test(
            features,
            wrong_symbol,
            plan,
            config,
            frozen_candidate,
            candidate["candidate_id"],
            "EURUSD",
        )

    label_rows = labels.to_pylist()
    label_rows[0]["decision_time"] += timedelta(minutes=1)
    wrong_time = pa.Table.from_pylist(label_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="decisions are not aligned"):
        score_locked_test(
            features,
            wrong_time,
            plan,
            config,
            frozen_candidate,
            candidate["candidate_id"],
            "EURUSD",
        )

    before_rows = features.to_pylist()
    before_label_rows = labels.to_pylist()
    for feature, label in zip(before_rows, before_label_rows, strict=True):
        feature["bar_end"] = plan.locked_test_start - timedelta(minutes=10)
        label["decision_time"] = plan.locked_test_start - timedelta(minutes=10)
    before_rows[1]["bar_end"] += timedelta(minutes=5)
    before_label_rows[1]["decision_time"] += timedelta(minutes=5)
    before = pa.Table.from_pylist(before_rows, schema=FEATURE_SCHEMA)
    before_labels = pa.Table.from_pylist(before_label_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="no eligible"):
        score_locked_test(
            before,
            before_labels,
            plan,
            config,
            frozen_candidate,
            candidate["candidate_id"],
            "EURUSD",
        )

    for field, value, message in (
        (
            "entry_time",
            plan.locked_test_start + timedelta(minutes=6),
            "entry time",
        ),
        (
            "exit_time_15m",
            plan.locked_test_start + timedelta(minutes=30),
            "exit time",
        ),
    ):
        changed_rows = labels.to_pylist()
        changed_rows[0][field] = value
        changed = pa.Table.from_pylist(changed_rows, schema=labels.schema)
        with pytest.raises(RuntimeError, match=message):
            score_locked_test(
                features,
                changed,
                plan,
                config,
                frozen_candidate,
                candidate["candidate_id"],
                "EURUSD",
            )


def test_separate_scoring_phases_reject_tampered_contracts(
    frozen_candidate: Path,
) -> None:
    plan = load_validation_plan(VALIDATION_CONFIG)
    config = load_baseline_config(MODEL_CONFIG)
    candidate = json.loads((frozen_candidate / "candidate.json").read_text())
    candidate_id = candidate["candidate_id"]
    features, labels = _locked_tables()

    with pytest.raises(ValueError, match="non-empty"):
        score_locked_features(
            features.slice(0, 0),
            plan,
            config,
            frozen_candidate,
            candidate_id,
            "EURUSD",
        )
    with pytest.raises(ValueError, match="validation plan"):
        score_locked_features(
            features.replace_schema_metadata({}),
            plan,
            config,
            frozen_candidate,
            candidate_id,
            "EURUSD",
        )
    with pytest.raises(ValueError, match="requested symbol"):
        score_locked_features(
            features,
            plan,
            config,
            frozen_candidate,
            candidate_id,
            "GBPUSD",
        )
    field_index = features.schema.get_field_index("mid_return_1")
    wrong_type = features.set_column(
        field_index,
        pa.field("mid_return_1", pa.float32()),
        pa.array(features.column("mid_return_1").to_pylist(), type=pa.float32()),
    )
    with pytest.raises(ValueError, match="field mid_return_1 differs"):
        score_locked_features(
            wrong_type,
            plan,
            config,
            frozen_candidate,
            candidate_id,
            "EURUSD",
        )

    scores = score_locked_features(
        features, plan, config, frozen_candidate, candidate_id, "EURUSD"
    )
    with pytest.raises(ValueError, match="score provenance"):
        attach_locked_outcomes(
            scores.replace_schema_metadata({}),
            labels,
            plan,
            config,
            candidate_id,
            "EURUSD",
        )
    wrong_label_rows = labels.to_pylist()
    wrong_label_rows[0]["symbol"] = "GBPUSD"
    wrong_labels = pa.Table.from_pylist(wrong_label_rows, schema=labels.schema)
    with pytest.raises(ValueError, match="requested symbol"):
        attach_locked_outcomes(
            scores, wrong_labels, plan, config, candidate_id, "EURUSD"
        )
    with pytest.raises(ValueError, match="do not match eligible"):
        attach_locked_outcomes(
            scores.slice(1), labels, plan, config, candidate_id, "EURUSD"
        )
    duplicated = pa.concat_tables([scores, scores.slice(0, 1)])
    with pytest.raises(ValueError, match="duplicated"):
        attach_locked_outcomes(
            duplicated, labels, plan, config, candidate_id, "EURUSD"
        )

    score_rows = scores.to_pylist()
    score_rows[0]["action"] = "flat" if score_rows[0]["action"] != "flat" else "long"
    changed_action = pa.Table.from_pylist(score_rows, schema=scores.schema)
    with pytest.raises(ValueError, match="action differs"):
        attach_locked_outcomes(
            changed_action, labels, plan, config, candidate_id, "EURUSD"
        )
    score_rows = scores.to_pylist()
    score_rows[0]["predicted_long_return"] = float("inf")
    nonfinite = pa.Table.from_pylist(score_rows, schema=scores.schema)
    with pytest.raises(ValueError, match="predictions must be finite"):
        attach_locked_outcomes(
            nonfinite, labels, plan, config, candidate_id, "EURUSD"
        )


def _locked_dataset_config(path: Path) -> None:
    symbols = load_locked_test_config(PROTOCOL_CONFIG).symbols
    rows = [
        "format_version = 1",
        'id = "cleaned-ticks-locked-test-v1"',
        'dataset_name = "cleaned_ticks"',
        f'dataset_version = "sha256-{"e" * 64}"',
        's3_prefix = "datasets/cleaned_ticks"',
        'start = "2025-01-01T00:00:00Z"',
        'end_exclusive = "2026-03-11T00:00:00Z"',
    ]
    for index, symbol in enumerate(symbols):
        rows.extend(
            [
                "",
                "[[files]]",
                f'symbol = "{symbol}"',
                f'path = "{symbol}/2025/part-{index:02d}.parquet"',
                f'sha256 = "{index + 1:064x}"',
                "size_bytes = 1",
                "rows = 1",
            ]
        )
    path.write_text("\n".join(rows) + "\n")


def _grant(
    path: Path, candidate: Path, dataset_config: Path, code_reference: str
) -> None:
    manifest = json.loads((candidate / "candidate.json").read_text())
    core = {
        "format_version": 1,
        "test_set": "locked-test-evaluation-v1",
        "candidate_id": manifest["candidate_id"],
        "candidate_manifest_sha256": hashlib.sha256(
            (candidate / "candidate.json").read_bytes()
        ).hexdigest(),
        "protocol_config_sha256": hashlib.sha256(
            PROTOCOL_CONFIG.read_bytes()
        ).hexdigest(),
        "locked_dataset_config_sha256": hashlib.sha256(
            dataset_config.read_bytes()
        ).hexdigest(),
        "code_reference": code_reference,
        "one_shot_policy": ONE_SHOT_POLICY,
        "authorized_at": "2026-07-24T00:00:00Z",
    }
    grant_id = f"sha256-{hashlib.sha256(_canonical(core)).hexdigest()}"
    path.write_text(json.dumps({**core, "grant_id": grant_id}))


def test_one_shot_is_consumed_before_s3_and_failure_cannot_retry(
    tmp_path: Path, frozen_candidate: Path
) -> None:
    dataset = tmp_path / "locked-dataset.toml"
    grant = tmp_path / "grant.json"
    workdir = tmp_path / "work"
    _locked_dataset_config(dataset)
    _grant(grant, frozen_candidate, dataset, CODE_REFERENCE)
    factory_calls = 0

    def fail_after_marker(endpoint: str, region: str) -> object:
        nonlocal factory_calls
        factory_calls += 1
        assert endpoint == "https://s3.invalid"
        assert region == "us-east-1"
        assert (
            workdir / "locked-test-evaluation-v1" / "_LOCKED_TEST_STARTED.json"
        ).is_file()
        raise RuntimeError("injected infrastructure failure")

    with pytest.raises(RuntimeError, match="injected infrastructure failure"):
        run_locked_test_once(
            frozen_candidate,
            PROTOCOL_CONFIG,
            dataset,
            grant,
            workdir,
            CODE_REFERENCE,
            "locked-data",
            "https://s3.invalid",
            "us-east-1",
            s3_factory=fail_after_marker,
        )

    root = workdir / "locked-test-evaluation-v1"
    assert factory_calls == 1
    assert (root / "_FAILED.json").is_file()
    with pytest.raises(RuntimeError, match="already been consumed"):
        run_locked_test_once(
            frozen_candidate,
            PROTOCOL_CONFIG,
            dataset,
            grant,
            workdir,
            CODE_REFERENCE,
            "locked-data",
            "https://s3.invalid",
            "us-east-1",
            s3_factory=fail_after_marker,
        )
    assert factory_calls == 1


def test_valid_pre_marker_snapshot_can_be_reused(
    tmp_path: Path, frozen_candidate: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "locked-dataset.toml"
    grant = tmp_path / "grant.json"
    workdir = tmp_path / "work"
    _locked_dataset_config(dataset)
    _grant(grant, frozen_candidate, dataset, CODE_REFERENCE)
    actual_validate = locked_module._validate_grant
    validations = 0

    def fail_second_validation(*arguments: Any, **keywords: Any) -> None:
        nonlocal validations
        validations += 1
        actual_validate(*arguments, **keywords)
        if validations == 2:
            raise RuntimeError("injected pre-marker failure")

    monkeypatch.setattr(locked_module, "_validate_grant", fail_second_validation)
    with pytest.raises(RuntimeError, match="injected pre-marker failure"):
        run_locked_test_once(
            frozen_candidate,
            PROTOCOL_CONFIG,
            dataset,
            grant,
            workdir,
            CODE_REFERENCE,
            "locked-data",
            "https://s3.invalid",
            "us-east-1",
        )
    root = workdir / "locked-test-evaluation-v1"
    assert (root / "candidate-snapshot").is_dir()
    assert not (root / "_LOCKED_TEST_STARTED.json").exists()

    monkeypatch.setattr(locked_module, "_validate_grant", actual_validate)
    with pytest.raises(RuntimeError, match="after reused snapshot"):
        run_locked_test_once(
            frozen_candidate,
            PROTOCOL_CONFIG,
            dataset,
            grant,
            workdir,
            CODE_REFERENCE,
            "locked-data",
            "https://s3.invalid",
            "us-east-1",
            s3_factory=lambda endpoint, region: (_ for _ in ()).throw(
                RuntimeError("after reused snapshot")
            ),
        )
    assert (root / "_LOCKED_TEST_STARTED.json").is_file()


class _S3:
    def __init__(self, objects: dict[str, tuple[bytes, dict[str, str]]]) -> None:
        self.objects = objects

    def head_object(self, **arguments: Any) -> dict[str, object]:
        payload, metadata = self.objects[arguments["Key"]]
        return {
            "ContentLength": len(payload),
            "Metadata": metadata,
            "ETag": '"immutable"',
        }

    def get_object(self, **arguments: Any) -> dict[str, io.BytesIO]:
        payload, _ = self.objects[arguments["Key"]]
        return {"Body": io.BytesIO(payload)}


def _real_locked_dataset(tmp_path: Path) -> tuple[Path, _S3]:
    source = tmp_path / "locked-source"
    protocol = load_locked_test_config(PROTOCOL_CONFIG)
    for symbol_index, symbol in enumerate(protocol.symbols):
        first_tick = protocol.locked_test_end_exclusive - timedelta(minutes=125)
        timestamps = [first_tick + timedelta(minutes=5 * index) for index in range(25)]
        mids = [1.0 + symbol_index / 100.0 + index / 100_000.0 for index in range(25)]
        path = source / symbol / "2025" / "part-00.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(
            pa.table(
                {
                    "timestamp": pa.array(
                        timestamps, type=pa.timestamp("ns", tz="UTC")
                    ),
                    "bid": pa.array(
                        [value - 0.00005 for value in mids], type=pa.float64()
                    ),
                    "ask": pa.array(
                        [value + 0.00005 for value in mids], type=pa.float64()
                    ),
                    "mid": pa.array(mids, type=pa.float64()),
                    "spread": pa.array([0.0001] * len(mids), type=pa.float64()),
                }
            ),
            path,
            write_statistics=True,
        )
    manifest = build_manifest(source, "cleaned_ticks")
    config = tmp_path / "real-locked-dataset.toml"
    lines = [
        "format_version = 1",
        'id = "cleaned-ticks-locked-test-v1"',
        'dataset_name = "cleaned_ticks"',
        f'dataset_version = "{manifest["dataset_version"]}"',
        's3_prefix = "locked"',
        'start = "2025-01-01T00:00:00Z"',
        'end_exclusive = "2026-03-11T00:00:00Z"',
    ]
    objects: dict[str, tuple[bytes, dict[str, str]]] = {}
    root = f"locked/{manifest['dataset_version']}"
    objects[f"{root}/manifest.json"] = (manifest_bytes(manifest), {})
    for entry in manifest["files"]:
        lines.extend(
            [
                "",
                "[[files]]",
                f'symbol = "{Path(entry["path"]).parts[0]}"',
                f'path = "{entry["path"]}"',
                f'sha256 = "{entry["sha256"]}"',
                f"size_bytes = {entry['size_bytes']}",
                f"rows = {entry['rows']}",
            ]
        )
        payload = (source / entry["path"]).read_bytes()
        objects[f"{root}/data/{entry['path']}"] = (
            payload,
            {
                "sha256": entry["sha256"],
                "dataset-version": manifest["dataset_version"],
            },
        )
    config.write_text("\n".join(lines) + "\n")
    return config, _S3(objects)


def test_complete_locked_compute_publishes_terminal_rejection(
    tmp_path: Path, frozen_candidate: Path
) -> None:
    dataset, client = _real_locked_dataset(tmp_path)
    grant = tmp_path / "real-grant.json"
    _grant(grant, frozen_candidate, dataset, CODE_REFERENCE)

    result = run_locked_test_once(
        frozen_candidate,
        PROTOCOL_CONFIG,
        dataset,
        grant,
        tmp_path / "real-work",
        CODE_REFERENCE,
        "locked-data",
        "https://s3.invalid",
        "us-east-1",
        s3_factory=lambda endpoint, region: client,
    )

    assert result.accepted is False
    assert (result.output / "artifact-manifest.json").is_file()
    assert (result.output / "locked-test-report.json").is_file()
    assert not (result.output.parent / "private").exists()
    report = json.loads((result.output / "locked-test-report.json").read_text())
    checks = {check["id"]: check["status"] for check in report["checks"]}
    assert checks["inference.all_keys_scored"] == "pass"
    assert checks["outcomes.complete_execution_cells"] == "fail"
    assert checks["portfolio.full_recomputation"] == "pass"
    assert report["summary"]["fail"] > 0

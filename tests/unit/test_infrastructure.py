from types import SimpleNamespace
from typing import Any

import pytest

from demofml import infrastructure


class _Body:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload


class _S3:
    def __init__(self, corrupt: bool = False) -> None:
        self.payload = b""
        self.corrupt = corrupt
        self.deleted: list[tuple[str, str]] = []

    def put_object(self, **arguments: Any) -> None:
        self.payload = arguments["Body"]

    def get_object(self, **arguments: Any) -> dict[str, _Body]:
        payload = b"corrupt" if self.corrupt else self.payload
        return {"Body": _Body(payload)}

    def delete_object(self, **arguments: Any) -> None:
        self.deleted.append((arguments["Bucket"], arguments["Key"]))


class _Mlflow:
    def __init__(self, experiment_exists: bool, fail: bool = False) -> None:
        self.experiment_exists = experiment_exists
        self.fail = fail
        self.terminated: list[tuple[str, str]] = []
        self.artifact_paths: list[str] = []

    def get_experiment_by_name(self, name: str) -> object | None:
        return (
            SimpleNamespace(experiment_id="existing")
            if self.experiment_exists
            else None
        )

    def create_experiment(self, name: str) -> str:
        return "created"

    def create_run(self, experiment_id: str, tags: dict[str, str]) -> object:
        return SimpleNamespace(info=SimpleNamespace(run_id="run-1"))

    def log_param(self, run_id: str, name: str, value: str) -> None:
        pass

    def log_metric(self, run_id: str, name: str, value: float) -> None:
        if self.fail:
            raise RuntimeError("metric failure")

    def log_artifact(self, run_id: str, path: str) -> None:
        self.artifact_paths.append(path)

    def set_terminated(self, run_id: str, status: str) -> None:
        self.terminated.append((run_id, status))


def _configure(
    monkeypatch: pytest.MonkeyPatch,
    s3: _S3,
    mlflow: _Mlflow,
) -> None:
    monkeypatch.setenv("S3_ENDPOINT_URL", "https://s3.invalid")
    monkeypatch.setenv("DEMOFML_DATA_BUCKET", "data")
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.invalid")
    monkeypatch.setattr(
        "demofml.infrastructure.boto3.client", lambda *args, **kwargs: s3
    )
    monkeypatch.setattr(infrastructure, "MlflowClient", lambda **kwargs: mlflow)
    monkeypatch.setattr(
        "demofml.infrastructure.uuid.uuid4",
        lambda: SimpleNamespace(hex="smoke-id"),
    )


@pytest.mark.parametrize("experiment_exists", [False, True])
def test_infrastructure_smoke_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    experiment_exists: bool,
) -> None:
    s3 = _S3()
    mlflow = _Mlflow(experiment_exists)
    _configure(monkeypatch, s3, mlflow)

    infrastructure.run_infrastructure_smoke()

    assert mlflow.terminated == [("run-1", "FINISHED")]
    assert len(mlflow.artifact_paths) == 1
    assert s3.deleted == [("data", "infrastructure-smoke/smoke-id.txt")]
    assert capsys.readouterr().out == "infrastructure smoke passed: run_id=run-1\n"


def test_infrastructure_smoke_marks_failed_and_cleans_s3(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3()
    mlflow = _Mlflow(False, fail=True)
    _configure(monkeypatch, s3, mlflow)

    with pytest.raises(RuntimeError, match="metric failure"):
        infrastructure.run_infrastructure_smoke()

    assert mlflow.terminated == [("run-1", "FAILED")]
    assert s3.deleted == [("data", "infrastructure-smoke/smoke-id.txt")]


def test_infrastructure_smoke_rejects_corrupt_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    s3 = _S3(corrupt=True)
    _configure(monkeypatch, s3, _Mlflow(False))

    with pytest.raises(RuntimeError, match="round-trip"):
        infrastructure.run_infrastructure_smoke()


def test_infrastructure_smoke_requires_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("S3_ENDPOINT_URL", raising=False)
    with pytest.raises(RuntimeError, match="S3_ENDPOINT_URL"):
        infrastructure.run_infrastructure_smoke()

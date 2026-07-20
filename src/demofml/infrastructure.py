"""Infrastructure connectivity checks used by Kubernetes smoke Jobs."""

import os
import tempfile
import uuid
from pathlib import Path

import boto3  # type: ignore[import-untyped]
from mlflow import MlflowClient


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable is missing: {name}")
    return value


def run_infrastructure_smoke() -> None:
    """Verify S3 object I/O plus MLflow metrics and artifact persistence."""
    s3_endpoint = _required_environment("S3_ENDPOINT_URL")
    data_bucket = _required_environment("DEMOFML_DATA_BUCKET")
    tracking_uri = _required_environment("MLFLOW_TRACKING_URI")
    smoke_id = uuid.uuid4().hex
    object_key = f"infrastructure-smoke/{smoke_id}.txt"
    payload = f"demofml infrastructure smoke {smoke_id}\n".encode()

    s3 = boto3.client("s3", endpoint_url=s3_endpoint)
    s3.put_object(Bucket=data_bucket, Key=object_key, Body=payload)
    stored_payload = s3.get_object(Bucket=data_bucket, Key=object_key)["Body"].read()
    if stored_payload != payload:
        raise RuntimeError("S3 smoke object content did not round-trip correctly")

    client = MlflowClient(tracking_uri=tracking_uri)
    experiment_name = "infrastructure-smoke"
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(experiment_name)
    )
    run = client.create_run(
        experiment_id,
        tags={"component": "phase-4", "smoke_id": smoke_id},
    )

    try:
        client.log_param(run.info.run_id, "s3_bucket", data_bucket)
        client.log_metric(run.info.run_id, "connectivity", 1.0)
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "smoke.txt"
            artifact.write_bytes(payload)
            client.log_artifact(run.info.run_id, str(artifact))
        client.set_terminated(run.info.run_id, status="FINISHED")
    except Exception:
        client.set_terminated(run.info.run_id, status="FAILED")
        raise
    finally:
        s3.delete_object(Bucket=data_bucket, Key=object_key)

    print(f"infrastructure smoke passed: run_id={run.info.run_id}")

"""Atomic isolation of aligned development research rows."""

import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.validation.splits import (
    ValidationPlan,
    validate_feature_label_schemas,
)


@dataclass(frozen=True)
class DevelopmentSliceResult:
    """Summary of an aligned feature/label development slice."""

    input_rows: int
    output_rows: int
    first_decision: datetime
    last_decision: datetime


def _validate_alignment(features: pa.Table, labels: pa.Table) -> None:
    if features.num_rows == 0 or features.num_rows != labels.num_rows:
        raise ValueError("feature and label row counts must match and be non-zero")
    if not features.column("symbol").equals(labels.column("symbol")):
        raise ValueError("feature and label symbols are not aligned")
    if not features.column("bar_end").equals(labels.column("decision_time")):
        raise ValueError("feature and label decision times are not aligned")


def isolate_development_rows(
    features_path: Path,
    labels_path: Path,
    plan: ValidationPlan,
    output: Path,
) -> DevelopmentSliceResult:
    """Publish aligned rows whose complete information window predates the lock."""
    features_path = features_path.expanduser().resolve()
    labels_path = labels_path.expanduser().resolve()
    output = output.expanduser().resolve()
    if not features_path.is_file() or not labels_path.is_file():
        raise RuntimeError("development slice inputs must be files")
    if output.exists():
        raise RuntimeError(f"Refusing to replace development slice: {output}")

    features = pq.read_table(features_path)
    labels = pq.read_table(labels_path)
    validate_feature_label_schemas(features.schema, labels.schema, plan)
    _validate_alignment(features, labels)
    decision_times = features.column("bar_end")
    mask = pc.and_(
        pc.greater_equal(decision_times, pa.scalar(plan.train_start)),
        pc.less(decision_times, pa.scalar(plan.development_decision_end)),
    )
    selected_features = features.filter(mask)
    selected_labels = labels.filter(mask)
    _validate_alignment(selected_features, selected_labels)
    selected_times = selected_features.column("bar_end").to_pylist()
    first = selected_times[0]
    last = selected_times[-1]
    if not isinstance(first, datetime) or not isinstance(last, datetime):
        raise ValueError("development decision times must be timestamps")
    if first < plan.train_start or last >= plan.development_decision_end:
        raise ValueError("development slice escaped its permitted interval")

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    partial.mkdir()
    try:
        pq.write_table(
            selected_features,
            partial / "features.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        pq.write_table(
            selected_labels,
            partial / "labels.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        if output.exists():
            raise RuntimeError(f"Development slice appeared during build: {output}")
        os.rename(partial, output)
    except FileExistsError as error:
        raise RuntimeError(
            f"Development slice appeared during build: {output}"
        ) from error
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return DevelopmentSliceResult(
        features.num_rows, selected_features.num_rows, first, last
    )

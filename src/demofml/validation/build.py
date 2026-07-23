"""Build deterministic validation manifests from versioned fold definitions."""

import argparse
import json
import os
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from demofml.validation.splits import ValidationPlan, load_validation_plan


@dataclass(frozen=True)
class ValidationBuildResult:
    """Summary of one validation-manifest build."""

    fold_count: int
    validation_set: str


def _timestamp(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def validation_manifest(plan: ValidationPlan) -> dict[str, Any]:
    """Render a stable JSON-serializable validation manifest."""
    folds = plan.folds()
    return {
        "format_version": 1,
        "id": plan.id,
        "strategy": plan.strategy,
        "interval_semantics": plan.interval_semantics,
        "feature_set": plan.feature_set,
        "label_set": plan.label_set,
        "purge_minutes": plan.purge_minutes,
        "maximum_information_window_minutes": int(
            plan.information_window.total_seconds() // 60
        ),
        "locked_test": {
            "start": _timestamp(plan.locked_test_start),
            "data_end_exclusive": _timestamp(plan.locked_test_end_exclusive),
            "decision_end_exclusive": _timestamp(plan.locked_test_decision_end),
        },
        "folds": [
            {
                "id": fold.id,
                "train_start": _timestamp(fold.train_start),
                "train_end_exclusive": _timestamp(fold.train_end_exclusive),
                "validation_start": _timestamp(fold.validation_start),
                "validation_end_exclusive": _timestamp(fold.validation_end_exclusive),
            }
            for fold in folds
        ],
    }


def build_validation_manifest(config: Path, output: Path) -> ValidationBuildResult:
    """Validate a fold config and publish its manifest atomically."""
    output = output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace validation manifest: {output}")
    plan = load_validation_plan(config)
    manifest = validation_manifest(plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    try:
        partial.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.link(partial, output)
    except FileExistsError as error:
        raise RuntimeError(
            f"Validation manifest appeared during build: {output}"
        ) from error
    finally:
        partial.unlink(missing_ok=True)
    return ValidationBuildResult(len(manifest["folds"]), plan.id)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a purged walk-forward validation manifest."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the validation-manifest command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = build_validation_manifest(arguments.config, arguments.output)
        print(
            f"built {result.fold_count} folds for {result.validation_set}: "
            f"{arguments.output}"
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

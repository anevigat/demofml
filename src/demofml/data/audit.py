"""CLI support for bounded-memory tick quality audits."""

import argparse
import json
import os
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.data.ticks import (
    TickContractError,
    TickQualityReport,
    audit_parquet_file,
)


def _stream_name(source: Path, relative: Path) -> str:
    if len(source.name) == 6 and source.name.isupper():
        return source.name
    first = relative.parts[0]
    return first if len(first) == 6 and first.isupper() else source.name


def audit_dataset(
    source: Path,
    max_row_groups_per_file: int | None,
) -> dict[str, Any]:
    """Audit every Parquet file below a source and return a JSON report."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise RuntimeError(f"Audit source is not a directory: {source}")
    paths = sorted(
        (path for path in source.rglob("*.parquet") if path.is_file()),
        key=lambda path: path.relative_to(source).as_posix(),
    )
    if not paths:
        raise RuntimeError(f"No Parquet files found below: {source}")

    files: list[dict[str, Any]] = []
    schema_errors: list[dict[str, str]] = []
    stream_reports: dict[str, TickQualityReport] = {}
    for index, path in enumerate(paths, start=1):
        relative_path = path.relative_to(source)
        relative = relative_path.as_posix()
        stream = _stream_name(source, relative_path)
        print(f"[{index}/{len(paths)}] auditing {relative}", flush=True)
        metadata = pq.read_metadata(path)
        scanned = (
            metadata.num_row_groups
            if max_row_groups_per_file is None
            else min(max_row_groups_per_file, metadata.num_row_groups)
        )
        try:
            quality = audit_parquet_file(
                path,
                max_row_groups_per_file,
                stream_reports.setdefault(stream, TickQualityReport()),
            )
        except TickContractError as error:
            schema_errors.append({"path": relative, "error": str(error)})
            continue
        files.append(
            {
                "path": relative,
                "stream": stream,
                "row_groups_scanned": scanned,
                "row_groups_total": metadata.num_row_groups,
                "stream_quality_after_file": quality.as_dict(),
            }
        )

    total_rows = sum(report.rows for report in stream_reports.values())
    total_violations = sum(
        report.critical_violations for report in stream_reports.values()
    )
    return {
        "format_version": 1,
        "source_name": source.name,
        "complete": max_row_groups_per_file is None,
        "file_count": len(paths),
        "audited_file_count": len(files),
        "rows_scanned": total_rows,
        "critical_violations": total_violations,
        "schema_errors": schema_errors,
        "streams": {
            name: report.as_dict() for name, report in sorted(stream_reports.items())
        },
        "files": files,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the canonical tick contract and quality invariants."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--max-row-groups-per-file",
        type=int,
        default=1,
        help="Use 0 to scan every row group; defaults to a lightweight sample.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/quality/tick-audit.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the tick audit command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.max_row_groups_per_file < 0:
        parser.error("--max-row-groups-per-file cannot be negative")
    limit = arguments.max_row_groups_per_file or None
    try:
        report = audit_dataset(arguments.source, limit)
        output: Path = arguments.output
        output.parent.mkdir(parents=True, exist_ok=True)
        partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
        partial.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        os.replace(partial, output)
        print(f"report: {output}")
        print(
            f"files: {report['audited_file_count']}/{report['file_count']}; "
            f"rows: {report['rows_scanned']}; "
            f"critical violations: {report['critical_violations']}"
        )
        if report["schema_errors"] or report["critical_violations"]:
            parser.exit(1, "tick audit failed\n")
    except (OSError, RuntimeError) as error:
        parser.exit(1, f"error: {error}\n")

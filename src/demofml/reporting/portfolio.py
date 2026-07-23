"""Reproducible portfolio reports and atomic artifact publication."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from demofml.evaluation.portfolio import (
    PortfolioConfig,
    PortfolioSimulation,
    load_portfolio_config,
    simulate_portfolio,
)
from demofml.validation.splits import load_validation_plan


@dataclass(frozen=True)
class PortfolioBuildResult:
    """Summary of one atomically published portfolio evaluation."""

    trades: int
    equity_events: int
    final_equity_usd: float


def _attribution(ledger: pa.Table, field: str) -> list[dict[str, Any]]:
    if ledger.num_rows == 0:
        return []
    winners = pc.cast(pc.greater(ledger.column("pnl_usd"), 0.0), pa.int64())
    source = ledger.select([field, "pnl_usd"]).append_column("_winner", winners)
    grouped = source.group_by(field).aggregate(
        [("pnl_usd", "count"), ("pnl_usd", "sum"), ("_winner", "sum")]
    )
    result: list[dict[str, Any]] = []
    for row in sorted(grouped.to_pylist(), key=lambda value: str(value[field])):
        key = row[field]
        trades = int(row["pnl_usd_count"])
        result.append(
            {
                field: key,
                "trades": trades,
                "pnl_usd": float(row["pnl_usd_sum"]),
                "win_rate": int(row["_winner_sum"]) / trades,
            }
        )
    return result


def portfolio_report(
    simulation: PortfolioSimulation,
    config: PortfolioConfig,
) -> dict[str, Any]:
    """Build deterministic portfolio metrics without claiming locked performance."""
    equity_rows = simulation.equity.to_pylist()
    if not equity_rows:
        raise ValueError("portfolio equity path cannot be empty")
    ledger = simulation.ledger
    final_equity = float(equity_rows[-1]["equity_usd"])
    drawdowns = [float(row["drawdown"]) for row in equity_rows]
    period_returns = np.asarray(simulation.period_returns, dtype=float)
    realized_volatility = (
        float(np.std(period_returns, ddof=1)) * math.sqrt(config.annualization_periods)
        if period_returns.size >= 2
        else 0.0
    )
    return {
        "format_version": 1,
        "portfolio_set": config.id,
        "prediction_set": config.prediction_set,
        "model_set": config.model_set,
        "validation_set": config.validation_set,
        "development_only": True,
        "initial_capital_usd": config.initial_capital_usd,
        "final_equity_usd": final_equity,
        "total_return": final_equity / config.initial_capital_usd - 1.0,
        "trades": ledger.num_rows,
        "suppressed_signals": simulation.suppressed_signals,
        "maximum_active_positions": simulation.maximum_active_positions,
        "maximum_gross_leverage": simulation.maximum_gross_leverage,
        "realized_annual_volatility": realized_volatility,
        "target_annual_volatility": config.target_annual_volatility,
        "maximum_drawdown": max(drawdowns),
        "drawdown_limit": config.maximum_drawdown,
        "drawdown_halt_triggered": any(bool(row["halted"]) for row in equity_rows),
        "attribution": {
            "symbols": _attribution(ledger, "symbol"),
            "horizons": _attribution(ledger, "horizon_minutes"),
            "folds": _attribution(ledger, "fold_id"),
        },
        "always_flat_comparator": {
            "final_equity_usd": config.initial_capital_usd,
            "total_return": 0.0,
            "maximum_drawdown": 0.0,
        },
        "accounting": config.return_accounting,
        "pnl_recognition": config.pnl_recognition,
        "interpretation": (
            "normalized_return_sleeves_with_exit_time_pnl_no_intratrade_mark_to_market"
        ),
    }


def run_portfolio_evaluation(
    prediction_paths: Sequence[Path],
    portfolio_config_path: Path,
    validation_config_path: Path,
    output: Path,
) -> PortfolioBuildResult:
    """Evaluate development predictions and publish all artifacts atomically."""
    if not prediction_paths:
        raise RuntimeError("At least one prediction path is required")
    paths = [path.expanduser().resolve() for path in prediction_paths]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"Prediction input is not a file: {missing[0]}")
    output = output.expanduser().resolve()
    if output.exists():
        raise RuntimeError(f"Refusing to replace portfolio evaluation: {output}")

    config = load_portfolio_config(portfolio_config_path)
    plan = load_validation_plan(validation_config_path)
    if plan.id != config.validation_set:
        raise ValueError("portfolio and temporal validation sets differ")
    tables = (pq.read_table(path) for path in paths)
    simulation = simulate_portfolio(tables, config, plan.locked_test_start)
    report = portfolio_report(simulation, config)

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.{uuid.uuid4().hex}.partial")
    partial.mkdir()
    try:
        pq.write_table(
            simulation.ledger,
            partial / "ledger.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        pq.write_table(
            simulation.equity,
            partial / "equity.parquet",
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )
        (partial / "metrics.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if output.exists():
            raise RuntimeError(f"Portfolio evaluation appeared during build: {output}")
        os.rename(partial, output)
    except FileExistsError as error:
        raise RuntimeError(
            f"Portfolio evaluation appeared during build: {output}"
        ) from error
    finally:
        if partial.exists():
            shutil.rmtree(partial)
    return PortfolioBuildResult(
        simulation.ledger.num_rows,
        simulation.equity.num_rows,
        float(simulation.equity.column("equity_usd")[-1].as_py()),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the development multi-symbol portfolio."
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--portfolio-config", type=Path, required=True)
    parser.add_argument("--validation-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the portfolio evaluation command line interface."""
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        result = run_portfolio_evaluation(
            arguments.predictions,
            arguments.portfolio_config,
            arguments.validation_config,
            arguments.output,
        )
        print(
            f"settled {result.trades} trades across {result.equity_events} "
            f"equity events; final equity USD {result.final_equity_usd:.2f}: "
            f"{arguments.output}"
        )
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"error: {error}\n")

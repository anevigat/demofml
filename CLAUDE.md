# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`demofml` is a research engine that determines whether historical FX bid/ask
tick data contains a stable, cost-aware executable trading signal at 15/30/60
minute horizons. It is **research-only**: live/paper trading is out of scope,
and a hard-locked out-of-sample test period (`2025-01-01` to `2026-03-10`) must
never be touched until a candidate passes the full development gate. Read
`README.md` before making contract or pipeline changes — it is the canonical,
extremely detailed spec of every phase, contract ID, and numeric result to
date, and much of it (research contract, acceptance thresholds, past
experiment outcomes) is not repeated here.

## Commands

```bash
# setup
python3.12 -m venv .venv && source .venv/bin/activate
python -m pip install -e ".[dev]"

# lint / type check / test (same gates as CI)
ruff check .
mypy src tests
coverage run -m pytest
coverage report

# run a single test
pytest tests/unit/test_quote_bars.py::test_name -v

# per-module 90% coverage gate (required for contract-critical modules; see below)
coverage report --include="src/demofml/data/ticks.py" --fail-under=90
```

The 90%-coverage module list (enforced in CI and `CONTRIBUTING.md`) is:
`data/ticks.py`, `bars/quotes.py`, `features/causal.py`, `labels/executable.py`,
`validation/splits.py`, `models/baseline.py`, `models/frozen.py`,
`models/locked.py`, `evaluation/signals.py`, `evaluation/portfolio.py`.
Global branch coverage must stay ≥80%. Run the full module loop from
`CONTRIBUTING.md`/CI, not just `pytest`, before considering a change done.

Container smoke test (also run in CI):
```bash
docker build --target runtime --tag demofml:ci .
docker run --rm demofml:ci   # expect: "demofml <version>"
```

## Architecture

### Contract-versioned pipeline stages

Every research stage (bars, features, labels, validation splits, models,
portfolio, screens) is an **immutable, versioned contract** identified by a
string ID (e.g. `causal-v1`, `executable-v1`, `baseline-ridge-v3`,
`purged-walk-forward-v1`). Contract definitions live in `configs/*.toml`
(dataset allowlists, feature/bar/label definitions, experiment folds and
thresholds) and are paired with a schema/ID constant in the corresponding
`src/demofml/<area>/*.py` module. Once a contract ID has been used by a run,
its definition must not change — a changed idea gets a new ID (`-v2`, `-v3`,
...), never a mutated existing one. When touching any contract module, check
whether the change requires a new version ID rather than an in-place edit.

### Module layout (`src/demofml/`)

- `data/` — tick contracts, quality audits (`audit.py`), S3 publishing
  (`publisher.py`, `remote.py`), Parquet splitting (`splitter.py`).
- `bars/` — causal quote-to-bar aggregation; half-open `[bar_start, bar_end)`
  intervals labelled by `bar_end`. `quotes.py` is v1, `quotes_v2.py` adds
  intrabar bid/ask transition imbalance for the microstructure line,
  `prospective.py` is the Campaign 2 variant.
- `features/` — causal feature sets (`causal.py` = v1, `causal_v2.py`,
  `cross_pair.py`); features may only use a completed bar and bounded
  trailing state, and a missing 5-minute bucket resets trailing windows.
- `labels/` — `executable.py` defines cost-aware long/short returns priced at
  actual bid/ask on entry/exit ticks; kept structurally separate from
  features so leakage checks stay explicit.
- `validation/` — purged walk-forward temporal splitting (`splits.py`); folds
  reject randomness, overlap, and non-UTC timestamps, and enforce the
  65-minute train/validation purge (max horizon + quote latency).
- `models/` — `baseline.py` (per-symbol/fold/horizon ridge), `frozen.py`
  (candidate freeze), `locked.py` (one-shot locked-test evaluation).
- `evaluation/` — `signals.py` (per-fold/horizon metrics), `portfolio.py`
  (cost-aware, vol-targeted, drawdown-halting portfolio accounting).
- `orchestration/` — `development.py` is the resumable multi-stage DAG runner
  (fingerprint + output hash + checkpoint per stage, safe to re-invoke after a
  crash); `locked.py` handles candidate freeze and the one-shot locked test.
- `reporting/` — acceptance-gate recomputation/replay (`acceptance.py`),
  screen evaluation (`screen.py`, `cross_pair_historical.py`); these
  **recompute metrics from raw predictions/ledgers and replay portfolio
  accounting**, they never trust a stored summary JSON as sufficient evidence.
- `prospective/` — Campaign 2: outcome-free schemas, cross-pair factors,
  custody boundary, prequalification/qualification gates for a prospective
  (not yet collected) holdout period.
- `calendars/` — trading-calendar / session logic used by prospective factors.
- `infrastructure.py`, `cli.py` — Kubernetes/MLflow/S3 smoke checks and the
  `demofml` CLI dispatcher (see `cli.py` for the subcommand list: each
  subcommand lazily imports its implementation module).

### Key invariants to preserve when editing this code

- **Causality everywhere**: bars, features, and labels must only ever look
  backward from a decision point in UTC half-open intervals. Anything that
  could leak a future tick/bar into a feature or label is a correctness bug.
- **Immutable contracts, atomic outputs**: pipeline stages write outputs
  atomically (temp file + replace, or no-replace where re-runs must not
  clobber a completed stage — see `_write_json_no_replace` /
  `_write_json_replace` in `orchestration/development.py`) and are keyed by a
  run fingerprint so a repeated invocation verifies-and-skips finished stages.
- **Locked test is forbidden by default**: `locked_test_policy = "forbidden"`
  contracts, the `freeze-candidate`/`evaluate-locked-test` CLI commands, and
  the one-shot grant/marker mechanism in `orchestration/locked.py` exist to
  make it structurally hard to touch the 2025-01-01+ locked period before a
  full development acceptance pass. Do not weaken these checks to "make a run
  work."
- **No raw data or credentials in the repo or in logs**: raw ticks, generated
  datasets, model artifacts, and credentials are `.gitignore`d and must never
  be logged to MLflow or written outside `artifacts/`/environment variables.
- **Config loading is strict**: TOML experiment/feature/dataset configs are
  parsed with `tomllib` and validated against expected contract IDs/schemas;
  prefer extending existing strict-loader patterns (reject unknown fields,
  mismatched versions) over adding permissive fallbacks.

### Infrastructure

Execution infrastructure (cluster, namespace, deployment manifests, operator
commands) is intentionally **not part of this repository** and must never be
committed here — see `infra/README.md`. S3 (MinIO) and MLflow connection
details come only from environment variables, never hardcoded — see the
"Dataset Publication" section of `README.md` for the env vars involved.

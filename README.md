# demofml

Machine-learning research engine for cost-aware Forex trading signals.

The first objective is to determine whether the available bid/ask tick history
contains a stable, executable signal at 15, 30, or 60 minute horizons. Live and
paper-trading integrations are intentionally out of scope until a model passes
walk-forward validation and a locked out-of-sample test.

## Research Contract

- Portfolio: AUDUSD, EURCHF, EURJPY, EURUSD, GBPJPY, GBPUSD, USDCAD, USDJPY.
- Decision interval: 5 minutes.
- Prediction horizons: 15, 30, and 60 minutes.
- Actions: long, short, or no trade.
- Execution: next quote tick using historical bid and ask.
- Initial capital: USD 100,000.
- Target annual volatility: 10%.
- Maximum drawdown: 10%.
- Validation: purged walk-forward; no random time-series splits.
- Locked test: 2025-01-01 through 2026-03-10.

## Project Layout

```text
src/demofml/
  data/         Data contracts, manifests, and quality checks
  bars/         Causal quote-to-bar aggregation
  features/     Feature definitions and transformations
  labels/       Executable long/short targets
  validation/   Temporal splits and leakage controls
  models/       Training and inference interfaces
  evaluation/   Cost-aware portfolio metrics
  reporting/    Reproducible experiment reports
configs/        Versioned dataset, feature, and experiment specifications
infra/          Infrastructure definitions added in later phases
tests/          Unit, integration, and synthetic fixtures
```

Raw data, generated datasets, model artifacts, and credentials must never be
committed. They are excluded in `.gitignore` and will be stored privately.

## Local Setup

Python 3.12 is the reference development version.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
ruff check .
mypy src
python -m demofml
```

## Container Image

Images are published for Linux AMD64 and ARM64 at
[`anevigat/demofml`](https://hub.docker.com/r/anevigat/demofml).

```bash
docker pull anevigat/demofml:main
docker run --rm anevigat/demofml:main
```

The `main` tag is convenient for local inspection. Reproducible jobs must pin
the immutable image digest printed by the image publishing workflow:

```text
anevigat/demofml@sha256:<digest>
```

The same workflow publishes an `mlflow-main` variant containing the tracking
server and PostgreSQL driver used by the Kubernetes infrastructure.

## Dataset Publication

The publisher builds a deterministic manifest from Parquet footers and SHA-256
checksums, then uploads the dataset under a content-addressed S3 prefix. Local
manifests are written below the ignored `artifacts/` directory. No endpoint or
credential is stored in the repository.

Install the project and configure the private connection from Kubernetes:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"

mkdir -p "$HOME/.config/demofml"
kubectl get secret demofml-minio-tls -n demofml \
  -o go-template='{{index .data "tls.crt" | base64decode}}' \
  > "$HOME/.config/demofml/minio-ca.crt"

export AWS_ACCESS_KEY_ID="$(kubectl get secret demofml-services -n demofml -o go-template='{{index .data "AWS_ACCESS_KEY_ID" | base64decode}}')"
export AWS_SECRET_ACCESS_KEY="$(kubectl get secret demofml-services -n demofml -o go-template='{{index .data "AWS_SECRET_ACCESS_KEY" | base64decode}}')"
export AWS_CA_BUNDLE="$HOME/.config/demofml/minio-ca.crt"
export S3_ENDPOINT_URL="https://$(kubectl get ingress minio -n demofml -o jsonpath='{.spec.rules[0].host}')"
export DEMOFML_DATA_BUCKET="demofml-data"
```

Inspect the manifest without connecting to S3, then publish:

```bash
python scripts/publish_dataset.py \
  --source /path/to/cleaned_ticks \
  --dry-run

python scripts/publish_dataset.py \
  --source /path/to/cleaned_ticks
```

Hashing and uploading display a percentage progress bar. Multipart uploads use
16 MiB parts by default so unstable connections have less work to retry. If
execution is interrupted, run the same command again: verified objects are
skipped and uploaded parts are reused.

### Splitting Large Parquet Files

Stop any active publisher before changing its source dataset. The streaming
converter groups existing row groups into files of approximately 128 MiB and
validates row counts and schemas before replacing an original. It uses bounded
memory and processes one source file at a time:

```bash
python scripts/split_parquet_dataset.py \
  --source /path/to/cleaned_ticks \
  --target-size-mib 128 \
  --replace-source
```

`--replace-source` is explicit because originals are deleted after successful
validation. Keep an independent backup when possible. An interrupted conversion
can be resumed with the same command; completed temporary parts are reused.

To retain the originals instead, provide an output directory on a filesystem
with enough free space:

```bash
python scripts/split_parquet_dataset.py \
  --source /path/to/cleaned_ticks \
  --output /path/to/cleaned_ticks_split \
  --target-size-mib 128
```

Publish the converted source with the normal publisher command. It creates a
new content-addressed dataset version. After publication, remove credentials
from the current shell:

```bash
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_CA_BUNDLE
unset S3_ENDPOINT_URL DEMOFML_DATA_BUCKET
```

## Tick Quality And Quote Bars

The canonical tick contract requires ordered UTC timestamps with microsecond or
nanosecond precision and float64 `bid`, `ask`, `mid`, and `spread` columns. The
audit checks null and non-finite values, positive executable prices, crossed
quotes, derived mid/spread consistency, ordering, and exact duplicates.

Run a lightweight audit while data publication is still active. This reads only
one row group per file:

```bash
python scripts/audit_ticks.py \
  --source /path/to/cleaned_ticks \
  --max-row-groups-per-file 1 \
  --output artifacts/quality/tick-audit-sample.json
```

After publication completes, pass `0` to perform the locked full audit:

```bash
python scripts/audit_ticks.py \
  --source /path/to/cleaned_ticks \
  --max-row-groups-per-file 0 \
  --output artifacts/quality/tick-audit-full.json
```

Quote bars use half-open intervals `[bar_start, bar_end)` and are labelled by
`bar_end`. A tick exactly on a five-minute boundary belongs only to the next
bar. Build one symbol at a time with bounded memory:

```bash
python scripts/build_quote_bars.py \
  --source /path/to/cleaned_ticks/EURUSD \
  --output artifacts/bars/EURUSD/quotes-5m.parquet \
  --symbol EURUSD
```

The output includes separate bid, ask, and mid OHLC values, spread statistics,
quote count, first/last tick times, and close-time staleness. Generated quality
reports and bars remain below the ignored `artifacts/` directory.

## Causal Features And Executable Labels

Feature set `causal-v1` uses only a completed bar and bounded trailing state.
It includes mid-price returns, realized volatility, spread level/z-score,
intrabar range, quote activity, staleness, elapsed time, and UTC calendar
cycles. Missing five-minute buckets reset all trailing windows so weekend or
outage gaps cannot silently enter fixed-bar lookbacks. Build it independently
for each symbol:

```bash
python scripts/build_features.py \
  --source artifacts/bars/EURUSD/quotes-5m.parquet \
  --output artifacts/features/EURUSD/causal-v1.parquet \
  --symbol EURUSD
```

Label set `executable-v1` enters on the first quote at or after each decision.
Long returns pay the entry ask and receive the horizon exit bid; short returns
receive the entry bid and pay the horizon exit ask. Entry and exit quotes must
arrive within five minutes of their scheduled time; otherwise the affected
label is null. Horizons must align to the five-minute bar grid. Labels are kept
separate from features to make leakage checks explicit:

```bash
python scripts/build_labels.py \
  --source artifacts/bars/EURUSD/quotes-5m.parquet \
  --output artifacts/labels/EURUSD/executable-v1.parquet \
  --horizons-minutes 15,30,60
```

The immutable definitions are recorded in `configs/features/causal-v1.toml`
and `configs/experiments/executable-labels-v1.toml`.

## Purged Walk-Forward Validation

Validation set `purged-walk-forward-v1` defines 36 monthly folds from January
2022 through the end of the development period. Training starts in 2018 and
expands for each fold. A 65-minute interval between training and validation
covers the 60-minute maximum horizon plus the five-minute quote latency; rows
inside that interval belong to neither side of the fold.

All ranges are half-open UTC intervals. The locked test starts on 2025-01-01
and its data interval ends on 2026-03-11, making 2026-03-10 the final covered
UTC date. Development and locked-test decision cutoffs are shortened by 65
minutes so no label reads quotes from outside its permitted interval.

Build the deterministic split manifest without accessing market data:

```bash
python scripts/build_validation_splits.py \
  --config configs/experiments/purged-walk-forward-v1.toml \
  --output artifacts/validation/purged-walk-forward-v1.json
```

The implementation rejects random or overlapping folds, insufficient purges,
non-UTC timestamps, and feature/label schemas whose version or information
window differs from the validation plan. The locked test must not be inspected
for model or feature selection.

## Development Ridge Baseline

Model set `baseline-ridge-v1` trains one deterministic ridge model per symbol,
fold, and horizon. Each model predicts long and short executable returns from
`causal-v1`; it selects the larger positive prediction or abstains to `flat`.
Missing features use medians fitted only on the fold's training rows, followed
by training-only standardization. Rows with unresolved executable targets are
excluded for that horizon.

The runner rejects key misalignment, contract-version differences, insufficient
training rows, and every timestamp in the locked period. It writes predictions
and cost-aware development metrics atomically below one ignored directory:

```bash
python scripts/run_baseline_experiment.py \
  --features artifacts/features/EURUSD/causal-v1.parquet \
  --labels artifacts/labels/EURUSD/executable-v1.parquet \
  --validation-config configs/experiments/purged-walk-forward-v1.toml \
  --model-config configs/experiments/baseline-ridge-v1.toml \
  --output artifacts/experiments/EURUSD/baseline-ridge-v1
```

Metrics include trade rate, mean executable return, dispersion, and hit rate by
fold and horizon, plus aggregate results and an always-flat comparator. They do
not claim portfolio performance because overlapping-position accounting,
position sizing, volatility targeting, and drawdown controls remain separate.
The locked test remains forbidden until one development configuration is frozen.

## Causal Portfolio Evaluation

Portfolio set `normalized-sleeve-portfolio-v1` combines the canonical eight
symbols and all three horizons as independent lots. Capital is split equally by
symbol and horizon, then divided by `horizon / 5 minutes` to account for the
scheduled overlap. This makes the fully invested steady-state gross allocation
one before risk scaling. Missing symbol decisions are handled event-by-event;
portfolio state continues across monthly fold boundaries.

Sizing uses only returns recognized at actual executable `exit_time`. A trailing
five-minute return window targets 10% annual volatility, uses 1x leverage during
warm-up, and caps leverage at 2x. If settled equity reaches 10% drawdown, the
engine permanently blocks new positions while allowing every open lot to settle.
The trigger cannot guarantee drawdown remains exactly below 10% because exits
can jump through the threshold and no intratrade mark-to-market data is available.

Phase 9 prediction set `walk-forward-predictions-v2` includes actual entry and
exit times for this accounting. Run the development portfolio after producing
predictions for all eight symbols:

```bash
python scripts/run_portfolio_evaluation.py \
  --predictions artifacts/experiments/*/baseline-ridge-v1/predictions.parquet \
  --portfolio-config configs/experiments/portfolio-v1.toml \
  --validation-config configs/experiments/purged-walk-forward-v1.toml \
  --output artifacts/portfolio/normalized-sleeve-portfolio-v1
```

The atomic output contains `ledger.parquet`, `equity.parquet`, and
`metrics.json`, including attribution by symbol, horizon, and fold. P&L applies
dimensionless executable returns to USD-normalized sleeve notional; it is not a
broker-unit FX conversion ledger. Any locked-test prediction is rejected.

## Resumable Development Pipeline

Pipeline set `development-pipeline-v1` executes the complete development DAG:
validation manifest, then bars, features, executable labels, an aligned temporal
slice and ridge baseline for each symbol, followed by the eight-symbol portfolio.
Every stage has a run fingerprint, output hashes, and a pre-build intent record.
A repeated invocation verifies and skips completed stages; if a process stopped
after atomically publishing output but before its checkpoint, the next invocation
recovers that output instead of rebuilding it.

Dataset set `cleaned-ticks-development-v1` pins 14 objects from the immutable
source publication by path, size, row count, and SHA-256. It contains only the
2018-2024 partitions. The runner waits for the source `manifest.json`, downloads
only that allowlist, and scans every actual timestamp before reading prices. Any
row outside `[2018-01-01, 2025-01-01)` is rejected. Features and labels are then
sliced at `development_decision_end`, 65 minutes before the lock, so their full
information windows remain outside the locked test.

Run locally or inside the digest-pinned Kubernetes image:

```bash
export DEMOFML_IMAGE_DIGEST="sha256:<runtime-image-digest>"
demofml run-development \
  --pipeline-config configs/experiments/development-pipeline-v1.toml \
  --workdir artifacts/runs
```

S3 and MLflow endpoints, buckets, and credentials come only from environment
variables. The run identity binds the image digest and every referenced config.
MLflow records provenance, portfolio metrics, per-symbol predictions and reports;
raw ticks, generated features, labels, and credentials are never logged. A local
file lock prevents concurrent processes from sharing one run directory.

## Status

Phase 5 publication is in progress. Phases 6-11 contracts and pipelines are
implemented; full-data execution starts only after the immutable source manifest
appears at the end of the upload.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.

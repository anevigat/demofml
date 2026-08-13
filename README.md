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

Install the project and configure the private connection to the operator's
object storage:

```bash
source .venv/bin/activate
python -m pip install -e ".[dev]"

export AWS_ACCESS_KEY_ID="<provided by the operator>"
export AWS_SECRET_ACCESS_KEY="<provided by the operator>"
export AWS_CA_BUNDLE="<path to the operator's CA bundle, if required>"
export S3_ENDPOINT_URL="<operator's private S3 endpoint>"
export DEMOFML_DATA_BUCKET="demofml-data"
```

How these values are retrieved (cluster, namespace, secret names, ingress
hosts) is operator-specific and intentionally not documented here — it is
never part of this repository.

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

The atomic output contains `ledger.parquet`, `equity.parquet`,
`period-returns.parquet`, and `metrics.json`, including attribution by symbol,
horizon, and fold. P&L applies
dimensionless executable returns to USD-normalized sleeve notional; it is not a
broker-unit FX conversion ledger. Any locked-test prediction is rejected.

## Resumable Development Pipeline

Phase 11 pipeline set `development-pipeline-v1` executes the complete development DAG:
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
  --pipeline-config configs/experiments/development-pipeline-v2.toml \
  --workdir artifacts/runs
```

S3 and MLflow endpoints, buckets, and credentials come only from environment
variables. The run identity binds the image digest and every referenced config.
MLflow records provenance, portfolio metrics, per-symbol predictions and reports;
raw ticks, generated features, labels, and credentials are never logged. A local
file lock prevents concurrent processes from sharing one run directory.

## Development Acceptance And Profiling

Phase 12 pipeline set `development-pipeline-v2` extends, rather than mutates, the
published Phase 11 contract. Acceptance set `development-acceptance-v1` freezes
its criteria before the full run is visible. It requires all eight symbols, 36
monthly folds and three horizons; positive pooled executable returns at every
horizon; at least 24 positive folds and six positive symbols per horizon; and at
least 100 trades per symbol/horizon. Portfolio acceptance requires positive
return, drawdown below 10%, realized volatility between 7.5% and 12.5%, leverage
no greater than 2x, no drawdown halt, and attribution reconciliation within USD
0.01.

The gate recomputes signal metrics from predictions and replays portfolio
accounting from those predictions. Stored ledger, equity, observed five-minute
returns, risk metrics and attribution must match the replay; summary JSON alone
is never sufficient evidence.

These criteria apply only to development. A rejection is a valid research result:
the pipeline and MLflow run still finish, but `development_accepted` is logged as
zero and the locked test remains forbidden. Every run also writes an
`execution-report.json` with stage action, elapsed nanoseconds, output bytes and
rows, and process peak RSS. The report contains no raw object paths.

Follow-up model set `baseline-ridge-v2` preserves the ridge, features, labels,
walk-forward folds, portfolio rules and acceptance thresholds. It fits a
non-negative affine calibrator on one purged month before each validation fold
and trades only when the calibrated executable return is positive. Its
development replay was also rejected: all three horizon means remained
negative, only eight folds per horizon were positive, and the portfolio returned
-7.19%. This result must not be promoted to the locked test.

An additive spline-ridge hypothesis was screened using only pre-2022 data before
any further walk-forward run. It made the already negative 30- and 60-minute
means materially worse and was not promoted to a versioned candidate. The
current feature set therefore has no demonstrated stable executable signal under
either the linear ridge or the screened nonlinear transformation.

Re-evaluate an existing run without rebuilding it:

```bash
demofml evaluate-development \
  --run-root artifacts/runs/development-pipeline-v2/sha256-<run-id> \
  --acceptance-config configs/experiments/development-acceptance-v1.toml
```

## Campaign 3 Protocol: Sealed Envelopes

`docs/research/campaign-3-protocol-v1.md` binds every Campaign 3 variant. Before
a variant's first fold is run, four documents are committed together — the
hypothesis, the validation contract, the model contract including its complete
hyperparameter search space, and the acceptance contract including every
pre-registered threshold — and their SHA-256 digests are recorded in a
`sealed-envelope-v1` TOML committed in the same change.

The seal is what makes "this was decided in advance" checkable rather than
merely asserted: it proves the four documents the acceptance gate evaluated are
byte-identical to the four that were committed before any result was observed.
Verify one on demand:

```bash
demofml verify-sealed-envelope \
  --envelope configs/experiments/campaign-3-<variant>-envelope-v1.toml
```

An acceptance contract that declares `sealed_envelope` is verified
automatically by the development gate, which refuses to evaluate a run whose
seal is broken or whose acceptance contract is not the sealed one. Contracts
that declare no envelope — every Campaign 1 and Campaign 2 contract — behave
exactly as before.

## Next Research Phase: Tick Microstructure Features

The next hypothesis adds information discarded by `quote-bars-v1`, rather than
changing the model or tuning an action threshold. A new `quote-bars-v2` contract
will aggregate causal intrabar bid/ask update imbalance, mid-price upticks versus
downticks, spread widening versus narrowing, and quote inter-arrival dispersion.
`causal-v2` will expose fixed 15- and 60-minute trailing summaries of those
quantities alongside the existing features.

The first screen is restricted to development history before 2022. It trains on
`[2018-01-01, 2021-01-01)` and evaluates `[2021-01-01, 2022-01-01)`, preserving
the 65-minute information purge. The ridge specification, executable labels,
symbols, horizons, portfolio accounting and zero-bps action threshold remain
unchanged so the feature contribution is isolated. No feature, window or model
selection may use 2022-2024 or locked-test outcomes.

Promotion to a new immutable full walk-forward contract requires, on the 2021
screen, a positive pooled executable mean at all three horizons, at least six of
eight positive symbols at every horizon, and at least 100 trades in every
symbol/horizon cell. If any screen condition fails, this research line stops. If
all pass, the feature contract is frozen before one 2022-2024 walk-forward run,
which remains subject to the unchanged Phase 12 acceptance gates. The locked
test remains forbidden throughout.

The implementation is frozen before execution. The versioned contracts are
`quote-bars-v2`, `causal-v2`, `executable-v2`,
`causal-v2-screen-2021-v1`, `baseline-ridge-v3`,
`normalized-sleeve-portfolio-v3`, and
`microstructure-screen-acceptance-v1`. Label formulas, ridge parameters,
symbols, horizons, action threshold and portfolio policies are identical to the
original static baseline; new IDs record only the changed data provenance.

`quote-bars-v2` compares consecutive quotes only inside each half-open bar. It
uses the canonical physical order for equal timestamps, computes rolling
imbalances from summed transition counts, and defines inter-arrival dispersion
as population standard deviation in seconds. `causal-v2` uses fixed 15- and
60-minute windows and resets every trailing window after a missing five-minute
bar. These choices must not be changed after observing the 2021 screen.

The official execution path is the resumable 43-stage
`microstructure-screen-pipeline-v1`. It binds every contract and the runtime
digest into one run identity, applies `2022-01-01T00:00:00Z` as the bar-data
cutoff, and requires all eight symbol outputs before evaluating the gates. The
standalone `evaluate-microstructure-screen` command emits a scientific result
only and cannot authorize promotion; the final acceptance envelope additionally
verifies the pipeline run, stage inventory, screen checkpoint and config hashes.
This pipeline does not authorize a 2022-2024 run or any locked-test access.

The single authorized 2021 screen completed all 43 stages on runtime digest
`sha256:a24cd0b03331eb743c00c077a292d8cc40553f9b0732949224eb5876c3201f9d`
with pipeline run
`sha256-76ca5d4051004414428fe4aff5a2a614a37cdfaca1106a971aa736118702d325`.
It was rejected. Pooled executable means for 15/30/60 minutes were
`-0.004315/-0.013522/-0.028682` bps, with only `1/3/3` positive symbols.
The minimum-trade gate also failed in five 15-minute cells: AUDUSD 95, EURCHF
78, GBPJPY 22, GBPUSD 46 and USDCAD 49. The diagnostic portfolio returned
`-2.96%` with 3.25% maximum drawdown and 127,884 trades. The acceptance envelope
sets `promotion_authorized=false` and `next_action=stop_microstructure_research_line`.
Per the frozen protocol, this line is closed without a 2022-2024 follow-up.

## Frozen Candidate And One-Shot Locked Test

Phase 13 protocol `locked-test-evaluation-v1` is fixed before development
results are available. It does not relax any Phase 11 or Phase 12
`locked_test_policy = "forbidden"` contract. After, and only after, one Phase 12
run is accepted, freeze its final candidate without S3 credentials:

```bash
demofml freeze-candidate \
  --run-root artifacts/runs/development-pipeline-v2/sha256-<run-id> \
  --protocol-config configs/experiments/locked-test-evaluation-v1.toml \
  --output artifacts/locked-test/candidate-v1 \
  --code-reference sha256:<phase-13-image-digest>
```

The command recomputes development acceptance, requires the recorded report to
match, fits one final ridge model per symbol/horizon using only fully resolved
development rows, and stores portable numeric JSON rather than Python pickles.
The package contains 24 models, 73 pre-lock context bars per symbol, contract
copies, accepted stage markers and hashes for every artifact. Training reads
only temporary source snapshots whose hashes match those captured markers. The
package refuses replacement and records explicitly that no locked data was
accessed.

The locked evaluator scores every eligible decision before consulting outcome
availability. Hidden executable labels are joined only afterward; unresolved
executions remain in the evaluation as explicit flat, zero-return rows and fail
the frozen completeness gate above 5%, so they cannot selectively remove a
model score. A custodian-supplied grant
binds the candidate manifest, its complete contract bundle, the locked dataset
allowlist and runtime image by SHA-256. A local durable
`_LOCKED_TEST_STARTED.json` marker is created before the S3 client is
constructed, and the same claim root cannot retry after a failure:

```bash
demofml evaluate-locked-test \
  --candidate-root artifacts/locked-test/candidate-v1 \
  --protocol-config configs/experiments/locked-test-evaluation-v1.toml \
  --dataset-config /protected/cleaned-ticks-locked-test-v1.toml \
  --grant /protected/locked-test-grant-v1.json \
  --workdir /protected/locked-test-work \
  --code-reference sha256:<same-phase-13-image-digest>
```

Do not run either command merely because the code exists. Candidate freeze
requires a completed and accepted full development run. Locked evaluation also
requires a separately protected, read-only data principal, an externally issued
grant and storage where researchers cannot delete the consumed marker. A PVC
file alone cannot enforce one-shot behavior against a namespace administrator,
nor prevent the same grant being used with another path, so the external claim
authority must atomically consume the grant ID first. No deployable locked-test
Job is included yet.

## Status

Phase 5 publication and the full tick audit are complete. Phases 6-13 contracts
and pipelines are implemented. The Phase 12 full-development run and the
development-only calibrated follow-up both failed the unchanged acceptance
criteria; the pre-2022 nonlinear screen also failed. Phase 13 remains inactive,
and the locked test remains forbidden. The pre-2022 tick-microstructure screen
also failed all three promotion gates, so that research line is closed.

Campaign 1 is formally closed in
[`docs/research/campaign-1-closeout.md`](docs/research/campaign-1-closeout.md).
The Campaign 2 engineering protocol is
[`docs/research/campaign-2-prospective-factor-plan.md`](docs/research/campaign-2-prospective-factor-plan.md).
Its outcome-free schemas, synchronous cross-pair factors, calendar, coverage
ledger, strict config loader, append-only manifest contracts, engineering-only
bundle, and prequalification claim envelope are implemented. It reserves
`[2026-09-01, 2027-09-01)` as a prospective holdout. No real collector run or
external bundle freeze has occurred. Campaign 2 requires a separately
administered on-prem custody boundary; the shared development MinIO is not
eligible, and no real custody tenant, attestation, or preflight exists. The
existing locked test remains excluded; fitting, collection, scoring, evaluation,
and every Campaign 2 performance run remain unauthorized.

The data-free Campaign 2 engineering verification passed in the operator's
on-prem environment on 2026-08-04. Its immutable runtime/contract identities and
non-authorization result are recorded in
[`docs/research/campaign-2-engineering-verification-2026-08-04.md`](docs/research/campaign-2-engineering-verification-2026-08-04.md).
Campaign 2 v1 qualification is now fail-closed because its frozen prospective
receipt interval cannot reach the required coverage without pre-existing
custodial evidence. The calculation and prohibited recovery paths are recorded
in
[`docs/research/campaign-2-v1-qualification-blocker-2026-08-05.md`](docs/research/campaign-2-v1-qualification-blocker-2026-08-05.md).

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.

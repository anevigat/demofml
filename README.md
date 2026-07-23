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

## Status

Phase 5 data publication and Phase 6 tick quality/bar construction in progress.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.

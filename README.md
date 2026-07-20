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

## Status

Phase 4, Kubernetes research infrastructure.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.

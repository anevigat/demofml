# Configuration

All research runs will be defined by versioned configuration. Dataset,
feature-set, and experiment identifiers must be immutable once used by a run.

- `datasets/`: content-addressed source allowlists, hashes, and permitted ranges.
- `features/`: feature names, lookbacks, and transformations.
- `experiments/`: folds, models, horizons, costs, and risk constraints.

`experiments/locked-test-evaluation-v1.toml` freezes the Phase 13 protocol but
does not authorize test access. The locked object allowlist and one-shot grant
are custodial inputs created only after publication and candidate acceptance;
they must not be inferred from test results or replaced between attempts.

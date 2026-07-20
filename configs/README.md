# Configuration

All research runs will be defined by versioned configuration. Dataset,
feature-set, and experiment identifiers must be immutable once used by a run.

- `datasets/`: source manifests, date ranges, and partitions.
- `features/`: feature names, lookbacks, and transformations.
- `experiments/`: folds, models, horizons, costs, and risk constraints.

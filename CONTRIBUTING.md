# Contributing

## Delivery Discipline

Complete each project phase or other significant change with a commit and push
after its tests and static checks pass. Before staging, verify that the changes
contain no private hostnames, credentials, private keys, local absolute paths,
market data, generated datasets, or private artifacts.

Generated research outputs belong below ignored directories such as
`artifacts/`; secrets remain in Kubernetes or local environment variables.

## Test Coverage

Every significant change must preserve at least 80% branch coverage globally
and 90% coverage individually for tick contracts, quote bars, causal features,
and executable labels. Run the same gates enforced by CI before committing:

```bash
coverage run -m pytest
coverage report
for module in \
  src/demofml/data/ticks.py \
  src/demofml/bars/quotes.py \
  src/demofml/features/causal.py \
  src/demofml/labels/executable.py; do
  coverage report --include="$module" --fail-under=90
done
```

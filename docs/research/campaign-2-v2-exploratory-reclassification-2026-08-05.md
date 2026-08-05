# Campaign 2 v2 Exploratory Reclassification

Status: frozen before historical outcome access

Declared at: `2026-08-05T11:24:26Z`

## Decision

The user selected a historical profitability screen instead of preserving
Campaign 2 v2 as an outcome-blind prospective attempt. Campaign 2 v2 is
therefore reclassified as exploratory and cannot produce confirmatory evidence,
even if the historical result is favorable. Its previous engineering
verification remains valid only as technical evidence.

No locked-test or post-2024 object is authorized. Prospective collection,
prospective scoring, raw prospective access, and confirmatory evaluation remain
false. A future confirmation requires a new campaign version and a new interval
declared before collection.

## Frozen Screen

The exact contract is
`configs/experiments/campaign2-cross-pair-historical-screen-2020-v1.toml`.
The screen is fixed as follows:

- Train on `[2018-01-01T00:00:00Z, 2019-12-31T22:55:00Z)`.
- Preserve a 65-minute purge before the screen.
- Evaluate decisions on
  `[2020-01-01T00:00:00Z, 2020-12-31T22:55:00Z)`.
- Read no source row at or after `2021-01-01T00:00:00Z`.
- Use the canonical eight pairs and horizons 15, 30, and 60 minutes.
- Fit per-symbol ridge models with alpha `1.0`, `lsqr`, training medians,
  training-only standardization, and zero-bps action threshold.
- Compare a `causal-v1` control with a candidate that appends the twelve frozen
  cross-pair columns.
- Fit and score both arms on identical synchronized factor-ready keys.
- Count absent, unresolved, or unscored expected opportunities as flat zero in
  the expected-calendar denominator.
- Use executable bid/ask returns already present in `executable-v1`; historical
  event time is not represented as trusted receipt time.
- Report executable return per expected opportunity. Overlapping trade returns
  are not compounded into portfolio return, leverage, or deployable capital.

The sixteen input Parquet hashes and expected full-file row counts from
development run
`sha256-c8903538bf0c2a81cd866e318e4eb6ac0a8abf21fedd96eee8b10322ebf41758`
were frozen before computing any return metric. The source derivatives cover
2018-2024, but the runner must use predicate-filtered rows ending before 2021
and fail if any selected timestamp reaches the locked boundary.

## Interpretation

The result may answer whether the cross-pair information set was historically
promising enough to motivate another prospectively sealed attempt. It cannot
demonstrate future profitability because all pre-2025 history was previously
available to Campaign 1, and the 2020 regime is not an untouched holdout.

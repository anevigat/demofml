# Campaign 3 Stage B Hypothesis — gated `lightgbm` / `causal-v2`

Status: sealed, not yet executed

Written: 2026-08-20

This document is the `hypothesis` role of the Stage B sealed envelope (see
`campaign-3-protocol-v1.md`). It is written after Stage A's result was observed
and before any Stage B fold has been run.

## Hypothesis

Stage A's estimator was not wrong about direction everywhere; it was wrong
about *when to act*. The executable edge exists on a minority of decisions and
is smaller than the spread on the rest, so a rule that trades every
positive-scoring decision converts a thin edge into a systematic loss.

Restricting execution to high-conviction decisions — those in the upper tail of
the model's own predicted-return distribution, optionally further restricted to
narrow-spread regimes — isolates a subset whose mean cost-aware executable
return is positive, while the discarded majority is where the bleed lives.

The estimator is unchanged from Stage A. Only the action rule changes.

## Why this and not a bigger model

Stage A's evidence points at execution, not at capacity:

1. **The loss was not volatility.** Realized annualized volatility was 0.61%
   against a 10% target, and the portfolio still lost 10.05% over 192,499
   trades. Losing that much while barely moving is per-trade cost, not risk.
2. **The damage scaled with turnover.** Per-horizon means deteriorated
   monotonically (`-0.018` → `-0.038` → `-0.070` bps) and the worst cells were
   the highest-turnover pairs, USDJPY and EURUSD.
3. **More capacity had already been rejected from inside.** All eight symbols'
   inner cross-validation chose the most regularized of three candidates, using
   training rows only. Enlarging the model is the one direction Stage A's own
   selection already argued against.

## What would refute it

Stage B is refuted if any of the following holds:

- The pooled mean executable return fails the acceptance contract's minimum at
  all three horizons, i.e. gating does not turn the sign.
- **The gate degenerates.** If the surviving subset is too thin to matter, the
  contract's floor of 100 trades per symbol-horizon fails. This is a
  deliberately pre-registered discriminator: it separates "there is an edge
  hidden under costs" from "there is no edge, and the only way to stop losing
  is to stop trading". A configuration that wins by trading almost nothing must
  not be recorded as a success.
- The inner cross-validation selects `all-signals` — Stage A's ungated rule —
  for most cells. The gates are then not earning their place on training data,
  independently of what the validation folds say.
- The portfolio replay violates its risk envelope.

## Data already observed (Rule 3 disclosure)

This is the section that matters most, and it is uncomfortable.

**Stage B's validation period is already consumed.** Stage A observed the
2022-2024 walk-forward in full — pooled means, per-symbol means, per-fold
counts, trade counts and portfolio behaviour — and this hypothesis is a direct
response to what those numbers showed. Under Rule 3 the same folds cannot serve
as independent confirmation of the decision they motivated.

No clean alternative exists. 2018-2021 carries three prior screens, 2022-2024
is now consumed, and 2025-01-01 onward is the locked test. There is no unused
development period left to move Stage B onto.

Two specific dependencies on observed data are recorded rather than hidden:

| Design element | Informed by |
| --- | --- |
| Gating on conviction and spread regime at all | Stage A's cost-bleed pattern on 2022-2024 |
| The quantile grid `{1.0, 0.5, 0.25}` | Stage A's observed per-cell trade counts, so no cell is structurally disqualified by the 100-trade floor |

The consequence is stated in advance, not argued afterwards: **a passing Stage B
development result is candidate-generating only.** It is weaker evidence than
Stage A's would have been, because it is the second design shaped by the same
observations. Independent confirmation can come only from the locked test.

## Variant budget

Each additional variant tuned against 2022-2024 erodes the independence of the
eventual locked test, because whatever reaches it will have been chosen by
repeated looks at the same development data. To keep that erosion bounded and
visible, this is pre-registered here:

**Stage B is variant 2 of at most 3 development variants in Campaign 3.** If a
third variant also fails, the campaign closes as a null result on this data and
feature family. The locked test is not spent on a candidate that needed four
attempts to appear.

## Sealed design commitments

- **Estimator**: unchanged from Stage A — LightGBM, per symbol/fold/horizon,
  `causal-v2` features, `executable-v2` labels, the same three candidates, the
  same `campaign-3-walk-forward-v1` folds. The model contract is a new id
  (`gbm-lightgbm-v2`) because its action rule differs; `gbm-lightgbm-v1` has
  been used by a run and is never edited.
- **Action rule**: a decision trades only if its score clears a conviction cut
  and, for gates that declare one, its `spread_zscore_72` is at or below the
  declared bound. A missing spread z-score never trades.
- **Gate calibration**: the conviction cut is a quantile of the score
  distribution over that fold's *training* rows, restricted to decisions the
  base rule would take. Only the policy — which quantile, which spread bound —
  is frozen across the walk-forward; the level is recomputed per fold from
  training data alone. This mirrors the per-fold calibration pattern already
  used by `baseline-ridge-v2`.
- **Known caveat, disclosed rather than smoothed over**: the cut is computed
  from in-sample training scores, which are more dispersed than out-of-fold
  scores. The gate is defined relatively (top *q*) precisely so that this shift
  moves the cut rather than the policy, but it does mean the realized retained
  fraction on validation rows will not be exactly *q*.
- **Selection**: candidate and gate are chosen jointly by expanding inner
  cross-validation over the first fold's training window, with the same
  65-minute purge at every inner boundary. Gates are scored on predictions the
  candidates already produced, so the wider search space costs no extra fits.
- **Acceptance**: thresholds byte-identical to Stage A and to
  `development-acceptance-v2`. The bar does not move between variants.

## Non-goals

- No change to features, labels, horizons, folds, or portfolio rules.
- No locked-test access under any Stage B outcome.

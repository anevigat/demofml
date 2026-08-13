# Campaign 3 Stage A Hypothesis — `lightgbm` / `causal-v2`

Status: sealed, not yet executed

Written: 2026-08-13

This document is the `hypothesis` role of the Stage A sealed envelope (see
`campaign-3-protocol-v1.md`). It is written before any Campaign 3 fold has been
run, and it is byte-frozen from the moment the envelope is committed.

## Hypothesis

Cost-aware executable returns at 15, 30, and 60 minute horizons depend on
causal five-minute quote and microstructure summaries through **interactions
and thresholds that a linear model in those same summaries cannot represent**.
Specifically: the sign and size of the tradeable edge is conditional on the
liquidity regime — on spread level, quote intensity, and interarrival
dispersion jointly — rather than being an additive function of each.

A gradient-boosted tree ensemble over `causal-v2` features, with hyperparameters
selected inside each fold's training window, therefore produces a positive
cost-aware executable edge on the 2022-2024 development walk-forward where
Campaign 1's ridge produced none.

## Why this is not a rerun of Campaign 1

Campaign 1's null result is specific and narrow. It tested one functional form
— a linear map, at a single unsearched `alpha = 1.0`, in all three of its model
contracts — and refuted it. It did not test whether the relationship exists in
a form a linear map cannot express. Two independent facts motivate the
non-linear hypothesis rather than mere hope:

1. **The microstructure features carried the correct sign.** Campaign 2's 2020
   cross-pair historical screen found the hypothesized direction present but of
   insufficient magnitude, using the same linear model. Sign-correct but
   magnitude-insufficient under a linear fit is exactly the signature expected
   when the true response is conditional rather than additive.
2. **Liquidity regimes are structurally multiplicative.** An executable edge
   priced at real bid/ask must exceed the spread it pays. A predictor's value
   is therefore inherently interacted with the cost state at the same instant,
   which a model additive in "signal" and "spread" cannot represent.

## What would refute it

The pre-registered thresholds live in the sealed acceptance contract; this
section states what they mean, not what they are. Stage A is refuted if any of
the following holds on the development walk-forward:

- The pooled mean executable return fails the acceptance contract's minimum at
  all three horizons.
- The edge does not generalize across the universe: fewer than the required
  number of symbols are positive per horizon.
- The edge exists in-sample but not out-of-fold — i.e. training-window inner-CV
  scores are strong while outer validation folds are not, indicating the
  ensemble fit noise the ridge was too constrained to fit.
- The portfolio replay violates its risk envelope (drawdown, realized
  volatility band, gross leverage, or a drawdown halt).

A refutation here is a real result and closes Stage A. It is not grounds for
re-searching hyperparameters on the same folds; that would violate Rule 3 of
the protocol.

## Data already observed (Rule 3 disclosure)

Honesty about consumed periods is the point of this section. As of sealing:

| Period | Observed as | Consequence for Stage A |
| --- | --- | --- |
| 2018-2021 | Campaign 1 pre-2022 spline screen; Campaign 1 `causal-v2` 2021 screen; Campaign 2 2020 cross-pair screen | Training only. Never an independent validation period for Stage A. |
| 2022-2024 | Campaign 1 `baseline-ridge-v1`/`v2` development walk-forward, observed in full | See below. |
| 2025-01-01 – 2026-03-10 | Never observed | Locked test. Remains forbidden. |

The 2022-2024 disclosure matters and is stated plainly: the decision to move to
a non-linear model class was itself informed by observing that ridge failed on
those exact folds. The information that crossed from those observations into
this hypothesis is coarse — "the linear form found nothing" — and no Campaign 3
feature, hyperparameter, or threshold was chosen by inspecting fold-level
Campaign 1 results. But it is not zero.

The consequence is therefore recorded in advance, not argued afterwards: **a
passing Stage A development result is a candidate-generating result, not
independent confirmation.** Independent confirmation can only come from the
locked test, which no campaign has touched. This is the reason the locked-test
prohibition is worth its cost, and Stage A must not be allowed to erode it.

## Sealed design commitments

- **Model family**: gradient-boosted decision trees (LightGBM), per symbol, per
  fold, per horizon — the same partitioning as `baseline.py`, so the comparison
  against Campaign 1 is like-for-like.
- **Feature family**: `causal-v2` (the `causal-v1` set plus intrabar bid/ask
  transition imbalance, tick imbalance, spread-change imbalance, and
  interarrival dispersion at 15 and 60 minutes).
- **Label family**: `executable-v2` cost-aware long/short returns priced at
  actual bid/ask, unchanged from Campaign 1. `minimum_return_bps` stays at
  `0.0` for comparability; a commission-floor sensitivity run is a separate,
  separately versioned variant and is not part of this envelope.
- **Validation**: purged expanding walk-forward, monthly folds over 2022-2024,
  65-minute purge, `causal-v2` feature set. This requires a new validation
  contract id, since `purged-walk-forward-v1` is bound to `causal-v1` and is
  immutable.
- **Hyperparameter search**: a fixed, fully enumerated candidate set declared in
  the sealed model contract. Selection follows Rule 2 of the protocol — inside
  the training window only.
- **Estimation hygiene**: every fitted transformation is fit on training rows
  only, reusing the existing fit-on-train-only pattern. LightGBM's native NaN
  handling replaces median imputation, which removes one fitted statistic
  rather than adding one.

## Non-goals

- No live or prospective data collection.
- No deep learning: it requires GPU capacity that does not exist here.
- No locked-test access under any Stage A outcome.

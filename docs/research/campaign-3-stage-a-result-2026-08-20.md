# Campaign 3 Stage A Result — `lightgbm` / `causal-v2`

Status: refuted, closed

Result date: 2026-08-20

Stage A tested whether the executable edge exists in a form a linear model
cannot represent. It does not — at least not in the form this variant looked
for. The hypothesis in
`campaign-3-lightgbm-causal-v2-hypothesis-v1.md` is refuted on its own
pre-registered terms, and Stage A is closed without a promoted candidate.

## Run identity

| Field | Value |
| --- | --- |
| Pipeline | `campaign-3-lightgbm-causal-v2-pipeline-v1` |
| Pipeline run | `sha256-24706754d3685e4b4e3153645225638cf7dec65b58a18b2ba191f4c8308b5aca` |
| Runtime digest | `sha256:933966f7d4b5e403004b2520d0c4ce12d2a6d1130e0c4e35fddde8e31bc58cb3` |
| MLflow run | `2c99b467a3314f74bf936a8bf7eb73ac` |
| Sealed envelope | `campaign-3-lightgbm-causal-v2-envelope-v1`, sealed `2026-08-13T00:00:00Z` |
| Seal verification | All four documents unchanged; verified in-image before the run and again by the acceptance gate |
| Source data | 14 authorized files, 1,624,981,795 rows, ending `2025-01-01T00:00:00Z` |
| Execution | 42/42 stages, exit code 0, no restart |
| Acceptance | `accepted=false`, 13 pass / 7 fail / 0 blocked |

The seal held end to end. The contract the gate evaluated is byte-identical to
the contract committed before the first fold ran, and the acceptance report
records the digest of all four sealed documents.

## Acceptance failures

| Check | Observed | Required |
| --- | --- | --- |
| `model.positive_horizon_means` | `-0.0184 / -0.0381 / -0.0702` bps | `> 0` at each horizon |
| `model.positive_folds` | `7 / 6 / 9` of 36 | `>= 24` |
| `model.positive_symbols` | `0 / 0 / 0` of 8 | `>= 6` |
| `portfolio.total_return` | `-10.05%` | `> 0` |
| `portfolio.maximum_drawdown` | `10.06%` | `< 10%` |
| `portfolio.realized_volatility` | `0.61%` annualized | `[7.5%, 12.5%]` |
| `portfolio.drawdown_halt` | halted | no halt |

Everything structural passed: provenance, 42 verified stages, 36 folds, the
development-timestamp safety checks on both predictions and portfolio, full
portfolio recomputation, and ledger/equity reconciliation. The failure is a
research result, not an engineering fault.

## Per-symbol mean executable return (bps)

| Symbol | 15m | 30m | 60m | Trades 15/30/60 |
| --- | --- | --- | --- | --- |
| AUDUSD | -0.0146 | -0.0462 | -0.1170 | 1,441 / 5,318 / 12,998 |
| EURCHF | -0.0026 | -0.0147 | -0.0425 | 609 / 3,360 / 10,232 |
| EURJPY | -0.0186 | -0.0266 | -0.0458 | 6,871 / 18,488 / 42,553 |
| EURUSD | -0.0280 | -0.0401 | -0.0910 | 20,279 / 41,535 / 73,893 |
| GBPJPY | -0.0080 | -0.0220 | -0.0425 | 1,796 / 7,076 / 17,480 |
| GBPUSD | -0.0202 | -0.0491 | -0.0609 | 5,496 / 12,022 / 22,910 |
| USDCAD | -0.0091 | -0.0318 | -0.0778 | 1,504 / 4,708 / 12,533 |
| USDJPY | -0.0457 | -0.0744 | -0.0840 | 16,738 / 38,540 / 85,034 |

All 24 symbol-horizon cells are negative. Not one is positive, at any horizon,
for any pair. This is not a marginal miss of a threshold.

## What the evidence says

**The loss scales with exposure, not with volatility.** Realized annualized
volatility was 0.61% against a 10% target — the vol-targeting sleeve scaled
positions far down — and the portfolio still lost 10.05% and tripped the
drawdown halt after 192,499 trades. A strategy that bleeds while barely moving
is paying costs, not taking bad risk. The per-horizon means deteriorate
monotonically (`-0.018` → `-0.038` → `-0.070` bps) and the worst cells are the
highest-turnover pairs, which is the signature of spread paid per trade rather
than of a signal with the wrong sign.

**The inner cross-validation reached the same conclusion independently.** All
eight symbols selected `shallow-strong-regularization`, the most constrained of
the three sealed candidates, at every horizon. That choice was made strictly
inside the first fold's training window, with no sight of any validation fold.
Even where the model was free to fit, additional flexibility did not pay. This
matters: it is evidence against the hypothesis that does not depend on the
outer folds at all.

**Non-linearity did not rescue the signal; it cost more.** Campaign 1's
`baseline-ridge-v2` failed the same 36 folds, the same eight symbols, the same
horizons and the same portfolio rules with a `-7.19%` portfolio return and
eight positive folds per horizon. Stage A returns `-10.05%` with `7/6/9`.
The comparison is close but not controlled — ridge-v2 ran on `causal-v1` while
Stage A ran on `causal-v2` — so the honest reading is that the added model
capacity and the added microstructure features together produced no
improvement, and plausibly more turnover to pay for.

## What is and is not refuted

Refuted: that gradient-boosted trees over `causal-v2`, with the sealed
three-point search space and this action rule, produce a positive cost-aware
executable edge at 15, 30 or 60 minutes on 2022-2024.

Not refuted: that some exploitable structure exists in this data under a
different label definition, holding period, execution model, or instrument
universe. Stage A says nothing about those, and nothing observed here licenses
a claim about them.

## Consequences under the protocol

- **Rule 3 applies now.** The 2022-2024 walk-forward has been observed for the
  gradient-boosting-over-`causal-v2` question. It may still be used for
  training, but it can no longer serve as independent validation of any
  decision informed by this result.
- **No re-search on these folds.** Re-tuning hyperparameters, pruning features
  by observed importance, or adjusting the action threshold and re-scoring the
  same folds is exactly the double use the hypothesis document forbade in
  advance. Any of those requires a new variant, a new sealed envelope, and a
  ledger entry naming this result as the motivation.
- **The locked test remains forbidden and untouched.** Stage A produced no
  candidate, so nothing is eligible for it. `2025-01-01` to `2026-03-10` has
  still never been read.

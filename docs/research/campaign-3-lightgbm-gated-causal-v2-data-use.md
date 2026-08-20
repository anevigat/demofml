# Campaign 3 Stage B Data-Use Ledger — gated `lightgbm` / `causal-v2`

Append-only. Entries are never edited or removed; a correction is a new entry.
Not part of the sealed envelope — it is meant to grow, which is exactly what a
sealed document may not do.

| Date | Event | Periods observed | Motivating result | Consequence |
| --- | --- | --- | --- | --- |
| 2026-08-20 | Stage B opened. Estimator unchanged; a conviction and spread-regime gate is added to the action rule, selected inside the training window. | None new. Stage B inherits Stage A's full observation of 2022-2024. | Stage A's refutation: -10.05% portfolio at 0.61% realized volatility over 192,499 trades, means worsening monotonically with horizon and worst on the highest-turnover pairs — cost bleed, not mispriced risk. | Design shaped by observed 2022-2024 results. Under Rule 3 those folds cannot confirm this design independently, so a Stage B pass is candidate-generating only. Pre-registered variant budget: Stage B is variant 2 of at most 3 before the campaign closes or the locked test is spent. |
| 2026-08-20 | Quantile grid fixed at `{1.0, 0.5, 0.25}`. | None new. | Stage A's observed per-cell trade counts (thinnest cell: EURCHF 15m, 609 trades). | The grid is sized so no symbol-horizon cell is structurally disqualified by the contract's 100-trade floor. Recorded because it is a second, narrower dependency on observed results. |
| 2026-08-20 | Envelope `campaign-3-lightgbm-gated-causal-v2-envelope-v1` sealed over all four documents. | None. No fold has been run. | None — this is the pre-registration itself. | The design is frozen. Any later change requires a version bump, a new envelope and run id, and an entry here naming the result that motivated it. |

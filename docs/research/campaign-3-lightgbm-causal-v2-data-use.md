# Campaign 3 Stage A Data-Use Ledger — `lightgbm` / `causal-v2`

Append-only. Entries are never edited or removed; a correction is a new entry.
This file is intentionally **not** part of the sealed envelope — it is meant to
grow after sealing, which is exactly what a sealed document may not do.

Every entry records what was observed or changed, and which observed result
motivated it, so that Rule 3 of `campaign-3-protocol-v1.md` (an observed period
is consumed for the decision it informed) can be applied by a later reader
without reconstructing the history from commits.

| Date | Event | Periods observed | Motivating result | Consequence |
| --- | --- | --- | --- | --- |
| 2026-08-13 | Stage A hypothesis written; protocol `campaign-3-protocol-v1` adopted. | None in this campaign. Prior observations inherited from Campaigns 1 and 2 are disclosed in the hypothesis document. | Campaign 1's ridge null result on the 2022-2024 walk-forward. | Non-linear model class selected. 2022-2024 is candidate-generating for Stage A, not independent confirmation. |
| 2026-08-13 | Envelope `campaign-3-lightgbm-causal-v2-envelope-v1` sealed over all four documents. Search space fixed at three candidates; selection policy fixed to `inner-purged-cv-first-fold-v1`; acceptance thresholds copied unchanged from `development-acceptance-v2`. | None. No fold has been run. | None — this is the pre-registration itself. | The design is now frozen. Any later change requires a version bump, a new envelope and run id, and an entry here naming the result that motivated it. |
| 2026-08-20 | Stage A executed and evaluated. Run `sha256-24706754…8b5aca`, acceptance `accepted=false` (13 pass / 7 fail). Result recorded in `campaign-3-stage-a-result-2026-08-20.md`. | **2022-2024 monthly walk-forward, all 8 symbols, all 3 horizons — now observed in full.** | None; this is the pre-registered run itself, executed without any change to the sealed documents. | Stage A is refuted and closed. Under Rule 3, 2022-2024 is consumed for the gradient-boosting-over-`causal-v2` question: still usable for training, never again as independent validation of a decision informed by this result. Re-tuning or feature pruning on these folds requires a new variant, a new envelope, and an entry here. |

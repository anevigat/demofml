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

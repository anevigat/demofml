# Research Campaign 1 Closeout

Status: closed

Closure date: 2026-08-03

## Decision

Campaign 1 is closed without a promoted candidate. Phase 13 remains inactive,
the locked test remains forbidden, and no Campaign 1 model may be retuned or
re-screened on the periods already observed.

The campaign tested whether causal five-minute Forex quote summaries support a
stable executable directional signal across eight pairs and 15, 30, and 60
minute horizons. Every promoted or screened line failed its versioned acceptance
contract or its declared promotion screen.

## Evidence

| Line | Evidence period | Outcome |
| --- | --- | --- |
| `baseline-ridge-v1` / `causal-v1` | 2022-2024 walk-forward | Rejected by the terminal development acceptance report; no candidate was promoted. |
| `baseline-ridge-v2` calibration | 2022-2024 development replay | Rejected; only 8 positive folds per horizon and portfolio return `-7.19%`. |
| Additive spline ridge | Pre-2022 screen | Rejected; 30 and 60 minute means deteriorated. |
| `baseline-ridge-v3` / `causal-v2` | 2021 one-year screen | Rejected by all three promotion gates. |

The final microstructure run is the authoritative terminal artifact:

| Field | Value |
| --- | --- |
| Pipeline | `microstructure-screen-pipeline-v1` |
| Pipeline run | `sha256-76ca5d4051004414428fe4aff5a2a614a37cdfaca1106a971aa736118702d325` |
| Runtime digest | `sha256:a24cd0b03331eb743c00c077a292d8cc40553f9b0732949224eb5876c3201f9d` |
| MLflow run | `be457c25d5d24ff988b840ae88978491` |
| Technical result | 43/43 stages, exit code 0, no restart |
| Acceptance | `accepted=false`, `promotion_authorized=false` |
| Pooled means, 15/30/60m | `-0.004315/-0.013522/-0.028682` bps |
| Positive symbols, 15/30/60m | `1/3/3`, required `6/6/6` |
| Portfolio | `-2.96%`, 3.25% maximum drawdown, 127,884 trades |
| Required next action | `stop_microstructure_research_line` |
| Scientific result SHA-256 | `7166d5652a6ae192fc9a1bc6576614f0e8b1da5c90d0b90106f52deaeeeb901a` |
| Pipeline config SHA-256 | `2e8aea35abd291f218b0187056db8c2ef8013013bffefd7dd40094fbe411b96a` |
| Acceptance config SHA-256 | `c6da369f27f8b3fa69a878e8d0e608fc062f818967b21408843fac9a354d5792` |

The minimum-trade gate also failed for AUDUSD, EURCHF, GBPJPY, GBPUSD, and
USDCAD at 15 minutes. This was not the only reason for rejection: pooled means
and cross-symbol consistency independently failed at every horizon.

## Data-Use Ledger

| Interval | Campaign 1 use after closeout |
| --- | --- |
| `[2018-01-01, 2025-01-01)` | Observed development history; training and engineering only, never future confirmation evidence. |
| `[2021-01-01, 2022-01-01)` | Observed microstructure screen; cannot be reused to tune `causal-v2`. |
| `[2022-01-01, 2025-01-01)` | Observed walk-forward development; cannot be presented as an untouched holdout. |
| `[2025-01-01, 2026-03-11)` | Existing locked test; no authorized pipeline recorded access, and it remains permanently forbidden to development. |

No inference about locked-test performance is made from the development
failures. The locked dataset is not repurposed as development data. The
no-access statement is based on the authorized pipeline records; it is not a
substitute for an independent custodial attestation.

## Rejected Follow-Ups

- Do not tune `causal-v2` windows, imbalance formulas, ridge alpha, or action
  threshold against the 2021 result.
- Do not run `causal-v2` on 2022-2024 after its failed promotion screen.
- Do not freeze a Campaign 1 candidate or invoke the locked evaluator.
- Do not treat a new transformation of the same per-pair summaries as fresh
  confirmation evidence on an already observed period.

## Operational Retention

- Retain the final acceptance envelope, stage markers, MLflow run, execution
  record, and content-addressed inputs.
- Retain the run's working storage until the terminal artifacts have been
  backed up.
- The execution record may expire under the operator's retention policy;
  deleting it does not authorize a rerun.
- Any reproduction must use the same run identity and verified checkpoints and
  is an engineering audit, not another scientific attempt.

## Lessons Carried Forward

- The tested `causal-v1` and `causal-v2` summaries did not produce stable
  executable directionality under their frozen models and periods.
- The tested affine calibration and additive spline basis did not rescue those
  specific model lines.
- The tested intrabar activity features changed trade frequency but not
  cross-pair consistency or pooled profitability.
- A new campaign must test a new information set and must obtain new,
  prospectively sealed confirmation data.

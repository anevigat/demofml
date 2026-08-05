# Research Campaign 2 v2: Prospective Cross-Pair Factors

Status: engineering implementation authorized; collection, fitting, scoring,
raw access, and evaluation are not authorized

Declaration date: 2026-08-05

Contract family: `prospective-cross-pair-factor-v2`

## Reason For A New Attempt

Campaign 2 v1 is closed because its qualification interval cannot reach the
frozen 95% completeness threshold and no trusted receipt-time collection
evidence exists. It remains historical evidence and must not be backfilled,
retimestamped, or reused as v2 evidence. Version 2 is a new prospectively
declared attempt with new artifact identities and later intervals.

## Frozen Intervals

| Role | Half-open UTC interval | Policy |
| --- | --- | --- |
| Historical fit | `[2018-01-01, 2025-01-01)` | Contract reference only; fitting is not authorized. |
| Existing locked test | `[2025-01-01, 2026-03-11)` | Permanently forbidden for input, context, labels, and metrics. |
| Unused gap | `[2026-03-11, 2026-09-01)` | Not v2 qualification evidence and not eligible for backfill. |
| Collector qualification | `[2026-09-01, 2027-03-01)` | Trusted receipt-time, schema, coverage, and infrastructure checks only. |
| Prospective source | `[2027-03-01, 2028-03-01)` | Custodial one-shot holdout. |
| Prospective decisions | `[2027-03-01, 2028-02-29T22:55:00Z)` | Complete 65-minute information window remains in source. |

State-only context is exactly
`[2027-02-28T18:00:00Z, 2027-03-01T00:00:00Z)`. It may initialize trailing
features but cannot generate a decision, label, score, return, or metric.

## Frozen Technical Contract

Version 2 changes only the campaign identity, artifact-set identities, and
declared intervals. The pairs, 5-minute calendar, receipt-time semantics,
cross-pair transform, feature order, control and candidate shapes, horizons,
missingness treatment, resource limits, endpoint, operational gates, and
one-shot evaluation rules remain exactly as declared in
`campaign-2-prospective-factor-plan.md` before v1 closed.

Qualification still requires at least 95% complete synchronized eight-pair
boundaries globally and in every UTC month, no more than 36 consecutive missing
boundaries, deterministic output, no more than one second of feature-build time
per boundary, and no more than 1 GiB peak resident memory. Qualification cannot
grant collection, fitting, scoring, raw-access, or evaluation capability.

## Custody And Stop Rules

Custody must satisfy `campaign-2-onprem-custody-requirements.md` in a separately
administered on-prem tenant. Shared development MinIO is not an eligible
boundary. There is no authorized collector or scoring workload in this
repository.

Any use of v1 artifacts as v2 evidence, inferred receipt timestamps, access to
the locked interval, raw or outcome-bearing access, post-freeze behavior change,
or premature evaluation closes v2. A later attempt requires another identity
and another interval declared before collection.

## Authorization Boundary

Approval on 2026-08-05 authorizes this v2 engineering contract, data-free
verification, synthetic tests, outcome-free ledgers, and non-authorizing custody
and qualification record formats. Collection, fitting, scoring, raw prospective
access, and evaluation remain false and require separate explicit authorization.

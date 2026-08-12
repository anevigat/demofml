# Campaign 2 v1 Qualification Blocker

Status: fail-closed; Campaign 2 v1 cannot start qualification or collection

Observed at: `2026-08-05T09:42:01Z`

## Frozen Requirement

Campaign 2 v1 fixes its qualification interval at
`[2026-03-11T00:00:00Z, 2026-09-01T00:00:00Z)` and requires at least 95%
complete expected boundaries globally and in every UTC month, with no more than
36 consecutive missing boundaries. Trusted receipt timestamps are prospective
facts and cannot be reconstructed from historical event timestamps.

The frozen calendar contains 35,387 expected boundaries. Qualification therefore
requires at least 33,618 complete boundaries.

## Feasibility Check

At the observation time:

- 30,084 expected boundaries had already elapsed.
- 5,303 expected boundaries remained.
- Perfect collection over every remaining boundary could cover at most
  `5303 / 35387 = 14.9857292226%` of the frozen interval.
- The minimum unavoidable shortfall was 28,315 complete boundaries.

The repository contains no Campaign 2 collector or collection manifest. The
operator's on-prem environment contains only the three data-free Campaign 2
engineering-verification workloads; it has no Campaign 2 collector Deployment,
StatefulSet, DaemonSet, Job, or CronJob. This check inspected only repository and
Kubernetes resource metadata and did not access prospective or raw data.

## Decision

Campaign 2 v1 is blocked unless a separately administered on-prem custodian can
provide pre-existing, valid, trusted-receipt evidence covering the elapsed
qualification interval. No such evidence is currently known or referenced.

Backfill, inferred receipt timestamps, use of event timestamps as receipt time,
or substitution of historical/locked-test data is prohibited. Collection,
fitting, scoring, evaluation, raw access, and qualification completion remain
false.

Without qualifying pre-existing evidence, the next admissible step is a new
campaign version with a prospectively declared qualification start, holdout
interval, and freeze deadline. Exact dates require explicit approval and must
not mutate or reuse v1 identities.

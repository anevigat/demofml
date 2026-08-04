# Research Campaign 2: Prospective Cross-Pair Factors

Status: engineering implementation authorized; fitting, scoring, and evaluation
are not authorized

Plan date: 2026-08-03

Working contract family: `prospective-cross-pair-factor-v1`

## Objective

Test one new hypothesis: cross-pair consistency residuals derived from a
synchronous eight-pair quote cross-section contain prospective executable
directional information beyond the existing independent per-pair features.

This is a new cross-pair information set derived from the same raw quote source,
not a new raw data source. Version 1 keeps the horizons, costs, threshold,
portfolio accounting, and per-symbol static ridge unchanged. It uses the new
`prospective-executable-v1` label contract because publication receipt time,
rather than decision event time, determines entry eligibility.
A paired `causal-v1` control and a `causal-v1 + cross-pair` candidate will use
the same decisions so the incremental contribution is directly measurable.

The claim is prospective predictive association, not economic causality.

## Data Boundaries

Campaign 2 will not claim confirmation on any period whose outcomes were
observed during Campaign 1.

| Role | Half-open UTC interval | Policy |
| --- | --- | --- |
| Historical fit | `[2018-01-01, 2025-01-01)` | Deterministic model fitting only; no supervised metric may select version 1. |
| Existing locked test | `[2025-01-01, 2026-03-11)` | Permanently forbidden; no input, context, label, or metric may be read. |
| Collector qualification | `[2026-03-11, 2026-09-01)` | Schema, receipt-time, coverage, and infrastructure checks only; no return metrics. |
| Prospective source interval | `[2026-09-01, 2027-09-01)` | Custodial one-shot holdout. |
| Prospective decision interval | `[2026-09-01, 2027-08-31T22:55:00Z)` | Decisions whose complete 65-minute information window remains in the source interval. |

The qualification collector may provide state-only context from
`[2026-08-31T18:00:00Z, 2026-09-01T00:00:00Z)`. Context may initialize trailing
features and the prior close needed by the first return. It cannot generate a
decision, label, score, return, or metric and remains covered by the
qualification custody policy. This context does not overlap the existing
locked test.

The exact candidate, control, preprocessing, calendar, data manifests,
acceptance code, and runtime must be frozen before
`2026-09-01T00:00:00Z`. Evaluation is prohibited until the prospective source
interval closes and its terminal manifest is published.

## Custody And One-Shot Enforcement

Logical separation of predictions and labels is insufficient because raw quotes
can reconstruct outcomes. The prospective raw dataset and all outcome-bearing
derivatives must therefore be inaccessible to the researcher identity.

Before scoring begins, an external custodian must provide:

- A write-only or append-only prediction sink for the scoring identity.
- A raw-data identity available only to the custodial scoring workload.
- Signed receipt timestamps and an append-only prediction publication log.
- Immutable object versions and an externally timestamped candidate bundle.
- A consumable grant binding candidate, control, calendar, data allowlist,
  acceptance code, and runtime digest.
- A terminal claim store that researchers cannot delete or recreate.

For the AWS provider contract, every raw object version must remain under S3
Object Lock `COMPLIANCE` retention through at least `2028-09-01T00:00:00Z`.
The exact metadata-only role, signed policy digests, KMS verification key,
bucket, prefix, and prohibited API capabilities are specified in
`campaign-2-aws-custody-requirements.md`. This repository does not provision or
authorize those external resources.

Each prediction must be committed before its horizon resolves. The commitment
records model ID, symbol, horizon, score, action, decision boundary, maximum
feature receipt time, publication time, and artifact hash. Researchers may
observe service health and aggregate missingness only, not scores, actions, raw
quotes, labels, returns, attribution, or portfolio state.

Any raw holdout access, outcome reconstruction, partial metric query, or grant
reuse rejects and closes `v1`. It does not move the same attempt to a later
interval. A future attempt requires a new version, a new interval declared
before collection, and explicit multiplicity treatment.

## Operational Causality

Provider event timestamps alone do not establish that all cross-pair inputs were
available before entry. The prospective collector must add a trusted UTC
`received_at` timestamp at ingestion and preserve provider timestamp separately.

For a bar ending at market boundary `T`:

1. All eight bars must be complete.
2. A symbol bar `[T - 5 minutes, T)` becomes final only when the collector
   receives that symbol's next quote with provider timestamp greater than or
   equal to `T`. Version 1 does not accept synthetic or heartbeat watermarks.
3. `bar_finalized_at` is the trusted `received_at` of that watermark. Any later
   message older than the watermark is quarantined, records the affected
   boundary as invalid, and terminally fails that symbol stream. It is a
   qualification failure before launch and rejects version 1 after launch. It
   is never inserted into an already published bar.
4. `feature_available_at` is the maximum `bar_finalized_at` across all eight
   symbols plus a frozen one-second computation allowance.
5. The prediction must be durably published no earlier than
   `feature_available_at` and before any eligible entry.
6. `published_at` must be no later than `T + 5 minutes`; otherwise the expected
   opportunity is recorded flat and missing.
7. Entry is the first received executable quote after `published_at` and no
   later than `T + 5 minutes`.
8. Exit is the first received executable quote at or after `T + horizon` and no
   later than `T + horizon + 5 minutes`.

Five minutes is a maximum quote wait, not a fixed latency. The candidate bundle
uses nanosecond UTC receipt timestamps and strictly increasing unsigned ingest
sequence numbers. Receipt time and sequence must be monotone. Provider time must
be monotone within each symbol stream; reordered messages are quarantined and
cannot mutate finalized state. Every persisted bar records the provider time and
ingest sequence of its closing quote watermark. The frozen maximum provider
clock lead is 100 milliseconds; the custody attestation must demonstrate that
bound and identify the collector clock source.

If trusted receipt time cannot be collected for all eight pairs, Campaign 2 is
blocked before the holdout. Historical event-time simulation may test code but
cannot substitute for prospective executable evidence.

## Exact Cross-Pair Transform

### Pair and currency ordering

Pairs are ordered `AUDUSD`, `EURCHF`, `EURJPY`, `EURUSD`, `GBPJPY`, `GBPUSD`,
`USDCAD`, `USDJPY`. Non-USD currency columns are ordered `AUD`, `CAD`, `CHF`,
`EUR`, `GBP`, `JPY`; USD is anchored at zero.

At boundary `T`, pair input is the one-bar close-to-close log mid return:

```text
y_pair(T) = log(mid_close_pair(T) / mid_close_pair(T - 5 minutes))
```

The fixed design row contains `+1` for the base currency, `-1` for the quote
currency, and zero elsewhere; USD contributes no column. The six-column design
has full rank. Version 1 therefore uses the following closed-form float64
solution, not a library least-squares routine or a numerical rank tolerance.
For inputs `a, x, u, e, v, g, d, j` in the fixed pair order:

```text
AUD = a
CAD = -d
JPY = (-u - v + e + g - 2j) / 4
EUR = (u + e + JPY) / 2
GBP = (v + g + JPY) / 2
CHF = EUR - x
```

Residuals are then computed by direct float64 substitution in fixed pair order.
Dispersion uses a deterministic two-pass population standard deviation. Golden
vectors freeze operation order before holdout start.

```text
strength(T) = argmin_s ||A s - y(T)||_2
residual(T) = y(T) - A strength(T)
```

The graph leaves two residual degrees of freedom. The hypothesis concerns those
cross-pair consistency residuals; currency-strength estimates are coordinate
features, not independently observed economic factors.

### Candidate columns

The control contains the existing `causal-v1` columns. The candidate appends,
in this fixed order, for each pair:

- `base_strength_1`, `base_strength_sum_3`, `base_strength_sum_12`.
- `quote_strength_1`, `quote_strength_sum_3`, `quote_strength_sum_12`.
- `pair_factor_residual_1`, `pair_factor_residual_sum_3`,
  `pair_factor_residual_sum_12`.
- `cross_pair_residual_dispersion_1`.
- `cross_pair_residual_dispersion_mean_3` and
  `cross_pair_residual_dispersion_mean_12`.

Dispersion is the population standard deviation of the eight current residuals.
Trailing sums and means include the current completed cross-section. All windows
reset after an incomplete cross-section. No forward fill is allowed.

Control and candidate are scored only on the same fixed decision calendar.
Incomplete cross-sections generate an explicit missing prediction recorded as a
flat zero-return opportunity for both arms; they cannot disappear from means.

## Frozen Model Shape

| Parameter | Control and candidate value |
| --- | --- |
| Model | Per-symbol ridge |
| Alpha / solver | `1.0` / `lsqr` |
| Imputation | Historical-training median |
| Standardization | Historical-training only |
| Action threshold | `0.0` bps |
| Horizons | `15, 30, 60` minutes |
| Symbols | Canonical eight pairs |
| Portfolio | Existing normalized sleeve accounting and causal risk policy |

No pooled model, hyperparameter sweep, nonlinear transform, calibration, or
threshold selection belongs to version 1. No supervised historical return,
hit-rate, trade, attribution, or portfolio metric may be computed to choose or
modify the listed features. Model fitting may emit only coefficients,
preprocessing state, optimization status, and causal/data-quality invariants.

Before holdout start, one externally timestamped content-addressed bundle must
contain:

- This protocol and every behavior-affecting config.
- Historical dataset allowlist and hashes.
- Exact feature order and synchronized decision calendar rules.
- Fitted control and candidate coefficients, medians, and scalers.
- Source commit, runtime digest, random seed, and numeric-library versions.
- Prospective collector schema, expected calendar, missingness policy, gates,
  and terminal marker policy.

Any bundle change consumes version 1 before prospective scoring begins.

## Coverage And Completeness

The expected decision calendar runs from Sunday 17:05 inclusive through Friday
16:00 exclusive in `America/New_York`, converted to UTC with the timezone
database version frozen in the candidate bundle. The Sunday boundary requires
the first complete `[17:00, 17:05)` bar. The Friday cutoff makes the last
decision 15:55 so its 60-minute horizon and five-minute exit allowance end no
later than the 17:00 market close.

Version 1 requires:

- At least 95% complete eight-pair cross-sections globally and in every calendar
  month.
- No more than 36 consecutive missing expected bars.
- No more than 5% unresolved executable labels per symbol/horizon.
- Exactly one committed control and candidate record per expected
  symbol/horizon opportunity, including explicit flat records for missing input.
- All eight symbols present for the complete prospective decision interval.
- Outcome-free qualification feature construction no slower than one second per
  expected boundary and peak resident memory no greater than 1 GiB in the frozen
  runtime.

Coverage is computed without returns and may be monitored by the custodian.
Failure discovered before the holdout blocks launch. Failure after holdout start
rejects version 1 and does not authorize a missingness-policy change.

## Prospective Endpoint And Gates

The primary endpoint is the candidate-minus-control difference in pooled
30-minute executable return per expected decision. It must have:

- Point estimate at least `0.01` bps.
- A strictly positive lower bound from a frozen 95% weekly-block bootstrap with
  10,000 paired resamples, Monday 00:00 UTC block boundaries, and seed 1729.

The 30-minute endpoint is primary; 15 and 60 minutes are secondary operational
gates and are not presented as independent confirmatory tests. Symbols are also
not treated as independent replications because they share currencies.

The candidate must additionally pass every existing operational gate:

- Positive pooled executable mean at all three horizons.
- At least six of eight positive symbols at every horizon.
- At least 100 trades in every symbol/horizon cell.
- Positive portfolio total return.
- Maximum drawdown below 10%.
- Realized annual volatility between 7.5% and 12.5%.
- Maximum leverage no greater than 2x and no drawdown halt.
- Complete provenance, exact artifact reconciliation, and no forbidden-period
  access.

The portfolio is not computed or updated for researchers during the holdout.
After the source interval closes, the custodian performs one causal replay from
the committed prediction stream and releases only the terminal acceptance
envelope.

## Implementation Milestones

Engineering status as of 2026-08-04: receipt-time schemas, bars, synchronized
features, expected-opportunity coverage, strict config loading, append-only
manifest-chain validation, an engineering-only bundle, and an outcome-free
prequalification envelope are implemented with synthetic tests. No collection
manifest has been published, no real collector qualification has run, and no
bundle has been externally timestamped. The engineering bundle contains no
models or data and cannot serve as the scoring candidate bundle required below.
Its local checks cannot set `qualification_complete` without a separately
trusted bundle ID and external attestation.

The AWS metadata-only preflight verifier is also implemented with fake clients.
It has no endpoint, credentials, or raw-read/write API and has not run against
AWS. A real provider config, one-session role credentials, signed KMS
attestation, archived policy digests, Object Lock bucket, and externally trusted
runtime clock remain absent and externally controlled.

1. Approve this hypothesis shape and custody model.
2. Publish a content-addressed append-only manifest for post-lock collection.
3. Implement receipt-time ticks, synchronized bars, factor features, paired
   feature construction, explicit missing opportunities, and later the
   separately authorized paired scoring service.
4. Add algebraic golden vectors, permutation, gap-reset, streaming,
   publication-time, and no-lookahead tests.
5. Qualify schema, coverage, determinism, runtime, and memory without supervised
   metrics.
6. Fit control and candidate once without exposing historical performance.
7. Freeze and externally timestamp the complete candidate bundle before
   2026-09-01.
8. Obtain separate explicit authorization for custodial prospective scoring.
9. Consume one evaluation grant after 2027-09-01 and publish acceptance or
   rejection without retries.

## Stop Rules

- Failure of the primary endpoint or any operational gate rejects version 1.
- Any outcome-bearing access before terminal evaluation rejects version 1.
- Any candidate, calendar, data, custody, or code mutation after freeze rejects
  version 1.
- Failure to freeze the complete bundle before 2026-09-01 cancels version 1;
  launch requires a new version and a later prospectively declared interval.
- Any technical failure may resume only from verified checkpoints with the same
  bundle and grant; changing behavior creates a failed attempt, not a retry.
- Existing locked-test data never substitutes for the prospective holdout.

## Authorization Required

The user approved implementation of the hypothesis and custody architecture on
2026-08-04. This authorizes schemas, deterministic transforms, synthetic tests,
coverage checks, append-only manifest and bundle contracts, qualification
claim envelopes, and outcome-free opportunity ledgers only. It does not authorize
fitting, scoring, raw prospective collection, or access to outcomes.
Prospective scoring requires a second approval after the complete bundle is
frozen. Evaluation remains prohibited until the prospective source interval
closes.

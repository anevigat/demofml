# Campaign 2 Cross-Pair Historical Screen 2020 Result

Status: terminal exploratory rejection; no promotion authorized

## Execution

- Screen contract: `campaign2-cross-pair-historical-screen-2020-v1`.
- Frozen config SHA-256:
  `1a2bdf972bbafc350ee4b4038580ebaf44c6148a5e55c9121fa9ad810d5d4d2b`.
- Input development run:
  `sha256-c8903538bf0c2a81cd866e318e4eb6ac0a8abf21fedd96eee8b10322ebf41758`.
- Runtime image:
  `anevigat/demofml@sha256:48451479b2234be7fc84c4a0fc00fd92966f0253fd16b13f05fa2e75a03896fd`.
- Source revision: `0d676be7c9ab7be9b1e67ae90bc0ce4b4d3b53ef`.
- Successful Job: `demofml-campaign2-cross-pair-screen-2020-v2`.
- Pod: `demofml-campaign2-cross-pair-screen-2020-v2-jhlqs`.
- Node: `[REDACTED-NODE]`.
- Result: 1/1 complete, zero restarts.
- Report ID:
  `sha256-02e736e0126adae26b5221c5a7366b43af853cdd1ae183bc069f55661a43fdc9`.

The report remains at
`/work/campaign2-cross-pair-historical-screen-2020-v1/report.json` on the
retained development PVC. Its content ID was independently recomputed from a
read-only pod and reconciled exactly.

## Failed Technical Attempt

Job `demofml-campaign2-cross-pair-screen-2020-v1` failed before loading screen
labels or calculating return metrics. It rejected valid historical source bars
outside the prospective decision calendar instead of filtering them. Commit
`0d676be` added source-only filtering without changing input hashes, model,
intervals, expected-opportunity denominator, or scientific contract. The failed
Job and pod remain retained as evidence.

## Results

All means below are executable basis points per expected opportunity, including
zero return for missing, unresolved, or unscored expected opportunities.
They are not compounded portfolio returns.

| Horizon | Control mean bps | Candidate mean bps | Candidate-control delta bps | Candidate positive symbols | Candidate trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 minutes | -0.0060849882 | -0.0040444984 | +0.0020404899 | 2/8 | 12,609 |
| 30 minutes | -0.0105577016 | -0.0055030137 | +0.0050546879 | 1/8 | 25,056 |
| 60 minutes | -0.0426638572 | -0.0347069780 | +0.0079568791 | 2/8 | 66,388 |

The expected screen calendar contained 74,767 boundaries and 598,136 pooled
symbol opportunities per horizon. The candidate and control were both scored
on 589,040 factor-ready symbol opportunities per horizon.

Candidate symbol means were positive only for:

- 15 minutes: `EURUSD`, `USDCAD`.
- 30 minutes: `USDCAD`.
- 60 minutes: `GBPJPY`, `USDCAD`.

The cross-pair features improved the pooled mean relative to the control at all
three horizons, but the candidate remained negative at every horizon and lacked
cross-symbol consistency.

## Decision

Reject the historical candidate. It does not show historical executable
profitability under the frozen 2020 screen and is not promoted. No additional
historical retuning, threshold search, feature selection, or rerun is authorized
from this result.

Campaign 2 v2 remains exploratory-only and cannot be used for a confirmatory
claim. Prospective collection, prospective scoring, locked-test access, and
promotion remain false. A materially different attempt requires a new declared
hypothesis and version before inspecting another outcome period.

# Campaign 2 On-Prem Engineering Verification

Status: technical verification passed; no research or data capability authorized

Run date: 2026-08-04

## Operator Execution v1

- Start: `2026-08-04T10:17:46Z`.
- Completion: `2026-08-04T10:17:55Z`.
- Result: `Complete`, one succeeded pod, zero restarts, `backoffLimit: 0`.

The workload mounted no Secret, ServiceAccount token, data PVC, MinIO config,
MLflow config, or external custody credential. A dedicated deny-all
NetworkPolicy selected the workload's pod. The base image and offline overlay
were immutable and digest-bound. Execution environment details (cluster,
namespace, job/pod/node identifiers) are operator-internal and intentionally
not recorded here.

## Identities

- Base image:
  `anevigat/demofml@sha256:a24cd0b03331eb743c00c077a292d8cc40553f9b0732949224eb5876c3201f9d`.
- Runtime overlay:
  `sha256:abccb11397f9002aa6b7897ff1bd56d3d5747b769801e97a2204e400932a6301`.
- Contract set:
  `sha256:42af2130e90b9434d5f7daec80a8343db911be02d1c86e0113447c4ccd9ad378`.
- Verification:
  `sha256-17ac3ff8eda074f5c5a897e1f40ea38b1dd1b394111a65c7c8d3147de6123028`.

## Checks

All five engineering checks passed:

- Exact engineering config and five contract hashes.
- Locked interval fixed at `[2025-01-01, 2026-03-11)` and excluded.
- Closed-form cross-pair golden vector with maximum residual
  `3.469446951953614e-18` and zero maximum strength error.
- Frozen weekly calendar with 1,427 boundaries.
- Authorization boundary with collection, fitting, scoring, evaluation, and raw
  access all false.

The runtime reported Python 3.12.13, NumPy 2.5.1, PyArrow 25.0.0,
scikit-learn 1.9.0, and tzdata 2025.2.

## Image-Native Verification v2

The overlay-free verification ran after image workflow
[`30901637929`](https://github.com/anevigat/demofml/actions/runs/30901637929)
completed CI, container smoke testing, HIGH/CRITICAL Trivy scans, SBOM and
provenance generation, and multi-platform publication.

- Source revision: `f6ae9a59798b4aa12c93a0b98c170c1e93662457`.
- Runtime image:
  `anevigat/demofml@sha256:461660d383b7af4f621d139185b6fb48249b8c73c41e166df7f992e44263fcf5`.
- Start: `2026-08-04T10:57:22Z`.
- Completion: `2026-08-04T10:58:06Z`.
- Result: `Complete`, one succeeded pod, zero restarts, `backoffLimit: 0`.
- Verification:
  `sha256-e028d74b57003ea41f24fcac85728c75543ac115de57856e7fdc5ed1e43ac8ab`.

The v2 workload used the contracts and application installed in the immutable
runtime image. It had no init container, runtime overlay, Secret, ConfigMap,
ServiceAccount token, data PVC, MinIO config, MLflow config, or external custody
credential.
Its only volume was an ephemeral `/tmp` `emptyDir`; a deny-all NetworkPolicy
continued to select its pod. All five engineering checks and all authorization
denials matched v1.

## On-Prem Custody Verification v3

The provider correction was built and published by image workflow
[`30905803174`](https://github.com/anevigat/demofml/actions/runs/30905803174),
which passed CI, container smoke testing, HIGH/CRITICAL Trivy scans, SBOM and
provenance generation, and multi-platform publication.

- Source revision: `c161ebe44341b178bdd25271bd6160eef2bd45e9`.
- Runtime image:
  `anevigat/demofml@sha256:c86baaebef31cf0c696bb6bf570a97eb1a555fdc823cdb45b794e47047298d0e`.
- Start: `2026-08-04T12:38:49Z`.
- Completion: `2026-08-04T12:39:34Z`.
- Result: `Complete`, one succeeded pod, zero restarts, `backoffLimit: 0`.
- Protocol SHA-256:
  `923e25e6570a54a537f92956897c9c7fb639ec87ed2fc3f296280e08e61baf7d`.
- Verification:
  `sha256-3e86db4f30021a0fe6011d0d5e28616de45a6526fc19bcdab01f1def630455ef`.

V3 binds Campaign 2 to on-prem custody requirements and contains no AWS custody
module or cloud-provider preflight contract. The workload mounted no Secret,
ConfigMap, ServiceAccount token, data PVC, MinIO configuration, MLflow
configuration, or external custody credential. Its only volume was an
ephemeral `/tmp` `emptyDir`, and a deny-all NetworkPolicy selected its pod.
All five engineering checks passed while collection, fitting, scoring,
evaluation, raw access, and qualification completion remained false.

## Interpretation

Job completion establishes only that the Campaign 2 engineering contracts,
schemas, deterministic algebra, calendar, and authorization guards execute in
the on-prem Kubernetes runtime. It does not complete collector qualification,
freeze a scoring candidate, authorize collection or model fitting, access raw
prospective data, produce scores, or evaluate outcomes.

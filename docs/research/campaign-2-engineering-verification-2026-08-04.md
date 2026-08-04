# Campaign 2 On-Prem Engineering Verification

Status: technical verification passed; no research or data capability authorized

Run date: 2026-08-04

## Kubernetes Execution v1

- Context: `[REDACTED-CONTEXT]`.
- Namespace: `demofml`.
- Job: `demofml-campaign2-engineering-verify-v1`.
- Pod: `demofml-campaign2-engineering-verify-v1-6gnmr`.
- Node: `[REDACTED-NODE]`.
- Start: `2026-08-04T10:17:46Z`.
- Completion: `2026-08-04T10:17:55Z`.
- Result: `Complete`, one succeeded pod, zero restarts, `backoffLimit: 0`.

The Job mounted no Secret, ServiceAccount token, data PVC, MinIO config, MLflow
config, or external custody credential. A dedicated deny-all NetworkPolicy
selected the Job pod. The base image and offline overlay were immutable and
digest-bound.

## Identities

- Base image:
  `anevigat/demofml@sha256:a24cd0b03331eb743c00c077a292d8cc40553f9b0732949224eb5876c3201f9d`.
- Runtime overlay:
  `sha256:abccb11397f9002aa6b7897ff1bd56d3d5747b769801e97a2204e400932a6301`.
- Contract set:
  `sha256:42af2130e90b9434d5f7daec80a8343db911be02d1c86e0113447c4ccd9ad378`.
- Verification:
  `sha256-17ac3ff8eda074f5c5a897e1f40ea38b1dd1b394111a65c7c8d3147de6123028`.

The immutable ConfigMaps were:

- `demofml-campaign2-wheels-abccb11397f9`.
- `demofml-campaign2-contracts-42af2130e90b`.

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
- Job: `demofml-campaign2-engineering-verify-v2`.
- Pod: `demofml-campaign2-engineering-verify-v2-ct986`.
- Node: `[REDACTED-NODE]`.
- Start: `2026-08-04T10:57:22Z`.
- Completion: `2026-08-04T10:58:06Z`.
- Result: `Complete`, one succeeded pod, zero restarts, `backoffLimit: 0`.
- Verification:
  `sha256-e028d74b57003ea41f24fcac85728c75543ac115de57856e7fdc5ed1e43ac8ab`.

The v2 Job used the contracts and application installed in the immutable
runtime image. It had no init container, runtime overlay, Secret, ConfigMap,
ServiceAccount token, data PVC, MinIO config, MLflow config, or external custody
credential.
Its only volume was an ephemeral `/tmp` `emptyDir`; the deny-all NetworkPolicy
continued to select the pod. All five engineering checks and all authorization
denials matched v1.

## Interpretation

Job completion establishes only that the Campaign 2 engineering contracts,
schemas, deterministic algebra, calendar, and authorization guards execute in
the on-prem Kubernetes runtime. It does not complete collector qualification,
freeze a scoring candidate, authorize collection or model fitting, access raw
prospective data, produce scores, or evaluate outcomes.

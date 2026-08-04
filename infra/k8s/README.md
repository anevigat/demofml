# Kubernetes Infrastructure

This deployment is intentionally restricted to the existing `demofml`
namespace. Every workload targets the dedicated worker through the required
node selector and toleration.

## Components

- MinIO with a 200 GiB `local-path` PVC.
- PostgreSQL with a 10 GiB `local-path` PVC.
- MLflow backed by PostgreSQL with artifacts proxied to MinIO.
- A namespace-scoped ServiceAccount with API token mounting disabled.
- A bootstrap Job that creates private buckets and a scoped application user.
- A smoke Job that verifies S3 I/O and MLflow metric/artifact persistence.
- A digest-pinned, resumable development pipeline with a dedicated 64 GiB PVC.
- A TLS Ingress exposing only the MinIO S3 API on the internal network.

## Deploy

The secret script generates credentials in memory and writes them directly to
Kubernetes. It never prints or stores their values locally.

```bash
./infra/k8s/create-secrets.sh
./infra/k8s/deploy.sh
./infra/k8s/run-smoke.sh
```

Inspect resources:

```bash
kubectl get pods,services,pvc,jobs -n demofml
```

## Development Pipeline

The work PVC is separate from the Job so it can be created without starting the
experiment. Create it at any time, but apply the Job only after the publisher has
uploaded all 23 objects and the immutable source `manifest.json`:

```bash
kubectl apply -f infra/k8s/jobs/development-pipeline-pvc.yaml

# Run only after publication reports 23/23 and a verified manifest location.
kubectl apply -f infra/k8s/jobs/development-pipeline-v2.yaml
kubectl get job,pod,pvc -n demofml \
  -l app.kubernetes.io/name=development-pipeline
```

The v2 Job runs the Phase 12 acceptance and profiling contract. The v1 manifest is
retained as the immutable Phase 11 workload and must not be launched alongside
v2. The Job has no API token, uses only the `demofml` namespace, targets the
dedicated worker, and pins both its image and experiment code identity to the same
runtime digest. `backoffLimit: 0` prevents an automatic duplicate attempt. To
resume after a failure, delete only the v2 Job and apply it again; verified
checkpoints remain on `demofml-development-work-v1`. Never delete that PVC during
a retry.

## Microstructure Screen

The pre-2022 screen is a separate digest-pinned, resumable Job. It reuses the
development PVC, but has a distinct pipeline identity and never reads decisions
from 2022 onward. Launch it only after the published image has passed CI and
scanning, and only when no other development workload is active:

```bash
kubectl apply -f infra/k8s/jobs/development-pipeline-pvc.yaml
kubectl apply --dry-run=server \
  -f infra/k8s/jobs/microstructure-screen-2021-v1.yaml
kubectl apply -f infra/k8s/jobs/microstructure-screen-2021-v1.yaml
kubectl get job,pod -n demofml \
  -l app.kubernetes.io/name=microstructure-screen
```

The Job executes `microstructure-screen-pipeline-v1` across all eight symbols
and applies the frozen promotion gates after 43 verified stages. Job completion
means technical success only; inspect the final acceptance artifact or MLflow
metric `microstructure_screen_accepted` for the scientific decision. A rejected
screen ends this research line and does not authorize a 2022-2024 or locked-test
run.

## Campaign 2 Engineering Verification

The Campaign 2 engineering Job is data-free and runs directly in the on-prem
`demofml` namespace. It mounts immutable content-addressed wheel and contract
ConfigMaps, has deny-all networking, receives no Secret or data PVC, and keeps
every collection/model/scoring/evaluation authorization false.

The checked-in v1 Job references the exact immutable ConfigMaps from the
2026-08-04 run. They must already exist in the namespace with the annotation
digests recorded below; do not recreate those names with different bytes. A new
overlay or contract set requires new content-addressed ConfigMap names, a new Job
version, and a new verification record.

```bash
kubectl --context admin@intechsol-k8s apply --dry-run=server \
  -f infra/k8s/jobs/campaign2-engineering-verify-v1.yaml
kubectl --context admin@intechsol-k8s apply \
  -f infra/k8s/jobs/campaign2-engineering-verify-v1.yaml
kubectl --context admin@intechsol-k8s -n demofml wait \
  --for=condition=complete \
  job/demofml-campaign2-engineering-verify-v1 --timeout=600s
```

The exact 2026-08-04 execution and verification identity are recorded in
`docs/research/campaign-2-engineering-verification-2026-08-04.md`.

The private S3 Ingress is deliberately excluded from Kustomize so its hostname
never appears in the public repository. Configure it locally:

```bash
export DEMOFML_INGRESS_HOST="<private-hostname>"
./infra/k8s/deploy-ingress.sh
unset DEMOFML_INGRESS_HOST
```

It routes the root path directly to `minio:9000`; no prefix middleware is used
because modifying request paths would invalidate S3 signatures. The MinIO
console, MLflow, and PostgreSQL remain unexposed.

The Ingress uses a namespace-local self-signed certificate. Export its public
certificate before configuring an S3 client:

```bash
kubectl get secret demofml-minio-tls -n demofml \
  -o go-template='{{index .data "tls.crt" | base64decode}}' \
  > ~/.config/demofml-minio-ca.crt
```

The certificate is public material; its private key remains only in the
Kubernetes TLS Secret. S3 clients must use the exported certificate as their CA
bundle rather than disabling certificate verification.

Access internal UIs from this machine:

```bash
kubectl port-forward -n demofml service/mlflow 5000:5000
kubectl port-forward -n demofml service/minio 9001:9001
```

MLflow is then available at `http://127.0.0.1:5000`. MinIO is available at
`http://127.0.0.1:9001`; its root credentials remain only in the Kubernetes
Secret `demofml-minio-root`.

If `kubectl logs`, `exec`, or `port-forward` returns `tls: internal error`, the
API server cannot validate or establish its streaming connection to the
kubelet. Namespace workloads continue running, but a cluster administrator
must repair the node serving certificate before those commands are available.

## Locked Test Isolation

The repository intentionally does not contain a deployable locked-test Job.
Application checks consume a global marker before remote access, but Kubernetes
and a mutable PVC cannot stop an administrator from deleting that marker or
recreating a Job under another name. Before Phase 13 can be deployed, provision:

- A locked-data bucket or prefix unreachable by `demofml-services`.
- A short-lived GET-only identity restricted to the exact allowlisted objects.
- A protected grant issued after one accepted candidate is frozen.
- Append-only storage for the grant claim, terminal marker and final artifacts.
- Admission controls pinning the candidate/runtime digest and forbidding retries.
- `parallelism: 1`, `completions: 1`, `backoffLimit: 0` and `restartPolicy: Never`.

The freeze workload must receive no AWS credentials. The eventual locked
workload must not mount `demofml-development-work-v1` or reuse the broad
development secret. These are deployment blockers, not optional hardening.

## Data Safety

The `local-path` StorageClass has a `Delete` reclaim policy. StatefulSets retain
their claims when deleted or scaled, but deleting either PVC destroys the
underlying data. Never run `kubectl delete pvc` as part of routine deployment.
MinIO and PostgreSQL must be backed up before destructive maintenance.

Deleting and recreating `demofml-services` changes both database and S3
credentials and will break existing state. Credential rotation requires a
coordinated migration and is not performed by these scripts.

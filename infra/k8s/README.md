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

## Data Safety

The `local-path` StorageClass has a `Delete` reclaim policy. StatefulSets retain
their claims when deleted or scaled, but deleting either PVC destroys the
underlying data. Never run `kubectl delete pvc` as part of routine deployment.
MinIO and PostgreSQL must be backed up before destructive maintenance.

Deleting and recreating `demofml-services` changes both database and S3
credentials and will break existing state. Credential rotation requires a
coordinated migration and is not performed by these scripts.

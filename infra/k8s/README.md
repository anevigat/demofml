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

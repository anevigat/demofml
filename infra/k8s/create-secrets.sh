#!/usr/bin/env bash
set -euo pipefail

readonly NAMESPACE="demofml"
readonly ROOT_SECRET="demofml-minio-root"
readonly SERVICES_SECRET="demofml-services"

namespace_label="$(
  kubectl get namespace "${NAMESPACE}" \
    -o jsonpath='{.metadata.labels.intechsol\.com/workload}'
)"
if [[ "${namespace_label}" != "demofml" ]]; then
  printf 'Refusing to continue: namespace %s is missing the expected label.\n' \
    "${NAMESPACE}" >&2
  exit 1
fi

root_exists=false
services_exists=false
kubectl get secret "${ROOT_SECRET}" -n "${NAMESPACE}" >/dev/null 2>&1 \
  && root_exists=true
kubectl get secret "${SERVICES_SECRET}" -n "${NAMESPACE}" >/dev/null 2>&1 \
  && services_exists=true

if [[ "${root_exists}" == true && "${services_exists}" == true ]]; then
  printf 'Required secrets already exist in namespace %s.\n' "${NAMESPACE}"
  exit 0
fi

if [[ "${root_exists}" != "${services_exists}" ]]; then
  printf 'Refusing to rotate a partial secret set. Inspect namespace %s.\n' \
    "${NAMESPACE}" >&2
  exit 1
fi

minio_root_user="demofml-root"
minio_root_password="$(openssl rand -hex 32)"
aws_access_key_id="demofml"
aws_secret_access_key="$(openssl rand -hex 32)"
postgres_user="mlflow"
postgres_password="$(openssl rand -hex 32)"
postgres_db="mlflow"
backend_uri="postgresql+psycopg://${postgres_user}:${postgres_password}@postgres:5432/${postgres_db}"

kubectl create secret generic "${ROOT_SECRET}" \
  --namespace "${NAMESPACE}" \
  --from-literal=MINIO_ROOT_USER="${minio_root_user}" \
  --from-literal=MINIO_ROOT_PASSWORD="${minio_root_password}" \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic "${SERVICES_SECRET}" \
  --namespace "${NAMESPACE}" \
  --from-literal=AWS_ACCESS_KEY_ID="${aws_access_key_id}" \
  --from-literal=AWS_SECRET_ACCESS_KEY="${aws_secret_access_key}" \
  --from-literal=POSTGRES_USER="${postgres_user}" \
  --from-literal=POSTGRES_PASSWORD="${postgres_password}" \
  --from-literal=POSTGRES_DB="${postgres_db}" \
  --from-literal=MLFLOW_BACKEND_STORE_URI="${backend_uri}" \
  --dry-run=client -o yaml | kubectl apply -f -

unset minio_root_password aws_secret_access_key postgres_password backend_uri
printf 'Created service credentials in namespace %s.\n' "${NAMESPACE}"

#!/usr/bin/env bash
set -euo pipefail

readonly NAMESPACE="demofml"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/create-secrets.sh"
kubectl apply -k "${SCRIPT_DIR}/base"

kubectl rollout status statefulset/postgres -n "${NAMESPACE}" --timeout=5m
kubectl rollout status statefulset/minio -n "${NAMESPACE}" --timeout=5m
kubectl wait --for=condition=complete job/minio-bootstrap \
  -n "${NAMESPACE}" --timeout=5m
kubectl rollout status deployment/mlflow -n "${NAMESPACE}" --timeout=10m

printf 'demofml infrastructure is ready.\n'

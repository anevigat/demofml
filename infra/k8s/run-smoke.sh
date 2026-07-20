#!/usr/bin/env bash
set -euo pipefail

readonly NAMESPACE="demofml"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly JOB="demofml-infrastructure-smoke"

kubectl delete job "${JOB}" -n "${NAMESPACE}" --ignore-not-found
kubectl apply -f "${SCRIPT_DIR}/jobs/infrastructure-smoke.yaml"
kubectl wait --for=condition=complete "job/${JOB}" \
  -n "${NAMESPACE}" --timeout=5m
if ! kubectl logs "job/${JOB}" -n "${NAMESPACE}"; then
  printf 'Smoke Job completed, but kubelet logs are unavailable via the API.\n' >&2
fi

#!/usr/bin/env bash
set -euo pipefail

readonly NAMESPACE="demofml"
readonly INGRESS="minio"
readonly TLS_SECRET="demofml-minio-tls"
readonly HOST="${DEMOFML_INGRESS_HOST:?Set DEMOFML_INGRESS_HOST to the private hostname}"

if [[ ! "${HOST}" =~ ^[a-z0-9.-]+$ ]]; then
  printf 'Invalid ingress hostname.\n' >&2
  exit 1
fi

if kubectl get secret "${TLS_SECRET}" -n "${NAMESPACE}" >/dev/null 2>&1; then
  configured_host="$(
    kubectl get secret "${TLS_SECRET}" -n "${NAMESPACE}" \
      -o jsonpath='{.metadata.annotations.demofml\.io/ingress-host}'
  )"
  if [[ "${configured_host}" != "${HOST}" ]]; then
    printf 'Existing TLS secret belongs to a different hostname.\n' >&2
    exit 1
  fi
else
  temporary_directory="$(mktemp -d)"
  trap 'rm -rf "${temporary_directory}"' EXIT

  openssl req -x509 -nodes -newkey rsa:3072 -sha256 -days 825 \
    -keyout "${temporary_directory}/tls.key" \
    -out "${temporary_directory}/tls.crt" \
    -subj "/CN=${HOST}" \
    -addext "subjectAltName=DNS:${HOST}" \
    >/dev/null 2>&1

  kubectl create secret tls "${TLS_SECRET}" \
    --namespace "${NAMESPACE}" \
    --cert "${temporary_directory}/tls.crt" \
    --key "${temporary_directory}/tls.key" \
    --dry-run=client -o yaml \
    | kubectl annotate --local -f - \
        "demofml.io/ingress-host=${HOST}" -o yaml \
    | kubectl apply -f -
fi

kubectl apply -f - <<EOF
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ${INGRESS}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/name: minio
    app.kubernetes.io/part-of: demofml
    app.kubernetes.io/managed-by: demofml-script
spec:
  ingressClassName: traefik
  rules:
    - host: ${HOST}
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: minio
                port:
                  number: 9000
  tls:
    - hosts:
        - ${HOST}
      secretName: ${TLS_SECRET}
EOF

printf 'Configured private MinIO ingress.\n'

# Infrastructure

Kubernetes resources and operational instructions live in [`k8s/`](k8s/).
The deployment is namespace-scoped and must never create or modify resources
outside the existing `demofml` namespace.

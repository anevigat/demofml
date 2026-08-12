# Infrastructure

This repository intentionally contains no infrastructure-as-code, deployment
manifests, or operational commands. Execution infrastructure (cluster,
namespace, node placement, secrets, and the exact `kubectl` commands used to
run pipelines) is operator-internal and is kept out of version control by
policy — it is never committed here, in any form.

Researchers running this project operate their own execution environment by
hand and provide connection details (S3/MLflow endpoints, credentials) purely
through environment variables, as documented in the "Dataset Publication"
section of the top-level `README.md`.

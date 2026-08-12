# Campaign 2 On-Prem Custody Requirements

Status: requirements only; no custody tenant, collection, fitting, scoring, raw
access, or evaluation authorization has been provisioned

## Trust Boundary

Campaign 2 custody must remain on-prem and separately administered from the
research workflow. The custodian controls storage administration, credentials,
policy archives, signing material, trusted time, and terminal claim storage.
Researchers receive no raw-data or storage-administration identity.

The current shared MinIO service in the operator's on-prem environment is
development infrastructure, not a Campaign 2 custody boundary. It has one replica, one local
PVC, shared static application credentials with broad `s3:*` access, and no
configured Object Lock, versioning, retention, independent administration, or
custodial signature. It must not hold Campaign 2 qualification or holdout data.

Before any real collection, an on-prem custodian must provide:

- A separately administered MinIO tenant or equivalent immutable object store.
- A dedicated bucket created with Object Lock enabled and versioning `Enabled`.
- `COMPLIANCE` retention for every exact object version through at least
  `2028-09-01T00:00:00Z`.
- Separate least-privilege collector, scorer, metadata-verifier, and
  append-only prediction-sink identities.
- Explicit denial of raw reads to the metadata verifier and of deletion,
  overwrite, retention shortening, policy mutation, and administration to all
  workload identities.
- Archived canonical identity and storage-policy documents with SHA-256
  digests.
- An independently controlled asymmetric signing key, trusted UTC clock, audit
  records, backups, and terminal claim storage.

Application hashes, S3 request signatures, Kubernetes ServiceAccounts, and PVC
retention do not substitute for independent custody or immutable retention.

## Metadata Verification

A future preflight may inspect only storage metadata needed to verify:

- Bucket versioning and Object Lock status.
- Exact object-version identity, size, latest status, and `LastModified`.
- Absence of a latest delete marker.
- Exact-version `COMPLIANCE` retention through the frozen deadline.
- Custodian-signed bindings among the engineering bundle, collection terminal,
  prequalification envelope, object-version count, storage identity, policy
  digests, signing key, and trusted time.

The metadata-verifier identity must not be able to read object bytes or user
metadata, write, delete, change retention, sign attestations, administer
storage, collect data, score, or evaluate. No real on-prem preflight is
implemented or authorized yet.

## Storage Claims

- Every claimed object has a non-null immutable version ID.
- Every claimed version remains latest and has no latest delete marker.
- Keys are unique across all eight symbol chains and remain under the signed
  Campaign 2 prefix.
- Object size and exact UTC `LastModified` reconcile with signed chain claims.
- SHA-256, row counts, ingest ranges, receipt ranges, and collector identity are
  signed custodian claims because metadata-only verification cannot inspect raw
  object contents.

Passing a future custody preflight may complete external qualification only. It
must never authorize collection, fitting, scoring, evaluation, raw reads, or
model execution.

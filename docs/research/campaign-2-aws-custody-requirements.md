# Campaign 2 AWS Custody Preflight Requirements

Status: provider contract only; no AWS account, bucket, role, key, collection, or
scoring authorization has been provisioned

Contract ID: `campaign-2-aws-custody-preflight-v1`

## Trust Boundary

The preflight role is controlled outside the research workflow and uses
short-lived credentials. It is a metadata verifier, not the collector or
scoring identity. The role, bucket, KMS key, permissions boundary, bucket
policy, and key policy are supplied and externally approved before any real
preflight runs.

The preflight implementation may call only:

- STS `GetCallerIdentity`.
- S3 `GetBucketVersioning` and `GetBucketObjectLockConfiguration`.
- S3 `ListObjectVersions` restricted to the frozen Campaign 2 prefix.
- S3 `GetObjectRetention` for exact immutable version IDs.
- KMS `DescribeKey` and `Verify` for one full key ARN.

It must not receive `s3:GetObject`, `s3:GetObjectVersion`,
`s3:GetObjectAttributes`, `s3:SelectObjectContent`, write, delete, retention
mutation, `kms:Sign`, `kms:Decrypt`, data-key, grant, IAM, or role-assumption
capabilities. `HeadObject` is prohibited because AWS authorizes it with
`s3:GetObject`, which would violate the no-raw-read boundary.

S3, KMS, and STS clients must be created from one AWS SDK session and therefore
one short-lived credential source. The attestation validity window is checked
against the runtime UTC clock; callers cannot supply an alternate observation
time. Runtime clock synchronization is part of the external execution
attestation.

Application code cannot prove that no alternate AWS identity exists. Explicit
deny policies, account separation, policy immutability, and control of the
signing role remain external custody requirements.

## Frozen Provider Config

The custodian supplies a real TOML file with this exact shape. Placeholder
values below are documentation and are not a deployable config.

```toml
format_version = 1
id = "campaign-2-aws-custody-preflight-v1"
campaign_id = "prospective-cross-pair-factor-v1"
partition = "aws"
account_id = "123456789012"
region = "us-east-1"
bucket = "externally-controlled-campaign2"
prefix = "campaign-2/prospective-cross-pair-factor-v1/"
expected_preflight_role_arn = "arn:aws:iam::123456789012:role/campaign2-preflight"
kms_signing_key_arn = "arn:aws:kms:us-east-1:123456789012:key/00000000-0000-0000-0000-000000000000"
permissions_boundary_sha256 = "<64 lowercase hex>"
bucket_policy_sha256 = "<64 lowercase hex>"
kms_key_policy_sha256 = "<64 lowercase hex>"
signing_algorithm = "ECDSA_SHA_256"
object_lock_mode = "COMPLIANCE"
minimum_retain_until = "2028-09-01T00:00:00Z"
maximum_attestation_age_seconds = 86400
require_latest_version = true

[authorization]
preflight = true
raw_read = false
write = false
delete = false
sign = false
decrypt = false
collection = false
scoring = false
evaluation = false
```

All three policy digests are computed from externally archived canonical policy
documents. A KMS signature binds those digests, the AWS account and region,
bucket and prefix, engineering bundle ID, collection terminal ID,
prequalification ID, object-version count, preflight role, retention deadline,
and signing time. The signing key is a customer-managed `ECC_NIST_P256` key with
`SIGN_VERIFY` usage and `ECDSA_SHA_256` enabled. Aliases are not accepted.

The signed message is precisely the attestation object excluding
`attestation_id` and `signature_base64`, encoded as ASCII JSON with keys sorted,
compact `,`/`:` separators, escaped non-ASCII text, non-finite numbers rejected,
and one trailing LF byte. The custodian computes SHA-256 over those bytes and
calls KMS `Sign` with `MessageType="DIGEST"` and
`SigningAlgorithm="ECDSA_SHA_256"`. `attestation_id` is `sha256-` followed by
the SHA-256 of the same canonical bytes. The preflight sends that 32-byte digest
to KMS `Verify` with the same message type and algorithm.

## Storage Requirements

- Bucket versioning is `Enabled`, never `Suspended`.
- Bucket Object Lock is enabled at bucket creation.
- Every claimed object uses a non-null immutable `VersionId`.
- Every exact version has `COMPLIANCE` retention through at least
  `2028-09-01T00:00:00Z`, and retention must still be active when preflight runs.
- Every claimed version remains latest and has no latest delete marker.
- Keys are unique across all eight symbol chains and remain under the signed
  prefix.
- Object size and exact UTC `LastModified` reconcile with signed chain claims.
- SHA-256, row counts, ingest ranges, and receipt ranges are signed custodian
  claims because the metadata-only role cannot inspect object bytes or user
  metadata.

Passing preflight can complete external custody qualification, but it never
authorizes collection, scoring, evaluation, raw reads, or model execution.

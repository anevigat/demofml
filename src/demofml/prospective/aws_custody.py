"""Read-only AWS Object Lock and KMS preflight for external Campaign 2 custody."""

import base64
import binascii
import hashlib
import re
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

from demofml.features.cross_pair import PAIRS
from demofml.prospective.custody import validate_collection_terminal
from demofml.prospective.opportunities import CAMPAIGN_ID
from demofml.prospective.qualification import validate_qualification_envelope
from demofml.prospective.records import (
    CONTENT_ID_PATTERN,
    SHA256_PATTERN,
    canonical_json,
    content_id,
    read_strict_json,
)

AWS_CUSTODY_CONFIG_ID = "campaign-2-aws-custody-preflight-v1"
AWS_CUSTODY_ATTESTATION_SET_ID = "campaign-2-aws-custody-attestation-v1"
AWS_CUSTODY_PREFLIGHT_SET_ID = "campaign-2-aws-custody-preflight-result-v1"
AWS_SIGNING_ALGORITHM = "ECDSA_SHA_256"
MINIMUM_RETENTION_UNTIL = datetime(2028, 9, 1, tzinfo=UTC)
_ACCOUNT_PATTERN = re.compile(r"[0-9]{12}")
_REGION_PATTERN = re.compile(r"[a-z]{2}(?:-gov)?-[a-z]+-[0-9]")
_BUCKET_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")
_ROLE_NAME_PATTERN = re.compile(r"[A-Za-z0-9+=,.@_-]{1,64}")
_KMS_KEY_ID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


class S3CustodyMetadataClient(Protocol):
    """Narrow API surface that cannot read or mutate object bytes."""

    def get_bucket_versioning(self, **kwargs: Any) -> dict[str, Any]: ...

    def get_object_lock_configuration(self, **kwargs: Any) -> dict[str, Any]: ...

    def list_object_versions(self, **kwargs: Any) -> dict[str, Any]: ...

    def get_object_retention(self, **kwargs: Any) -> dict[str, Any]: ...


class KmsSignatureVerifier(Protocol):
    """KMS verification surface without sign, decrypt, or data-key methods."""

    def describe_key(self, **kwargs: Any) -> dict[str, Any]: ...

    def verify(self, **kwargs: Any) -> dict[str, Any]: ...


class StsIdentityClient(Protocol):
    """Caller-identity surface used to bind the metadata-only role."""

    def get_caller_identity(self, **kwargs: Any) -> dict[str, Any]: ...


class AwsCustodySession(Protocol):
    """One credential source for S3, KMS, and STS clients."""

    def client(
        self, service_name: str, *, region_name: str
    ) -> object: ...


@dataclass(frozen=True)
class AwsCustodyConfig:
    """Externally supplied AWS account and immutable-custody contract."""

    path: Path
    partition: str
    account_id: str
    region: str
    bucket: str
    prefix: str
    expected_preflight_role_arn: str
    kms_signing_key_arn: str
    permissions_boundary_sha256: str
    bucket_policy_sha256: str
    kms_key_policy_sha256: str
    minimum_retain_until: datetime
    maximum_attestation_age_seconds: int

    @property
    def role_name(self) -> str:
        return self.expected_preflight_role_arn.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class AwsCustodyPreflightReport:
    """Verified metadata evidence that never grants collection or scoring."""

    attestation_id: str
    engineering_bundle_id: str
    collection_terminal_id: str
    prequalification_id: str
    account_id: str
    caller_arn: str
    bucket: str
    object_versions_checked: int
    minimum_observed_retain_until: datetime
    custody_preflight_complete: bool
    qualification_complete: bool
    collection_authorized: bool = False
    scoring_authorized: bool = False
    evaluation_authorized: bool = False

    def as_record(self) -> dict[str, object]:
        """Return evidence without credentials, object bytes, or outcomes."""
        core: dict[str, object] = {
            "format_version": 1,
            "preflight_set": AWS_CUSTODY_PREFLIGHT_SET_ID,
            "campaign_id": CAMPAIGN_ID,
            "attestation_id": self.attestation_id,
            "engineering_bundle_id": self.engineering_bundle_id,
            "collection_terminal_id": self.collection_terminal_id,
            "prequalification_id": self.prequalification_id,
            "account_id": self.account_id,
            "caller_arn": self.caller_arn,
            "bucket": self.bucket,
            "object_versions_checked": self.object_versions_checked,
            "minimum_observed_retain_until": _format_utc(
                self.minimum_observed_retain_until
            ),
            "custody_preflight_complete": self.custody_preflight_complete,
            "qualification_complete": self.qualification_complete,
            "collection_authorized": self.collection_authorized,
            "scoring_authorized": self.scoring_authorized,
            "evaluation_authorized": self.evaluation_authorized,
        }
        return {**core, "preflight_id": content_id(core)}


def _format_utc(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("AWS custody timestamps must be timezone-aware UTC")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise ValueError(f"{name} is invalid") from error
    canonical = _format_utc(parsed)
    if canonical != value:
        raise ValueError(f"{name} is not canonically encoded")
    return parsed.astimezone(UTC)


def _exact_int(value: object, name: str, *, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _safe_prefix(value: str) -> None:
    path = PurePosixPath(value)
    if (
        not value
        or value.startswith("/")
        or not value.endswith("/")
        or "//" in value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("AWS custody prefix must be a safe relative prefix")


def load_aws_custody_config(path: Path) -> AwsCustodyConfig:
    """Load an external AWS contract without accepting endpoints or credentials."""
    requested = path.expanduser().absolute()
    if requested.is_symlink() or not requested.is_file():
        raise ValueError("AWS custody config must be a regular non-symlink file")
    try:
        with requested.open("rb") as source:
            values = tomllib.load(source)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ValueError("AWS custody config is invalid") from error
    expected = {
        "format_version",
        "id",
        "campaign_id",
        "partition",
        "account_id",
        "region",
        "bucket",
        "prefix",
        "expected_preflight_role_arn",
        "kms_signing_key_arn",
        "permissions_boundary_sha256",
        "bucket_policy_sha256",
        "kms_key_policy_sha256",
        "signing_algorithm",
        "object_lock_mode",
        "minimum_retain_until",
        "maximum_attestation_age_seconds",
        "require_latest_version",
        "authorization",
    }
    if set(values) != expected:
        raise ValueError("AWS custody config fields are incompatible")
    string_fields = (
        "id",
        "campaign_id",
        "partition",
        "account_id",
        "region",
        "bucket",
        "prefix",
        "expected_preflight_role_arn",
        "kms_signing_key_arn",
        "permissions_boundary_sha256",
        "bucket_policy_sha256",
        "kms_key_policy_sha256",
        "signing_algorithm",
        "object_lock_mode",
        "minimum_retain_until",
    )
    if any(not isinstance(values[field], str) for field in string_fields):
        raise ValueError("AWS custody string fields are incompatible")
    partition = str(values["partition"])
    account = str(values["account_id"])
    region = str(values["region"])
    bucket = str(values["bucket"])
    prefix = str(values["prefix"])
    role_arn = str(values["expected_preflight_role_arn"])
    key_arn = str(values["kms_signing_key_arn"])
    policy_digests = (
        values["permissions_boundary_sha256"],
        values["bucket_policy_sha256"],
        values["kms_key_policy_sha256"],
    )
    _safe_prefix(prefix)
    role_name = role_arn.rsplit("/", 1)[-1]
    expected_role = f"arn:{partition}:iam::{account}:role/{role_name}"
    key_prefix = f"arn:{partition}:kms:{region}:{account}:key/"
    authorization = values["authorization"]
    if (
        type(values["format_version"]) is not int
        or values["format_version"] != 1
        or values["id"] != AWS_CUSTODY_CONFIG_ID
        or values["campaign_id"] != CAMPAIGN_ID
        or partition not in {"aws", "aws-us-gov"}
        or _ACCOUNT_PATTERN.fullmatch(account) is None
        or _REGION_PATTERN.fullmatch(region) is None
        or _BUCKET_PATTERN.fullmatch(bucket) is None
        or role_arn != expected_role
        or _ROLE_NAME_PATTERN.fullmatch(role_name) is None
        or not key_arn.startswith(key_prefix)
        or _KMS_KEY_ID_PATTERN.fullmatch(key_arn.removeprefix(key_prefix)) is None
        or any(
            not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
            for value in policy_digests
        )
        or values["signing_algorithm"] != AWS_SIGNING_ALGORITHM
        or values["object_lock_mode"] != "COMPLIANCE"
        or values["require_latest_version"] is not True
        or not isinstance(authorization, dict)
        or authorization
        != {
            "preflight": True,
            "raw_read": False,
            "write": False,
            "delete": False,
            "sign": False,
            "decrypt": False,
            "collection": False,
            "scoring": False,
            "evaluation": False,
        }
        or any(type(value) is not bool for value in authorization.values())
    ):
        raise ValueError("AWS custody config values are incompatible")
    retain_until = _parse_utc(values["minimum_retain_until"], "retain until")
    if retain_until != MINIMUM_RETENTION_UNTIL:
        raise ValueError("AWS custody retention deadline differs from the protocol")
    maximum_age = _exact_int(
        values["maximum_attestation_age_seconds"],
        "maximum_attestation_age_seconds",
        minimum=1,
    )
    if maximum_age != 86_400:
        raise ValueError("AWS custody attestation age must be exactly one day")
    return AwsCustodyConfig(
        requested.resolve(),
        partition,
        account,
        region,
        bucket,
        prefix,
        role_arn,
        key_arn,
        str(values["permissions_boundary_sha256"]),
        str(values["bucket_policy_sha256"]),
        str(values["kms_key_policy_sha256"]),
        retain_until,
        maximum_age,
    )


def load_aws_custody_attestation(path: Path) -> dict[str, object]:
    """Load and structurally validate a signed external custody envelope."""
    attestation = read_strict_json(path, "AWS custody attestation")
    _validate_attestation_structure(attestation)
    return attestation


def _attestation_core(attestation: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in attestation.items()
        if key not in {"attestation_id", "signature_base64"}
    }


def _validate_attestation_structure(attestation: Mapping[str, object]) -> None:
    expected = {
        "format_version",
        "attestation_set",
        "campaign_id",
        "partition",
        "account_id",
        "region",
        "bucket",
        "prefix",
        "engineering_bundle_id",
        "collection_terminal_id",
        "prequalification_id",
        "object_version_count",
        "minimum_retain_until",
        "preflight_role_arn",
        "kms_signing_key_arn",
        "permissions_boundary_sha256",
        "bucket_policy_sha256",
        "kms_key_policy_sha256",
        "signing_algorithm",
        "signed_at",
        "expires_at",
        "attestation_id",
        "signature_base64",
    }
    if set(attestation) != expected:
        raise ValueError("AWS custody attestation fields are incompatible")
    string_fields = expected.difference({"format_version", "object_version_count"})
    if any(not isinstance(attestation[field], str) for field in string_fields):
        raise ValueError("AWS custody attestation string fields are incompatible")
    if (
        type(attestation["format_version"]) is not int
        or attestation["format_version"] != 1
        or attestation["attestation_set"] != AWS_CUSTODY_ATTESTATION_SET_ID
        or attestation["campaign_id"] != CAMPAIGN_ID
        or attestation["signing_algorithm"] != AWS_SIGNING_ALGORITHM
    ):
        raise ValueError("AWS custody attestation identity is incompatible")
    partition = str(attestation["partition"])
    account = str(attestation["account_id"])
    region = str(attestation["region"])
    bucket = str(attestation["bucket"])
    prefix = str(attestation["prefix"])
    role_arn = str(attestation["preflight_role_arn"])
    key_arn = str(attestation["kms_signing_key_arn"])
    role_name = role_arn.rsplit("/", 1)[-1]
    key_prefix = f"arn:{partition}:kms:{region}:{account}:key/"
    _safe_prefix(prefix)
    if (
        partition not in {"aws", "aws-us-gov"}
        or _ACCOUNT_PATTERN.fullmatch(account) is None
        or _REGION_PATTERN.fullmatch(region) is None
        or _BUCKET_PATTERN.fullmatch(bucket) is None
        or role_arn != f"arn:{partition}:iam::{account}:role/{role_name}"
        or _ROLE_NAME_PATTERN.fullmatch(role_name) is None
        or not key_arn.startswith(key_prefix)
        or _KMS_KEY_ID_PATTERN.fullmatch(key_arn.removeprefix(key_prefix)) is None
        or any(
            SHA256_PATTERN.fullmatch(str(attestation[field])) is None
            for field in (
                "permissions_boundary_sha256",
                "bucket_policy_sha256",
                "kms_key_policy_sha256",
            )
        )
    ):
        raise ValueError("AWS custody attestation provider fields are incompatible")
    for field in (
        "engineering_bundle_id",
        "collection_terminal_id",
        "prequalification_id",
        "attestation_id",
    ):
        value = attestation[field]
        if not isinstance(value, str) or CONTENT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError(f"AWS custody attestation {field} is invalid")
    _exact_int(attestation["object_version_count"], "object_version_count", minimum=1)
    signed_at = _parse_utc(attestation["signed_at"], "signed_at")
    expires_at = _parse_utc(attestation["expires_at"], "expires_at")
    retain_until = _parse_utc(
        attestation["minimum_retain_until"], "minimum_retain_until"
    )
    if retain_until != MINIMUM_RETENTION_UNTIL:
        raise ValueError("AWS custody attestation retention differs from protocol")
    if expires_at <= signed_at:
        raise ValueError("AWS custody attestation expiry is incompatible")
    signature = attestation["signature_base64"]
    if not isinstance(signature, str):
        raise ValueError("AWS custody signature must be base64 text")
    try:
        decoded = base64.b64decode(signature, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("AWS custody signature is invalid base64") from error
    if not decoded or len(decoded) > 8192:
        raise ValueError("AWS custody signature size is incompatible")
    if attestation["attestation_id"] != content_id(_attestation_core(attestation)):
        raise ValueError("AWS custody attestation content ID mismatch")


def run_aws_custody_preflight(
    *,
    config: AwsCustodyConfig,
    attestation: Mapping[str, object],
    collection_terminal: Mapping[str, object],
    collection_chains: Mapping[str, Sequence[Mapping[str, object]]],
    prequalification: Mapping[str, object],
    session: AwsCustodySession,
) -> AwsCustodyPreflightReport:
    """Verify signed AWS metadata without any raw-read or mutation API."""
    _validate_attestation_structure(attestation)
    validate_collection_terminal(collection_terminal, collection_chains)
    validate_qualification_envelope(prequalification)
    if set(collection_chains) != set(PAIRS):
        raise ValueError("AWS custody preflight requires all eight symbol chains")
    observed = _utc_now()
    signed_at = _parse_utc(attestation["signed_at"], "signed_at")
    expires_at = _parse_utc(attestation["expires_at"], "expires_at")
    terminal_recorded = _parse_utc(collection_terminal["recorded_at"], "recorded_at")
    if not (
        terminal_recorded <= signed_at <= observed <= expires_at
        and expires_at - signed_at
        <= timedelta(seconds=config.maximum_attestation_age_seconds)
    ):
        raise ValueError("AWS custody attestation time window is incompatible")
    _validate_attestation_bindings(
        config, attestation, collection_terminal, prequalification
    )
    objects = _collection_objects(collection_chains)
    if attestation["object_version_count"] != len(objects):
        raise ValueError("AWS custody attestation object count does not reconcile")

    s3 = cast(
        S3CustodyMetadataClient,
        session.client("s3", region_name=config.region),
    )
    kms = cast(
        KmsSignatureVerifier,
        session.client("kms", region_name=config.region),
    )
    sts = cast(
        StsIdentityClient,
        session.client("sts", region_name=config.region),
    )

    key_metadata = kms.describe_key(KeyId=config.kms_signing_key_arn)
    metadata_record = key_metadata.get("KeyMetadata")
    if not isinstance(metadata_record, dict):
        raise ValueError("KMS DescribeKey lacks KeyMetadata")
    if (
        metadata_record.get("Arn") != config.kms_signing_key_arn
        or metadata_record.get("Enabled") is not True
        or metadata_record.get("KeyState") != "Enabled"
        or metadata_record.get("KeyUsage") != "SIGN_VERIFY"
        or metadata_record.get("KeySpec") != "ECC_NIST_P256"
        or AWS_SIGNING_ALGORITHM
        not in metadata_record.get("SigningAlgorithms", [])
    ):
        raise ValueError("KMS signing key is incompatible")
    signature = base64.b64decode(str(attestation["signature_base64"]), validate=True)
    message_digest = hashlib.sha256(
        canonical_json(_attestation_core(attestation))
    ).digest()
    verification = kms.verify(
        KeyId=config.kms_signing_key_arn,
        Message=message_digest,
        MessageType="DIGEST",
        Signature=signature,
        SigningAlgorithm=AWS_SIGNING_ALGORITHM,
    )
    if verification.get("SignatureValid") is not True:
        raise ValueError("KMS rejected the custody attestation signature")
    if (
        verification.get("KeyId") != config.kms_signing_key_arn
        or verification.get("SigningAlgorithm") != AWS_SIGNING_ALGORITHM
    ):
        raise ValueError("KMS verification response identity is incompatible")

    identity = sts.get_caller_identity()
    caller_account = identity.get("Account")
    caller_arn = identity.get("Arn")
    expected_assumed_prefix = (
        f"arn:{config.partition}:sts::{config.account_id}:assumed-role/"
        f"{config.role_name}/"
    )
    if (
        caller_account != config.account_id
        or not isinstance(caller_arn, str)
        or not caller_arn.startswith(expected_assumed_prefix)
    ):
        raise ValueError("AWS caller identity is not the frozen preflight role")

    versioning = s3.get_bucket_versioning(
        Bucket=config.bucket,
        ExpectedBucketOwner=config.account_id,
    )
    if versioning.get("Status") != "Enabled":
        raise ValueError("AWS custody bucket versioning is not enabled")
    object_lock = s3.get_object_lock_configuration(
        Bucket=config.bucket,
        ExpectedBucketOwner=config.account_id,
    )
    lock_configuration = object_lock.get("ObjectLockConfiguration")
    if (
        not isinstance(lock_configuration, dict)
        or lock_configuration.get("ObjectLockEnabled") != "Enabled"
    ):
        raise ValueError("AWS custody bucket Object Lock is not enabled")

    minimum_observed: datetime | None = None
    for object_record in objects:
        key = str(object_record["key"])
        version_id = str(object_record["version_id"])
        if not key.startswith(config.prefix) or version_id == "null":
            raise ValueError("custody object is outside prefix or has a null version")
        version = _find_exact_object_version(
            s3,
            config,
            key,
            version_id,
        )
        if version.get("Size") != object_record["size_bytes"]:
            raise ValueError("custody object version size differs from the claim")
        if version.get("IsLatest") is not True:
            raise ValueError("custody object claim is not the latest version")
        last_modified = _require_utc_datetime(
            version.get("LastModified"), "object LastModified"
        )
        claimed_last_modified = _parse_utc(
            object_record.get("last_modified"), "claimed LastModified"
        )
        if last_modified != claimed_last_modified or last_modified > signed_at:
            raise ValueError("custody object LastModified differs from signed claim")
        retention = s3.get_object_retention(
            Bucket=config.bucket,
            Key=key,
            VersionId=version_id,
            ExpectedBucketOwner=config.account_id,
        ).get("Retention")
        if not isinstance(retention, dict) or retention.get("Mode") != "COMPLIANCE":
            raise ValueError("custody object lacks COMPLIANCE retention")
        retain_until = _require_utc_datetime(
            retention.get("RetainUntilDate"), "RetainUntilDate"
        )
        if retain_until < config.minimum_retain_until or retain_until <= observed:
            raise ValueError("custody object retention ends before the frozen deadline")
        minimum_observed = (
            retain_until
            if minimum_observed is None
            else min(minimum_observed, retain_until)
        )
    if minimum_observed is None:
        raise AssertionError("validated custody terminal must contain objects")
    engineering_checks_passed = prequalification.get("engineering_checks_passed")
    return AwsCustodyPreflightReport(
        str(attestation["attestation_id"]),
        str(attestation["engineering_bundle_id"]),
        str(attestation["collection_terminal_id"]),
        str(attestation["prequalification_id"]),
        config.account_id,
        caller_arn,
        config.bucket,
        len(objects),
        minimum_observed,
        True,
        engineering_checks_passed is True,
    )


def _validate_attestation_bindings(
    config: AwsCustodyConfig,
    attestation: Mapping[str, object],
    terminal: Mapping[str, object],
    prequalification: Mapping[str, object],
) -> None:
    if (
        attestation["partition"] != config.partition
        or attestation["account_id"] != config.account_id
        or attestation["region"] != config.region
        or attestation["bucket"] != config.bucket
        or attestation["prefix"] != config.prefix
        or attestation["preflight_role_arn"] != config.expected_preflight_role_arn
        or attestation["kms_signing_key_arn"] != config.kms_signing_key_arn
        or attestation["permissions_boundary_sha256"]
        != config.permissions_boundary_sha256
        or attestation["bucket_policy_sha256"] != config.bucket_policy_sha256
        or attestation["kms_key_policy_sha256"] != config.kms_key_policy_sha256
        or attestation["minimum_retain_until"]
        != _format_utc(config.minimum_retain_until)
        or attestation["engineering_bundle_id"] != prequalification["bundle_id"]
        or attestation["collection_terminal_id"] != terminal["terminal_id"]
        or attestation["collection_terminal_id"]
        != prequalification["collection_terminal_id"]
        or attestation["prequalification_id"]
        != prequalification["qualification_id"]
    ):
        raise ValueError("AWS custody attestation does not bind the frozen evidence")


def _collection_objects(
    chains: Mapping[str, Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for symbol in PAIRS:
        for segment in chains[symbol]:
            object_record = segment.get("object")
            if not isinstance(object_record, dict):
                raise ValueError("collection object claim must be a mapping")
            objects.append(object_record)
    return objects


def _find_exact_object_version(
    s3: S3CustodyMetadataClient,
    config: AwsCustodyConfig,
    key: str,
    version_id: str,
) -> dict[str, object]:
    matches: list[dict[str, object]] = []
    latest_delete_marker = False
    key_marker: str | None = None
    version_marker: str | None = None
    seen_markers: set[tuple[str | None, str | None]] = set()
    while True:
        marker = (key_marker, version_marker)
        if marker in seen_markers:
            raise ValueError("S3 object-version pagination repeated a marker")
        seen_markers.add(marker)
        arguments: dict[str, object] = {
            "Bucket": config.bucket,
            "Prefix": key,
            "ExpectedBucketOwner": config.account_id,
        }
        if key_marker is not None:
            arguments["KeyMarker"] = key_marker
        if version_marker is not None:
            arguments["VersionIdMarker"] = version_marker
        response = s3.list_object_versions(**arguments)
        versions = response.get("Versions", [])
        delete_markers = response.get("DeleteMarkers", [])
        if not isinstance(versions, list) or not isinstance(delete_markers, list):
            raise ValueError("S3 object-version listing is incompatible")
        for version in versions:
            if not isinstance(version, dict):
                raise ValueError("S3 object version must be a mapping")
            if version.get("Key") == key and version.get("VersionId") == version_id:
                matches.append(version)
        latest_delete_marker = latest_delete_marker or any(
            isinstance(item, dict)
            and item.get("Key") == key
            and item.get("IsLatest") is True
            for item in delete_markers
        )
        truncated = response.get("IsTruncated", False)
        if truncated is not True:
            if truncated is not False:
                raise ValueError("S3 IsTruncated must be boolean")
            break
        next_key = response.get("NextKeyMarker")
        next_version = response.get("NextVersionIdMarker")
        if not isinstance(next_key, str) or not isinstance(next_version, str):
            raise ValueError("S3 truncated listing lacks continuation markers")
        key_marker = next_key
        version_marker = next_version
    if latest_delete_marker:
        raise ValueError("custody object has a latest delete marker")
    if len(matches) != 1:
        raise ValueError("custody object version is missing or duplicated")
    return matches[0]


def _require_utc_datetime(value: object, name: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be timezone-aware UTC")
    return value.astimezone(UTC)


def _utc_now() -> datetime:
    """Read the runtime clock fixed by the externally attested execution image."""
    return datetime.now(UTC)

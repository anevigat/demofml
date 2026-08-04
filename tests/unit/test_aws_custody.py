import base64
import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from demofml.features.cross_pair import PAIRS
from demofml.prospective import aws_custody
from demofml.prospective.aws_custody import (
    AWS_CUSTODY_ATTESTATION_SET_ID,
    AWS_SIGNING_ALGORITHM,
    load_aws_custody_config,
    run_aws_custody_preflight,
)
from demofml.prospective.opportunities import CAMPAIGN_ID
from demofml.prospective.records import canonical_json, content_id

ACCOUNT = "123456789012"
REGION = "us-east-1"
BUCKET = "campaign2-external-custody"
PREFIX = "campaign-2/prospective-cross-pair-factor-v1/"
ROLE_ARN = f"arn:aws:iam::{ACCOUNT}:role/campaign2-preflight"
KEY_ARN = (
    f"arn:aws:kms:{REGION}:{ACCOUNT}:key/"
    "00000000-0000-0000-0000-000000000000"
)
BUNDLE_ID = "sha256-" + "1" * 64
TERMINAL_ID = "sha256-" + "2" * 64
QUALIFICATION_ID = "sha256-" + "3" * 64
POLICY_DIGESTS = ("4" * 64, "5" * 64, "6" * 64)
SIGNED_AT = datetime(2026, 9, 1, 1, tzinfo=UTC)
EXPIRES_AT = SIGNED_AT + timedelta(days=1)
RETAIN_UNTIL = datetime(2028, 9, 1, tzinfo=UTC)


def _write_config(path: Path, *, raw_read: bool = False) -> Path:
    path.write_text(
        f'''format_version = 1
id = "campaign-2-aws-custody-preflight-v1"
campaign_id = "{CAMPAIGN_ID}"
partition = "aws"
account_id = "{ACCOUNT}"
region = "{REGION}"
bucket = "{BUCKET}"
prefix = "{PREFIX}"
expected_preflight_role_arn = "{ROLE_ARN}"
kms_signing_key_arn = "{KEY_ARN}"
permissions_boundary_sha256 = "{POLICY_DIGESTS[0]}"
bucket_policy_sha256 = "{POLICY_DIGESTS[1]}"
kms_key_policy_sha256 = "{POLICY_DIGESTS[2]}"
signing_algorithm = "{AWS_SIGNING_ALGORITHM}"
object_lock_mode = "COMPLIANCE"
minimum_retain_until = "2028-09-01T00:00:00Z"
maximum_attestation_age_seconds = 86400
require_latest_version = true

[authorization]
preflight = true
raw_read = {str(raw_read).lower()}
write = false
delete = false
sign = false
decrypt = false
collection = false
scoring = false
evaluation = false
''',
        encoding="utf-8",
    )
    return path


def _evidence() -> tuple[
    dict[str, object],
    dict[str, list[dict[str, object]]],
    dict[str, object],
]:
    chains: dict[str, list[dict[str, object]]] = {
        symbol: [
            {
                "object": {
                    "key": f"{PREFIX}{symbol}/0000.parquet",
                    "version_id": f"version-{symbol}",
                    "size_bytes": 1024,
                    "last_modified": "2026-09-01T00:59:00Z",
                }
            }
        ]
        for symbol in PAIRS
    }
    terminal: dict[str, object] = {
        "terminal_id": TERMINAL_ID,
        "recorded_at": "2026-09-01T00:00:00Z",
    }
    prequalification = {
        "bundle_id": BUNDLE_ID,
        "collection_terminal_id": TERMINAL_ID,
        "qualification_id": QUALIFICATION_ID,
        "engineering_checks_passed": True,
    }
    return terminal, chains, prequalification


def _attestation() -> dict[str, object]:
    core: dict[str, object] = {
        "format_version": 1,
        "attestation_set": AWS_CUSTODY_ATTESTATION_SET_ID,
        "campaign_id": CAMPAIGN_ID,
        "partition": "aws",
        "account_id": ACCOUNT,
        "region": REGION,
        "bucket": BUCKET,
        "prefix": PREFIX,
        "engineering_bundle_id": BUNDLE_ID,
        "collection_terminal_id": TERMINAL_ID,
        "prequalification_id": QUALIFICATION_ID,
        "object_version_count": len(PAIRS),
        "minimum_retain_until": "2028-09-01T00:00:00Z",
        "preflight_role_arn": ROLE_ARN,
        "kms_signing_key_arn": KEY_ARN,
        "permissions_boundary_sha256": POLICY_DIGESTS[0],
        "bucket_policy_sha256": POLICY_DIGESTS[1],
        "kms_key_policy_sha256": POLICY_DIGESTS[2],
        "signing_algorithm": AWS_SIGNING_ALGORITHM,
        "signed_at": "2026-09-01T01:00:00Z",
        "expires_at": "2026-09-02T01:00:00Z",
    }
    return {
        **core,
        "attestation_id": content_id(core),
        "signature_base64": base64.b64encode(b"signature").decode("ascii"),
    }


class _Kms:
    def __init__(self, signature_valid: bool = True) -> None:
        self.signature_valid = signature_valid
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def describe_key(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("describe_key", kwargs))
        return {
            "KeyMetadata": {
                "Arn": KEY_ARN,
                "Enabled": True,
                "KeyState": "Enabled",
                "KeyUsage": "SIGN_VERIFY",
                "KeySpec": "ECC_NIST_P256",
                "SigningAlgorithms": [AWS_SIGNING_ALGORITHM],
            }
        }

    def verify(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("verify", kwargs))
        return {
            "SignatureValid": self.signature_valid,
            "KeyId": KEY_ARN,
            "SigningAlgorithm": AWS_SIGNING_ALGORITHM,
        }


class _Sts:
    def __init__(self) -> None:
        self.called = False

    def get_caller_identity(self, **kwargs: Any) -> dict[str, Any]:
        self.called = True
        assert kwargs == {}
        return {
            "Account": ACCOUNT,
            "Arn": f"arn:aws:sts::{ACCOUNT}:assumed-role/campaign2-preflight/test",
        }


class _S3:
    def __init__(self, retain_until: datetime = RETAIN_UNTIL) -> None:
        self.retain_until = retain_until
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def get_bucket_versioning(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_bucket_versioning", kwargs))
        return {"Status": "Enabled"}

    def get_object_lock_configuration(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_object_lock_configuration", kwargs))
        return {"ObjectLockConfiguration": {"ObjectLockEnabled": "Enabled"}}

    def list_object_versions(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_object_versions", kwargs))
        key = str(kwargs["Prefix"])
        symbol = key.split("/")[-2]
        return {
            "Versions": [
                {
                    "Key": key,
                    "VersionId": f"version-{symbol}",
                    "Size": 1024,
                    "IsLatest": True,
                    "LastModified": SIGNED_AT - timedelta(minutes=1),
                }
            ],
            "DeleteMarkers": [],
            "IsTruncated": False,
        }

    def get_object_retention(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("get_object_retention", kwargs))
        return {
            "Retention": {
                "Mode": "COMPLIANCE",
                "RetainUntilDate": self.retain_until,
            }
        }


class _PagedS3(_S3):
    def list_object_versions(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("list_object_versions", kwargs))
        key = str(kwargs["Prefix"])
        if "KeyMarker" not in kwargs:
            return {
                "Versions": [
                    {
                        "Key": key + ".sibling",
                        "VersionId": "not-the-claim",
                    }
                ],
                "DeleteMarkers": [],
                "IsTruncated": True,
                "NextKeyMarker": key,
                "NextVersionIdMarker": "page-2",
            }
        symbol = key.split("/")[-2]
        return {
            "Versions": [
                {
                    "Key": key,
                    "VersionId": f"version-{symbol}",
                    "Size": 1024,
                    "IsLatest": True,
                    "LastModified": SIGNED_AT - timedelta(minutes=1),
                }
            ],
            "DeleteMarkers": [],
            "IsTruncated": False,
        }


class _Session:
    def __init__(self, s3: _S3, kms: _Kms, sts: _Sts) -> None:
        self.s3 = s3
        self.kms = kms
        self.sts = sts
        self.calls: list[tuple[str, str]] = []

    def client(self, service_name: str, *, region_name: str) -> object:
        self.calls.append((service_name, region_name))
        return {"s3": self.s3, "kms": self.kms, "sts": self.sts}[service_name]


def _patch_local_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        aws_custody,
        "validate_collection_terminal",
        lambda terminal, chains: None,
    )
    monkeypatch.setattr(
        aws_custody,
        "validate_qualification_envelope",
        lambda envelope: None,
    )
    monkeypatch.setattr(
        aws_custody,
        "_utc_now",
        lambda: SIGNED_AT + timedelta(hours=1),
    )


def test_aws_preflight_verifies_signature_identity_lock_and_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_local_validation(monkeypatch)
    config = load_aws_custody_config(_write_config(tmp_path / "custody.toml"))
    terminal, chains, prequalification = _evidence()
    attestation = _attestation()
    s3 = _S3()
    kms = _Kms()
    sts = _Sts()
    session = _Session(s3, kms, sts)

    report = run_aws_custody_preflight(
        config=config,
        attestation=attestation,
        collection_terminal=terminal,
        collection_chains=chains,
        prequalification=prequalification,
        session=session,
    )

    assert report.custody_preflight_complete
    assert report.qualification_complete
    assert not report.collection_authorized
    assert not report.scoring_authorized
    assert report.object_versions_checked == 8
    preflight_id = report.as_record()["preflight_id"]
    assert isinstance(preflight_id, str)
    assert preflight_id.startswith("sha256-")
    assert sts.called
    assert [name for name, _ in kms.calls] == ["describe_key", "verify"]
    assert session.calls == [
        ("s3", REGION),
        ("kms", REGION),
        ("sts", REGION),
    ]
    assert all(
        arguments.get("ExpectedBucketOwner") == ACCOUNT
        for _, arguments in s3.calls
    )
    verify_call = kms.calls[1][1]
    expected_digest = hashlib.sha256(
        canonical_json(
            {
                key: value
                for key, value in attestation.items()
                if key not in {"attestation_id", "signature_base64"}
            }
        )
    ).digest()
    assert verify_call["Message"] == expected_digest
    assert verify_call["MessageType"] == "DIGEST"


def test_aws_preflight_stops_before_s3_when_kms_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_local_validation(monkeypatch)
    config = load_aws_custody_config(_write_config(tmp_path / "custody.toml"))
    terminal, chains, prequalification = _evidence()
    s3 = _S3()
    sts = _Sts()
    kms = _Kms(signature_valid=False)

    with pytest.raises(ValueError, match="rejected"):
        run_aws_custody_preflight(
            config=config,
            attestation=_attestation(),
            collection_terminal=terminal,
            collection_chains=chains,
            prequalification=prequalification,
            session=_Session(s3, kms, sts),
        )

    assert s3.calls == []
    assert not sts.called


def test_aws_preflight_rejects_short_compliance_retention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_local_validation(monkeypatch)
    config = load_aws_custody_config(_write_config(tmp_path / "custody.toml"))
    terminal, chains, prequalification = _evidence()
    s3 = _S3(RETAIN_UNTIL - timedelta(seconds=1))

    with pytest.raises(ValueError, match="retention ends"):
        run_aws_custody_preflight(
            config=config,
            attestation=_attestation(),
            collection_terminal=terminal,
            collection_chains=chains,
            prequalification=prequalification,
            session=_Session(s3, _Kms(), _Sts()),
        )


def test_aws_config_rejects_any_raw_read_capability(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="values are incompatible"):
        load_aws_custody_config(
            _write_config(tmp_path / "custody.toml", raw_read=True)
        )


def test_aws_attestation_binding_fails_before_external_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_local_validation(monkeypatch)
    config = load_aws_custody_config(_write_config(tmp_path / "custody.toml"))
    terminal, chains, prequalification = _evidence()
    attestation = _attestation()
    attestation["engineering_bundle_id"] = "sha256-" + "9" * 64
    core = {
        key: value
        for key, value in attestation.items()
        if key not in {"attestation_id", "signature_base64"}
    }
    attestation["attestation_id"] = content_id(core)
    kms = _Kms()
    s3 = _S3()
    sts = _Sts()

    with pytest.raises(ValueError, match="does not bind"):
        run_aws_custody_preflight(
            config=config,
            attestation=attestation,
            collection_terminal=terminal,
            collection_chains=chains,
            prequalification=prequalification,
            session=_Session(s3, kms, sts),
        )

    assert kms.calls == []


def test_aws_preflight_uses_trusted_runtime_clock_for_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_local_validation(monkeypatch)
    monkeypatch.setattr(
        aws_custody,
        "_utc_now",
        lambda: EXPIRES_AT + timedelta(seconds=1),
    )
    config = load_aws_custody_config(_write_config(tmp_path / "custody.toml"))
    terminal, chains, prequalification = _evidence()
    session = _Session(_S3(), _Kms(), _Sts())

    with pytest.raises(ValueError, match="time window"):
        run_aws_custody_preflight(
            config=config,
            attestation=_attestation(),
            collection_terminal=terminal,
            collection_chains=chains,
            prequalification=prequalification,
            session=session,
        )

    assert session.calls == []


def test_aws_preflight_exact_filters_paginated_version_listing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_local_validation(monkeypatch)
    config = load_aws_custody_config(_write_config(tmp_path / "custody.toml"))
    terminal, chains, prequalification = _evidence()
    s3 = _PagedS3()

    report = run_aws_custody_preflight(
        config=config,
        attestation=_attestation(),
        collection_terminal=terminal,
        collection_chains=chains,
        prequalification=prequalification,
        session=_Session(s3, _Kms(), _Sts()),
    )

    assert report.object_versions_checked == 8
    listing_calls = [
        arguments for name, arguments in s3.calls if name == "list_object_versions"
    ]
    assert len(listing_calls) == 16
    assert all("KeyMarker" in arguments for arguments in listing_calls[1::2])

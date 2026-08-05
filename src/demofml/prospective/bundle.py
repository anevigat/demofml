"""Standalone engineering-only Campaign 2 bundle freeze and verification."""

import hashlib
import os
import platform
from importlib import metadata
from pathlib import Path, PurePosixPath

import pyarrow as pa  # type: ignore[import-untyped]

from demofml.bars.prospective import PROSPECTIVE_BAR_SCHEMA
from demofml.data.prospective_ticks import PROSPECTIVE_TICK_SCHEMA
from demofml.features.cross_pair import (
    CANDIDATE_FEATURE_SCHEMA,
    CONTROL_FEATURE_SCHEMA,
)
from demofml.prospective.campaigns import (
    CAMPAIGN_V1,
    CampaignSpec,
    campaign_spec,
)
from demofml.prospective.config import (
    Campaign2EngineeringConfig,
    load_campaign2_engineering_config,
)
from demofml.prospective.opportunities import opportunity_schema
from demofml.prospective.records import (
    CONTENT_ID_PATTERN,
    IMAGE_DIGEST_PATTERN,
    SHA256_PATTERN,
    content_id,
    read_strict_json,
    sha256_file,
    write_immutable_json,
)

ENGINEERING_BUNDLE_SET_ID = CAMPAIGN_V1.engineering_bundle_set_id
ENGINEERING_BUNDLE_SCOPE = "engineering_contract_only_no_models_or_data"
_AUTHORIZATION = {
    "engineering": True,
    "fitting": False,
    "scoring": False,
    "collection": False,
    "evaluation": False,
    "raw_prospective_access": False,
}
_V1_SOURCE_PATHS = (
    "pyproject.toml",
    "configs/features/causal-v1.toml",
    "src/demofml/data/ticks.py",
    "src/demofml/data/prospective_ticks.py",
    "src/demofml/bars/quotes.py",
    "src/demofml/bars/prospective.py",
    "src/demofml/calendars/prospective_fx.py",
    "src/demofml/features/causal.py",
    "src/demofml/features/cross_pair.py",
    "src/demofml/prospective/__init__.py",
    "src/demofml/prospective/config.py",
    "src/demofml/prospective/records.py",
    "src/demofml/prospective/custody.py",
    "src/demofml/prospective/opportunities.py",
    "src/demofml/prospective/qualification.py",
    "src/demofml/prospective/verify.py",
    "src/demofml/prospective/bundle.py",
    "tests/unit/test_prospective_data.py",
    "tests/unit/test_cross_pair_features.py",
    "tests/unit/test_prospective_opportunities.py",
    "tests/unit/test_prospective_custody.py",
    "tests/unit/test_campaign2_verify.py",
    "docs/research/campaign-2-onprem-custody-requirements.md",
)


def _source_paths(campaign: CampaignSpec) -> tuple[str, ...]:
    if campaign is CAMPAIGN_V1:
        return _V1_SOURCE_PATHS
    return (
        *_V1_SOURCE_PATHS,
        "src/demofml/prospective/campaigns.py",
        "docs/research/campaign-2-prospective-factor-plan.md",
        "docs/research/campaign-2-v1-qualification-blocker-2026-08-05.md",
    )


def _expected_snapshot_paths(campaign: CampaignSpec) -> set[str]:
    contracts = {
        campaign.engineering_config_relative_path,
        campaign.protocol_relative_path,
        "configs/bars/prospective-quote-bars-v1.toml",
        "configs/features/cross-pair-v1.toml",
        "configs/experiments/prospective-executable-v1.toml",
    }
    return {f"snapshot/{path}" for path in (*_source_paths(campaign), *contracts)}


def _schema_sha256(schema: pa.Schema) -> str:
    return hashlib.sha256(schema.serialize().to_pybytes()).hexdigest()


def _runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "numpy": metadata.version("numpy"),
        "pyarrow": metadata.version("pyarrow"),
        "scikit_learn": metadata.version("scikit-learn"),
        "tzdata": metadata.version("tzdata"),
    }


def _schema_hashes(campaign: CampaignSpec) -> dict[str, str]:
    return {
        "prospective_ticks": _schema_sha256(PROSPECTIVE_TICK_SCHEMA),
        "prospective_bars": _schema_sha256(PROSPECTIVE_BAR_SCHEMA),
        "control_features": _schema_sha256(CONTROL_FEATURE_SCHEMA),
        "candidate_features": _schema_sha256(CANDIDATE_FEATURE_SCHEMA),
        "opportunities": _schema_sha256(opportunity_schema(campaign)),
    }


def _source_inventory(config: Campaign2EngineeringConfig) -> list[dict[str, object]]:
    roles = {
        config.path: "engineering_config",
        config.protocol_path: "protocol",
        config.bar_config_path: "bar_config",
        config.feature_config_path: "feature_config",
        config.label_contract_path: "label_contract",
    }
    for relative in _source_paths(config.spec):
        unresolved = config.project_root / relative
        if unresolved.is_symlink():
            raise RuntimeError(f"engineering bundle source is a symlink: {relative}")
        path = unresolved.resolve()
        if not path.is_file():
            raise RuntimeError(f"engineering bundle source is missing: {relative}")
        roles.setdefault(path, "implementation")
    records: list[dict[str, object]] = []
    for path, role in sorted(
        roles.items(),
        key=lambda item: item[0].relative_to(config.project_root).as_posix(),
    ):
        relative = path.relative_to(config.project_root).as_posix()
        payload = path.read_bytes()
        records.append(
            {
                "path": f"snapshot/{relative}",
                "role": role,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "payload": payload,
            }
        )
    return records


def freeze_engineering_bundle(
    config_path: Path,
    output: Path,
    code_reference: str,
) -> dict[str, object]:
    """Freeze code and contracts only; never models, raw data, scores, or outcomes."""
    if IMAGE_DIGEST_PATTERN.fullmatch(code_reference) is None:
        raise ValueError("code_reference must be an immutable sha256 image digest")
    config = load_campaign2_engineering_config(config_path)
    config.spec.require_artifact_creation()
    requested = output.expanduser().absolute()
    if requested.exists() or requested.is_symlink():
        raise RuntimeError(f"refusing to replace engineering bundle: {requested}")
    destination = requested.parent.resolve() / requested.name
    if destination.exists() or destination.is_symlink():
        raise RuntimeError(f"refusing to replace engineering bundle: {destination}")
    if not destination.parent.is_dir():
        raise ValueError("engineering bundle parent directory does not exist")
    inventory_with_payload = _source_inventory(config)
    inventory = [
        {key: value for key, value in record.items() if key != "payload"}
        for record in inventory_with_payload
    ]
    contract_roles = {
        "engineering_config": "engineering",
        "protocol": "protocol",
        "bar_config": "bars",
        "feature_config": "features",
        "label_contract": "labels",
    }
    contract_sha256 = {
        contract_roles[str(record["role"])]: str(record["sha256"])
        for record in inventory
        if record["role"] in contract_roles
    }
    if set(contract_sha256) != {
        "engineering",
        "protocol",
        "bars",
        "features",
        "labels",
    }:
        raise RuntimeError("captured bundle contracts do not reconcile")
    core: dict[str, object] = {
        "format_version": 1,
        "bundle_set": config.spec.engineering_bundle_set_id,
        "campaign_id": config.spec.campaign_id,
        "bundle_scope": ENGINEERING_BUNDLE_SCOPE,
        "code_reference": code_reference,
        "authorization": _AUTHORIZATION,
        "scoring_authorized": False,
        "contract_sha256": contract_sha256,
        "runtime_versions": _runtime_versions(),
        "schema_sha256": _schema_hashes(config.spec),
        "inventory": inventory,
    }
    manifest = {**core, "bundle_id": content_id(core)}

    destination.mkdir(mode=0o755, exist_ok=False)
    for record in inventory_with_payload:
        relative = str(record["path"])
        payload = record["payload"]
        if not isinstance(payload, bytes):
            raise AssertionError("captured bundle payload must be bytes")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _write_immutable_bytes(target, payload)
    write_immutable_json(destination / "bundle.json", manifest)
    marker = {
        "format_version": 1,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(destination / "bundle.json"),
        "bundle_scope": ENGINEERING_BUNDLE_SCOPE,
        "scoring_authorized": False,
    }
    write_immutable_json(destination / "_ENGINEERING_ONLY", marker)
    _fsync_tree_directories(destination)
    _fsync_directory(destination)
    _fsync_directory(destination.parent)
    verify_engineering_bundle(destination, str(manifest["bundle_id"]))
    return manifest


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"refusing to replace bundle artifact: {path}")
    with path.open("xb") as target:
        target.write(payload)
        target.flush()
        os.fsync(target.fileno())
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree_directories(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    ordered = sorted(
        directories, key=lambda path: len(path.parts), reverse=True
    )
    for directory in ordered:
        _fsync_directory(directory)


def _validate_snapshot_path(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("bundle inventory path must be a string")
    path = PurePosixPath(value)
    if (
        not value.startswith("snapshot/")
        or value.startswith("/")
        or "\\" in value
        or "//" in value
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.suffix not in {".py", ".toml", ".md"}
    ):
        raise ValueError("bundle inventory path is unsafe")
    return value


def verify_engineering_bundle(
    root: Path,
    expected_bundle_id: str,
) -> dict[str, object]:
    """Verify bundle bytes against a separately trusted content identity."""
    if CONTENT_ID_PATTERN.fullmatch(expected_bundle_id) is None:
        raise ValueError("expected_bundle_id must be a trusted content ID")
    requested = root.expanduser().absolute()
    if requested.is_symlink():
        raise ValueError("engineering bundle root cannot be a symlink")
    bundle_root = requested.resolve()
    if not bundle_root.is_dir():
        raise ValueError("engineering bundle root must be a non-symlink directory")
    manifest_path = bundle_root / "bundle.json"
    marker_path = bundle_root / "_ENGINEERING_ONLY"
    manifest = read_strict_json(manifest_path, "engineering bundle manifest")
    expected_manifest_fields = {
        "format_version",
        "bundle_set",
        "campaign_id",
        "bundle_scope",
        "code_reference",
        "authorization",
        "scoring_authorized",
        "contract_sha256",
        "runtime_versions",
        "schema_sha256",
        "inventory",
        "bundle_id",
    }
    if set(manifest) != expected_manifest_fields:
        raise ValueError("engineering bundle manifest fields are incompatible")
    campaign = campaign_spec(manifest["campaign_id"])
    if (
        type(manifest["format_version"]) is not int
        or manifest["format_version"] != 1
        or manifest["bundle_set"] != campaign.engineering_bundle_set_id
        or manifest["bundle_scope"] != ENGINEERING_BUNDLE_SCOPE
        or not isinstance(manifest["authorization"], dict)
        or any(type(value) is not bool for value in manifest["authorization"].values())
        or manifest["authorization"] != _AUTHORIZATION
        or manifest["scoring_authorized"] is not False
        or not isinstance(manifest["code_reference"], str)
        or IMAGE_DIGEST_PATTERN.fullmatch(manifest["code_reference"]) is None
        or not isinstance(manifest["bundle_id"], str)
        or CONTENT_ID_PATTERN.fullmatch(manifest["bundle_id"]) is None
        or manifest["bundle_id"] != expected_bundle_id
    ):
        raise ValueError("engineering bundle identity is incompatible")
    for field in ("contract_sha256", "schema_sha256"):
        values = manifest[field]
        if not isinstance(values, dict) or not values:
            raise ValueError(f"bundle {field} must be a non-empty object")
        if any(
            not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
            for value in values.values()
        ):
            raise ValueError(f"bundle {field} contains an invalid digest")
    if set(manifest["contract_sha256"]) != {
        "engineering",
        "protocol",
        "bars",
        "features",
        "labels",
    } or set(manifest["schema_sha256"]) != {
        "prospective_ticks",
        "prospective_bars",
        "control_features",
        "candidate_features",
        "opportunities",
    }:
        raise ValueError("bundle contract or schema digest roles are incompatible")
    runtime = manifest["runtime_versions"]
    if not isinstance(runtime, dict) or set(runtime) != {
        "python",
        "python_implementation",
        "numpy",
        "pyarrow",
        "scikit_learn",
        "tzdata",
    } or any(not isinstance(value, str) or not value for value in runtime.values()):
        raise ValueError("bundle runtime versions are incompatible")
    inventory = manifest["inventory"]
    if not isinstance(inventory, list) or not inventory:
        raise ValueError("bundle inventory must be a non-empty list")
    expected_paths: set[str] = set()
    contract_inventory: dict[str, str] = {}
    previous_path: str | None = None
    for record in inventory:
        if not isinstance(record, dict) or set(record) != {
            "path",
            "role",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("bundle inventory record fields are incompatible")
        relative = _validate_snapshot_path(record["path"])
        if previous_path is not None and relative <= previous_path:
            raise ValueError("bundle inventory paths must be unique and ordered")
        previous_path = relative
        expected_paths.add(relative)
        if record["role"] not in {
            "engineering_config",
            "protocol",
            "bar_config",
            "feature_config",
            "label_contract",
            "implementation",
        }:
            raise ValueError("bundle inventory role is incompatible")
        size = record["size_bytes"]
        digest = record["sha256"]
        if (
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            or not isinstance(digest, str)
            or SHA256_PATTERN.fullmatch(digest) is None
        ):
            raise ValueError("bundle inventory size or digest is incompatible")
        artifact = bundle_root / relative
        if (
            not artifact.is_file()
            or artifact.is_symlink()
            or artifact.stat().st_size != size
            or sha256_file(artifact) != digest
        ):
            raise ValueError(f"bundle artifact differs: {relative}")
        contract_role = {
            "engineering_config": "engineering",
            "protocol": "protocol",
            "bar_config": "bars",
            "feature_config": "features",
            "label_contract": "labels",
        }.get(str(record["role"]))
        if contract_role is not None:
            if contract_role in contract_inventory:
                raise ValueError("bundle repeats a contract role")
            contract_inventory[contract_role] = digest
    if contract_inventory != manifest["contract_sha256"]:
        raise ValueError("bundle contract hashes do not reconcile with inventory")
    if expected_paths != _expected_snapshot_paths(campaign):
        raise ValueError(
            "bundle inventory differs from the frozen engineering allowlist"
        )
    if manifest["schema_sha256"] != _schema_hashes(campaign):
        raise ValueError("bundle schema hashes differ from the trusted verifier")
    actual_paths = {
        path.relative_to(bundle_root).as_posix()
        for path in bundle_root.rglob("*")
        if path.is_file()
    }
    if any(path.is_symlink() for path in bundle_root.rglob("*")):
        raise ValueError("engineering bundle cannot contain symlinks")
    if actual_paths != expected_paths | {"bundle.json", "_ENGINEERING_ONLY"}:
        raise ValueError("engineering bundle inventory is not exact")
    core = {key: value for key, value in manifest.items() if key != "bundle_id"}
    if manifest["bundle_id"] != content_id(core):
        raise ValueError("engineering bundle content ID mismatch")

    bundled_config = (
        bundle_root / "snapshot" / campaign.engineering_config_relative_path
    )
    validated_config = load_campaign2_engineering_config(bundled_config)
    if (
        validated_config.spec != campaign
        or validated_config.contract_sha256 != manifest["contract_sha256"]
    ):
        raise ValueError("bundled contracts fail engineering-only validation")

    marker = read_strict_json(marker_path, "engineering-only marker")
    expected_marker = {
        "format_version": 1,
        "bundle_id": manifest["bundle_id"],
        "bundle_manifest_sha256": sha256_file(manifest_path),
        "bundle_scope": ENGINEERING_BUNDLE_SCOPE,
        "scoring_authorized": False,
    }
    if (
        type(marker.get("format_version")) is not int
        or type(marker.get("scoring_authorized")) is not bool
        or marker != expected_marker
    ):
        raise ValueError("engineering-only marker does not bind the bundle")
    return manifest

from pathlib import Path

import pytest

from demofml.prospective.verify import build_engineering_verification, main

PROJECT_ROOT = Path(__file__).parents[2]
CONFIG = PROJECT_ROOT / "configs/prospective/campaign-2-engineering-v1.toml"
CODE_REFERENCE = "sha256:" + "a" * 64
BASE_IMAGE = "anevigat/demofml@sha256:" + "b" * 64


def test_campaign2_engineering_verification_is_data_free_and_non_authorizing() -> None:
    report = build_engineering_verification(
        CONFIG,
        code_reference=CODE_REFERENCE,
        base_image_reference=BASE_IMAGE,
    )

    assert report["engineering_verified"] is True
    assert report["deployment_scope"] == "onprem_kubernetes_engineering_only"
    assert report["qualification_complete"] is False
    assert report["collection_authorized"] is False
    assert report["fitting_authorized"] is False
    assert report["scoring_authorized"] is False
    assert report["evaluation_authorized"] is False
    assert report["raw_access_authorized"] is False
    assert str(report["verification_id"]).startswith("sha256-")


def test_campaign2_verify_cli_publishes_immutable_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "report.json"
    main(
        [
            "--config",
            str(CONFIG),
            "--code-reference",
            CODE_REFERENCE,
            "--base-image-reference",
            BASE_IMAGE,
            "--output",
            str(output),
        ]
    )

    assert output.is_file()
    assert '"engineering_verified":true' in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="already exists"):
        main(
            [
                "--config",
                str(CONFIG),
                "--code-reference",
                CODE_REFERENCE,
                "--base-image-reference",
                BASE_IMAGE,
                "--output",
                str(output),
            ]
        )


def test_campaign2_verification_rejects_mutable_image_reference() -> None:
    with pytest.raises(ValueError, match="immutable runtime image"):
        build_engineering_verification(
            CONFIG,
            code_reference=CODE_REFERENCE,
            base_image_reference="anevigat/demofml:main",
        )

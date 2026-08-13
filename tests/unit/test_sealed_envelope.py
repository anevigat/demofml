from pathlib import Path

import pytest

from demofml.research.envelope import (
    ENVELOPE_SCHEMA_ID,
    REQUIRED_ROLES,
    file_sha256,
    load_sealed_envelope,
    main,
    verify_sealed_envelope,
)

_DOCUMENTS = {
    "hypothesis": "docs/research/campaign-x-hypothesis-v1.md",
    "validation": "configs/experiments/campaign-x-validation-v1.toml",
    "model": "configs/experiments/campaign-x-model-v1.toml",
    "acceptance": "configs/experiments/campaign-x-acceptance-v1.toml",
}


def _seal(root: Path, overrides: dict[str, str] | None = None) -> Path:
    """Write four sealed documents plus an envelope that seals them."""
    for role, relative in _DOCUMENTS.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"{role} content\n", encoding="utf-8")
    lines = [
        "format_version = 1",
        f'schema = "{ENVELOPE_SCHEMA_ID}"',
        'id = "campaign-x-envelope-v1"',
        'campaign = "campaign-3"',
        "sealed_at = 2026-08-13T00:00:00Z",
        'root = "../.."',
    ]
    for role, relative in _DOCUMENTS.items():
        digest = (overrides or {}).get(role) or file_sha256(root / relative)
        lines.append(f"[documents.{role}]")
        lines.append(f'path = "{relative}"')
        lines.append(f'sha256 = "{digest}"')
    envelope = root / "configs" / "experiments" / "campaign-x-envelope-v1.toml"
    envelope.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return envelope


def test_sealed_envelope_verifies_every_declared_document(tmp_path: Path) -> None:
    envelope = load_sealed_envelope(_seal(tmp_path))

    digests = verify_sealed_envelope(envelope)

    assert envelope.id == "campaign-x-envelope-v1"
    assert envelope.campaign == "campaign-3"
    assert envelope.root == tmp_path.resolve()
    assert tuple(document.role for document in envelope.documents) == REQUIRED_ROLES
    assert set(digests) == set(_DOCUMENTS)
    assert envelope.resolved_path("model") == (
        tmp_path / _DOCUMENTS["model"]
    ).resolve()
    with pytest.raises(KeyError, match="data_use"):
        envelope.document("data_use")


def test_editing_a_sealed_document_breaks_the_seal(tmp_path: Path) -> None:
    envelope = load_sealed_envelope(_seal(tmp_path))
    acceptance = tmp_path / _DOCUMENTS["acceptance"]
    acceptance.write_text("relaxed threshold\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="acceptance"):
        verify_sealed_envelope(envelope)


def test_every_broken_document_is_reported_together(tmp_path: Path) -> None:
    envelope = load_sealed_envelope(_seal(tmp_path))
    (tmp_path / _DOCUMENTS["model"]).write_text("wider search\n", encoding="utf-8")
    (tmp_path / _DOCUMENTS["hypothesis"]).unlink()

    with pytest.raises(RuntimeError) as error:
        verify_sealed_envelope(envelope)

    message = str(error.value)
    assert "model" in message
    assert "hypothesis: missing" in message


def test_sealed_envelope_rejects_incomplete_and_unknown_roles(tmp_path: Path) -> None:
    envelope = _seal(tmp_path)
    text = envelope.read_text(encoding="utf-8")

    envelope.write_text(
        text + '[documents.data_use]\npath = "x.md"\nsha256 = "' + "0" * 64 + '"\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="must declare exactly"):
        load_sealed_envelope(envelope)

    envelope.write_text(
        text.replace("[documents.model]", "[documents.notes]"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must declare exactly"):
        load_sealed_envelope(envelope)


def test_sealed_envelope_rejects_malformed_declarations(tmp_path: Path) -> None:
    envelope = _seal(tmp_path)
    text = envelope.read_text(encoding="utf-8")

    schema_line = f'schema = "{ENVELOPE_SCHEMA_ID}"'
    sealed_line = "sealed_at = 2026-08-13T00:00:00Z"
    for source, replacement, expected in (
        ("format_version = 1", "format_version = 2", "format_version must be 1"),
        (schema_line, schema_line + "\nextra = 1", "unknown sealed envelope"),
        (schema_line, 'schema = "sealed-envelope-v2"', "schema is not supported"),
        (sealed_line, "sealed_at = 2026-08-13T00:00:00+02:00", "explicit UTC"),
        (sealed_line, "sealed_at = 2026-08-13", "TOML datetime"),
        ('path = "docs', 'note = "x"\npath = "docs', "unknown sealed document"),
    ):
        envelope.write_text(text.replace(source, replacement), encoding="utf-8")
        with pytest.raises(ValueError, match=expected):
            load_sealed_envelope(envelope)


def test_sealed_envelope_rejects_paths_that_escape_the_root(tmp_path: Path) -> None:
    envelope = _seal(tmp_path)
    text = envelope.read_text(encoding="utf-8")

    envelope.write_text(
        text.replace(_DOCUMENTS["model"], "../outside/model.toml"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stay inside the root"):
        load_sealed_envelope(envelope)

    envelope.write_text(
        text.replace(_DOCUMENTS["model"], "/etc/model.toml"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="stay inside the root"):
        load_sealed_envelope(envelope)


def test_sealed_envelope_rejects_invalid_digests_and_roots(tmp_path: Path) -> None:
    envelope = _seal(tmp_path, overrides={"validation": "not-a-digest"})
    with pytest.raises(ValueError, match="invalid digest"):
        load_sealed_envelope(envelope)

    envelope = _seal(tmp_path)
    text = envelope.read_text(encoding="utf-8")
    envelope.write_text(
        text.replace('root = "../.."', 'root = "../../missing"'), encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="root is not a directory"):
        load_sealed_envelope(envelope)

    envelope.write_text(
        "\n".join(
            [
                "format_version = 1",
                f'schema = "{ENVELOPE_SCHEMA_ID}"',
                'id = "campaign-x-envelope-v1"',
                'campaign = "campaign-3"',
                "sealed_at = 2026-08-13T00:00:00Z",
                'root = "../.."',
                'documents = "all four"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="documents must be a table"):
        load_sealed_envelope(envelope)

    with pytest.raises(RuntimeError, match="not a file"):
        load_sealed_envelope(tmp_path / "absent.toml")


def test_sealed_envelope_command_reports_verification(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    envelope = _seal(tmp_path)

    main(["--envelope", str(envelope)])
    assert "4 documents unchanged" in capsys.readouterr().out

    (tmp_path / _DOCUMENTS["acceptance"]).write_text("edited\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exit_error:
        main(["--envelope", str(envelope)])
    assert exit_error.value.code == 1
    assert "is broken" in capsys.readouterr().err

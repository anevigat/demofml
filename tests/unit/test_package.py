import pytest

from demofml import __version__
from demofml.cli import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main([])
    assert capsys.readouterr().out == "demofml 0.1.0\n"


def test_cli_delegates_development_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    received: list[list[str] | None] = []
    monkeypatch.setattr(
        "demofml.orchestration.development.main", lambda argv: received.append(argv)
    )

    main(["run-development", "--workdir", "/work"])

    assert received == [["--workdir", "/work"]]

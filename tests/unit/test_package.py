import pytest

from demofml import __version__
from demofml.cli import main


def test_version() -> None:
    assert __version__ == "0.1.0"


def test_cli(capsys: pytest.CaptureFixture[str]) -> None:
    main([])
    assert capsys.readouterr().out == "demofml 0.1.0\n"

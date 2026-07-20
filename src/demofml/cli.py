import argparse
from collections.abc import Sequence

from demofml import __version__


def main(argv: Sequence[str] | None = None) -> None:
    """Run the demofml command line interface."""
    parser = argparse.ArgumentParser(prog="demofml")
    parser.add_argument("command", nargs="?", choices=["smoke-infra"])
    arguments = parser.parse_args(argv)

    if arguments.command == "smoke-infra":
        from demofml.infrastructure import run_infrastructure_smoke

        run_infrastructure_smoke()
        return

    print(f"demofml {__version__}")

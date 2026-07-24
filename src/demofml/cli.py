import argparse
import sys
from collections.abc import Sequence

from demofml import __version__


def main(argv: Sequence[str] | None = None) -> None:
    """Run the demofml command line interface."""
    values = list(argv) if argv is not None else sys.argv[1:]
    if values and values[0] == "evaluate-locked-test":
        from demofml.orchestration.locked import evaluate_main

        evaluate_main(values[1:])
        return
    if values and values[0] == "freeze-candidate":
        from demofml.orchestration.locked import freeze_main

        freeze_main(values[1:])
        return
    if values and values[0] == "evaluate-development":
        from demofml.reporting.acceptance import main as evaluate_development

        evaluate_development(values[1:])
        return
    if values and values[0] == "run-development":
        from demofml.orchestration.development import main as run_development

        run_development(values[1:])
        return

    parser = argparse.ArgumentParser(prog="demofml")
    parser.add_argument(
        "command",
        nargs="?",
        choices=[
            "evaluate-development",
            "evaluate-locked-test",
            "freeze-candidate",
            "run-development",
            "smoke-infra",
        ],
    )
    arguments, remaining = parser.parse_known_args(values)

    if arguments.command == "smoke-infra":
        if remaining:
            parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        from demofml.infrastructure import run_infrastructure_smoke

        run_infrastructure_smoke()
        return

    if remaining:
        parser.error(f"unrecognized arguments: {' '.join(remaining)}")

    print(f"demofml {__version__}")

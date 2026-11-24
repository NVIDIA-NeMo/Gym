# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""`python -m scripts.harbor_conversion_checks <command>`.

Exit codes: 0 pass or done, 1 a check failed, 2 usage or input error (on stderr).
"""

import argparse
import sys

from scripts.harbor_conversion_checks import check, controls, fake_model, first_request, subset
from scripts.harbor_conversion_checks.messages import InputError


COMMANDS = {
    "check": check,
    "controls": controls,
    "fake-model": fake_model,
    "first-request": first_request,
    "subset": subset,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.harbor_conversion_checks",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, module in COMMANDS.items():
        summary = (module.__doc__ or "").strip().split("\n")[0]
        command = commands.add_parser(
            name, help=summary, description=module.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
        )
        module.add_arguments(command)
        command.set_defaults(run=module.run)
    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except InputError as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 2
    except (OSError, UnicodeError) as exc:
        print(f"{args.command}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

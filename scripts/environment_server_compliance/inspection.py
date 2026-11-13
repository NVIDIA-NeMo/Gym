# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Judge the server classes of one component against the rules, and write the findings as JSON.

Run as a module in the component's own venv, so that importing the component sees its dependencies.
"""

import argparse
import json
from pathlib import Path

from .component import server_classes
from .engine import KINDS, findings
from .rules import rules_for


def inspect_component(component: str) -> list[dict]:
    kind = KINDS[component.split("/", 1)[0]]
    entrypoint, classes = server_classes(Path(component), kind)
    return findings(f"{component}/{entrypoint}", classes, rules_for(kind), component=Path(component))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", help="component directory, such as responses_api_agents/my_agent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inspect_component(args.component)) + "\n")


if __name__ == "__main__":
    main()

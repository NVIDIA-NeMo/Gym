# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rules over the server classes that a component's `app.py` defines.

Run as a module in the component's own venv, it imports the component, so the import sees the component's
dependencies, and writes the findings as JSON.
"""

import argparse
import importlib
import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


AGENT_SERVER = "agent_server"
RESOURCES_SERVER = "resources_server"
Kind = Literal["agent_server", "resources_server"]

KINDS: dict[str, Kind] = {"responses_api_agents": AGENT_SERVER}


@dataclass(frozen=True)
class Rule:
    """A requirement on one kind of component. `message` states the expected pattern.

    `violation` returns the offending function or class, which locates the warning, or None.
    """

    id: str
    kind: Kind
    message: str
    violation: Callable[[type], object | None]


def defined_run(agent: type) -> object | None:
    return vars(agent).get("run")


RULES = (
    Rule(
        "A1", AGENT_SERVER, "Agent servers should not define `run()`; implement agent sessions instead.", defined_run
    ),
)


def _base_class(kind: Kind) -> type:
    if kind == AGENT_SERVER:
        from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent

        return SimpleResponsesAPIAgent
    if kind == RESOURCES_SERVER:
        from nemo_gym.base_resources_server import SimpleResourcesServer

        return SimpleResourcesServer
    raise ValueError(f"Unknown component kind: {kind!r}")


def findings(path: str, classes: list[type], kind: Kind) -> list[dict]:
    """Apply the rules for `kind` to the server classes defined in the file `path`."""
    results = []
    for cls in classes:
        for rule in (rule for rule in RULES if rule.kind == kind):
            violation = rule.violation(cls)
            if violation is not None:
                line = inspect.getsourcelines(inspect.unwrap(violation))[1]
                results.append({"rule": rule.id, "path": path, "line": line, "message": rule.message})
    return results


def inspect_component(component: str) -> list[dict]:
    kind = KINDS[component.split("/", 1)[0]]
    base = _base_class(kind)
    module = importlib.import_module(f"{component.replace('/', '.')}.app")
    classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and issubclass(value, base) and value.__module__ == module.__name__
    ]
    return findings(f"{component}/app.py", classes, kind)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", help="component directory, such as responses_api_agents/my_agent")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(inspect_component(args.component)) + "\n")


if __name__ == "__main__":
    main()

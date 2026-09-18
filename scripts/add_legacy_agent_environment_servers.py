# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declare a legacy-agent environment server beside every bound agent instance.

Rollout collection dispatches to environment servers, so every agent that still owns its episode
through `run()` needs one in front of it. Unbound agent templates get none: they are swap sources,
and composition replaces them before anything runs.

A server is named after the environment stem rather than the agent, so swapping the agent leaves
the name alone.

    python scripts/add_legacy_agent_environment_servers.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
ROOTS = ("benchmarks", "environments", "resources_servers", "responses_api_agents", "responses_api_models")
SUFFIX = "_environment_server"

DECLARED = """
{server}:
  environment_servers:
    legacy_agent:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: {agent}
"""

# Inheriting also retires the base's server: `_inherit_from` pops what it names.
INHERITED = """
{server}:
  _inherit_from: {source}
  environment_servers:
    legacy_agent:
      agent_server:
        name: {agent}
"""


def agent_type_of(instance: dict) -> str | None:
    agents = instance.get("responses_api_agents")
    if not isinstance(agents, dict) or len(agents) != 1:
        return None
    return next(iter(agents))


def is_bound(instance: dict) -> bool:
    """An unbound template leaves `resources_server.name` unset for composition to fill."""
    agent_type = agent_type_of(instance)
    agent = instance.get("responses_api_agents", {}).get(agent_type) if agent_type else None
    if not isinstance(agent, dict):
        return False
    return (agent.get("resources_server") or {}).get("name") != "???"


def server_name(instance_name: str, agent_type: str) -> str:
    """Strip the trailing agent type, matching `_composed_instance_name` in global_config."""
    stem = instance_name.removesuffix(f"_{agent_type}").removesuffix(agent_type).rstrip("_")
    if stem == instance_name:
        stem = instance_name.removesuffix("_agent").rstrip("_")
    return f"{stem}{SUFFIX}" if stem else f"{agent_type}{SUFFIX}"


def config_files() -> list[Path]:
    found = []
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.yaml")):
            if ".venv" not in path.parts and "site-packages" not in path.parts:
                found.append(path)
    return found


def load(path: Path) -> dict | None:
    try:
        document = yaml.safe_load(path.read_text())
    except Exception:
        return None
    return document if isinstance(document, dict) else None


def server_names_for(document: dict) -> dict[str, str]:
    """Map each bound agent instance in one document to its server name.

    Two harnesses for one environment share a stem, so a tie falls back to the full instance name.
    """
    stems: dict[str, str] = {}
    for name, instance in document.items():
        agent_type = agent_type_of(instance) if isinstance(instance, dict) else None
        if agent_type and is_bound(instance):
            stems[name] = server_name(name, agent_type)
    taken = [s for s in stems.values()]
    return {name: stem if taken.count(stem) == 1 else f"{name}{SUFFIX}" for name, stem in stems.items()}


def bound_agents_in_repo() -> dict[str, str]:
    """Map every bound agent instance in the repo to the server name declared beside it."""
    index: dict[str, str] = {}
    for path in config_files():
        document = load(path)
        if document is not None:
            index.update(server_names_for(document))
    return index


def stanzas_for(document: dict, bound_agents: dict[str, str]) -> list[str]:
    blocks: list[str] = []
    declared: set[str] = set()
    servers = server_names_for(document)
    for name, instance in document.items():
        server = servers.get(name)
        if server is None or server in document or server in declared:
            continue
        declared.add(server)
        source = instance.get("_inherit_from")
        inherited = bound_agents.get(source) if isinstance(source, str) else None
        if inherited and inherited != server:
            blocks.append(INHERITED.format(server=server, source=inherited, agent=name))
        else:
            blocks.append(DECLARED.format(server=server, agent=name))
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Report what is missing without writing.")
    args = parser.parse_args()

    bound_agents = bound_agents_in_repo()
    changed, added = 0, 0
    for path in config_files():
        document = load(path)
        if document is None:
            continue
        blocks = stanzas_for(document, bound_agents)
        if not blocks:
            continue
        if not args.check:
            text = path.read_text()
            if not text.endswith("\n"):
                text += "\n"
            path.write_text(text + "".join(blocks))
        changed += 1
        added += len(blocks)
        print(f"{'would add' if args.check else 'added'} {len(blocks)} to {path.relative_to(REPO)}")

    print(f"\nfiles touched: {changed}, blocks added: {added}")
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())

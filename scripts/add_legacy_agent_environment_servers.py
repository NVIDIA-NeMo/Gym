# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declare a legacy-agent environment server beside every bound agent instance.

Rollout collection dispatches to environment servers.
Every agent that still owns its episode through `run()` therefore needs one in front of it.
Unbound agent templates get none: they are swap sources, and composition replaces them before anything runs.
An agent that an environment server already names gets none either.
A second server in front of it would make agent-routed rows ambiguous.

A server is named after the environment stem rather than the agent, so swapping the agent leaves
the name alone.

    # Every config in this repository
    python scripts/add_legacy_agent_environment_servers.py [--check]

    # Your own configs (files or directories)
    python scripts/add_legacy_agent_environment_servers.py my_configs/ my_run.yaml [--check]

This repository's configs are always indexed.
A config that renames one of their agents with `_inherit_from` therefore gets a server that inherits that agent's server.
Inheriting moves the source server, so the renamed agent ends up with exactly one server.
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

LEGACY_AGENT = "legacy_agent"

# Inheriting also retires the base's server: `_inherit_from` pops what it names.
INHERITED = """
{server}:
  _inherit_from: {source}
  environment_servers:
    {server_type}:
      agent_server:
        name: {agent}
"""

# The renamed agent's server already has the name this script would give it, so point it at the new name.
RETARGETED = """
{server}:
  environment_servers:
    {server_type}:
      agent_server:
        name: {agent}
"""


class UnreadableConfigError(ValueError):
    """A config file cannot be read or parsed."""


def agent_type_of(instance: dict) -> str | None:
    agents = instance.get("responses_api_agents")
    if not isinstance(agents, dict) or len(agents) != 1:
        return None
    return next(iter(agents))


def is_rename(instance: dict) -> bool:
    """True for an instance that only renames another one through `_inherit_from`."""
    return set(instance) == {"_inherit_from"} and isinstance(instance["_inherit_from"], str)


def needs_environment_server(instance: dict) -> bool:
    """True for an agent instance a run dispatches to, so it needs a server in front of it.

    Two kinds are skipped. An unbound template leaves `resources_server.name` unset for
    composition to fill. A shared overlay names several benchmarks' agents to override one field
    on each; without an entrypoint or an `_inherit_from` supplying one, that name is not a server
    a run can start, and declaring a server for it strands the reference in every run that merges
    the overlay without the agent.
    """
    if is_rename(instance):
        return True
    agent_type = agent_type_of(instance)
    agent = instance.get("responses_api_agents", {}).get(agent_type) if agent_type else None
    if not isinstance(agent, dict):
        return False
    if (agent.get("resources_server") or {}).get("name") == "???":
        return False
    return bool(agent.get("entrypoint") or instance.get("_inherit_from"))


def server_name(instance_name: str, agent_type: str) -> str:
    """Strip the trailing agent type, matching `_composed_instance_name` in global_config."""
    stem = instance_name.removesuffix(f"_{agent_type}").removesuffix(agent_type).rstrip("_")
    if stem == instance_name:
        stem = instance_name.removesuffix("_agent").rstrip("_")
    return f"{stem}{SUFFIX}" if stem else f"{agent_type}{SUFFIX}"


def repo_config_files() -> list[Path]:
    found = []
    for root in ROOTS:
        for path in sorted((REPO / root).rglob("*.yaml")):
            if ".venv" not in path.parts and "site-packages" not in path.parts:
                found.append(path)
    return found


def config_files_in(paths: list[Path]) -> list[Path]:
    """Expand files and directories into YAML files, in a stable order."""
    found = []
    for path in paths:
        if path.is_dir():
            found.extend(
                p
                for pattern in ("*.yaml", "*.yml")
                for p in sorted(path.rglob(pattern))
                if ".venv" not in p.parts and "site-packages" not in p.parts
            )
        else:
            found.append(path)
    return found


def load(path: Path) -> dict | None:
    """Load the Gym config in one file, or None when the file is not a mapping.

    Raises for a file that cannot be read or parsed, so a migration never skips a config silently.
    """
    try:
        document = yaml.safe_load(path.read_text())
    except (OSError, yaml.YAMLError) as error:
        raise UnreadableConfigError(f"{path}: {error}") from error
    return document if isinstance(document, dict) else None


def declared_servers(document: dict) -> dict[str, tuple[str, str]]:
    """Map each agent that an environment server in the document names to that server's name and type."""
    fronting = {}
    for name, instance in document.items():
        servers = instance.get("environment_servers") if isinstance(instance, dict) else None
        for server_type, server in servers.items() if isinstance(servers, dict) else ():
            reference = server.get("agent_server") if isinstance(server, dict) else None
            if isinstance(reference, dict) and reference.get("name"):
                fronting[reference["name"]] = (name, server_type)
    return fronting


def agent_types_in(document: dict, known_types: dict[str, str]) -> dict[str, str]:
    """Map each agent instance in the document to its agent type.

    A rename has no `responses_api_agents` body of its own.
    It takes the type of the agent it renames.
    """
    types = {}
    for name, instance in document.items():
        if not isinstance(instance, dict):
            continue
        agent_type = known_types.get(instance["_inherit_from"]) if is_rename(instance) else agent_type_of(instance)
        if agent_type is not None:
            types[name] = agent_type
    return types


def server_names_for(document: dict, known_types: dict[str, str] | None = None) -> dict[str, str]:
    """Map each bound agent instance in one document to its server name.

    Two harnesses for one environment share a stem, so a tie falls back to the full instance name.
    """
    stems: dict[str, str] = {}
    for name, agent_type in agent_types_in(document, known_types or {}).items():
        if needs_environment_server(document[name]):
            stems[name] = server_name(name, agent_type)
    taken = list(stems.values())
    return {name: stem if taken.count(stem) == 1 else f"{name}{SUFFIX}" for name, stem in stems.items()}


def index(documents: list[dict]) -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    """Index agent types and environment servers across documents.

    Returns ``(agent_types, servers)``.
    ``agent_types`` maps each agent instance to its type.
    ``servers`` maps each agent instance to the name and type of the server that names it.
    An agent no server names gets the `legacy_agent` server this script would declare.
    Renaming configs look up the agent they rename here.
    """
    agent_types: dict[str, str] = {}
    for document in documents:
        agent_types.update(agent_types_in(document, agent_types))
    # A rename may precede the document defining its source.
    for document in documents:
        agent_types.update(agent_types_in(document, agent_types))
    servers: dict[str, tuple[str, str]] = {}
    for document in documents:
        servers.update(
            (agent, (server, LEGACY_AGENT)) for agent, server in server_names_for(document, agent_types).items()
        )
    # A declared server wins over the generated name, wherever it is declared.
    for document in documents:
        servers.update(declared_servers(document))
    return agent_types, servers


def stanzas_for(
    document: dict, bound_agents: dict[str, tuple[str, str]], known_types: dict[str, str] | None = None
) -> list[str]:
    blocks: list[str] = []
    declared: set[str] = set()
    fronted = declared_servers(document)
    servers = server_names_for(document, known_types)
    for name, instance in document.items():
        server = servers.get(name)
        if server is None or name in fronted or server in document or server in declared:
            continue
        declared.add(server)
        source = instance.get("_inherit_from")
        inherited = bound_agents.get(source) if isinstance(source, str) else None
        if inherited is None:
            blocks.append(DECLARED.format(server=server, agent=name))
        elif inherited[0] == server:
            blocks.append(RETARGETED.format(server=server, server_type=inherited[1], agent=name))
        else:
            blocks.append(INHERITED.format(server=server, source=inherited[0], server_type=inherited[1], agent=name))
    return blocks


def append_blocks(text: str, blocks: list[str]) -> str:
    """Return ``text`` with ``blocks`` appended, leaving the existing content, comments included, unchanged."""
    return (text if text.endswith("\n") or not text else text + "\n") + "".join(blocks)


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Config files or directories to update. Default: this repository's configs.",
    )
    parser.add_argument("--check", action="store_true", help="Report what is missing without writing.")
    args = parser.parse_args(argv)

    targets = config_files_in(args.paths) if args.paths else repo_config_files()
    unreadable: list[str] = []
    documents: dict[Path, dict] = {}
    for path in targets:
        try:
            document = load(path)
        except UnreadableConfigError as error:
            unreadable.append(str(error))
            continue
        if document is not None:
            documents[path] = document

    repo_documents = []
    for path in repo_config_files():
        try:
            document = load(path)
        except UnreadableConfigError:
            continue
        if document is not None:
            repo_documents.append(document)
    agent_types, bound_agents = index(repo_documents + list(documents.values()))

    changed, added = 0, 0
    for path, document in documents.items():
        blocks = stanzas_for(document, bound_agents, agent_types)
        if not blocks:
            continue
        if not args.check:
            path.write_text(append_blocks(path.read_text(), blocks))
        changed += 1
        added += len(blocks)
        print(f"{'would add' if args.check else 'added'} {len(blocks)} to {display(path)}")

    for message in unreadable:
        print(f"could not update {message}", file=sys.stderr)
    print(f"\nfiles touched: {changed}, blocks added: {added}, unreadable files: {len(unreadable)}")
    if unreadable:
        return 2
    return 1 if (args.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())

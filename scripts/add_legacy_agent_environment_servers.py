# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Declare a legacy-agent environment server beside every bound agent instance.

Rollout collection dispatches to environment servers.
Every agent that still owns its episode through `run()` therefore needs one in front of it.
Unbound agent templates get none: they are swap sources, and composition replaces them before anything runs.
An agent that an environment server already references gets none either.
A second server in front of it would make agent-routed rows ambiguous.

An environment server can reference several agents, such as a user and an assistant.
It references an agent through any field whose value has `type: responses_api_agents`,
and through `agent_server`, whose type may be left out.

A server is named after the environment stem rather than the agent, so swapping the agent leaves
the name alone.

    # Every config in this repository
    python scripts/add_legacy_agent_environment_servers.py [--check]

    # Your own configs (files or directories)
    python scripts/add_legacy_agent_environment_servers.py my_configs/ my_run.yaml [--check]

This repository's configs are always indexed, so a config may rename one of their agents with `_inherit_from`.
Gym automatically updates environment server references for an unambiguous top-level agent rename.
The script leaves those renames alone and declares a server only for an unfronted source agent.
If the source's config is not migrated, Gym generates a legacy relay for the renamed agent,
or reports it when `error_on_agent_without_environment_server` is enabled.

`--switch-simple-agent` replaces each `legacy_agent` relay in front of a `simple_agent` in the given configs
with a `single_agent_turn_legacy` server of the same name, so rows keep their routing.
That server runs the episode in steps, so a crash repeats only the current step instead of the whole episode.
It binds the agent's own Resources Server, as the relay did, and grants the agent direct HTTP access to its tools,
the routes the agent's `/run` called. The server names no Resources Server itself, so it follows an agent rename.
A relay stays when its agent binds no Resources Server, would get a grant `simple_agent` sessions reject
(a required MCP tool access, or more than one direct HTTP tool access), or binds a Resources Server whose
`seed_session` accepts only the legacy body, not the `ResourcesSeedSessionRequest` an Environment Server sends,
or whose request models read the row's rollout coordinates, which an Environment Server does not forward.
It also stays when it sets keys the switched server would drop, or when another config merges it with a
`legacy_agent` server of the same name, unless a relay it inherits switches: it cannot stay alone then, so the
script refuses (exit 2) and names the config, which should declare its own server instead.
The script lists each with its reason.
A relay that inherits another relay keeps inheriting it, and the two must switch in the same run;
the script refuses to switch either one alone. Pass the configs that declare the parents too,
such as the resources server configs a benchmark includes, or no paths for this repository's configs.

    # Report what would switch and what stays on the relay
    python scripts/add_legacy_agent_environment_servers.py --switch-simple-agent --check
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import warnings
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[1]
ROOTS = ("benchmarks", "environments", "resources_servers", "responses_api_agents", "responses_api_models")
SUFFIX = "_environment_server"
AGENT_SERVER = "agent_server"
AGENT_SERVER_TYPE = "responses_api_agents"

DECLARED = """
{server}:
  environment_servers:
    legacy_agent:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: {agent}
"""

RESOURCES_SERVERS_ROOT = REPO / "resources_servers"
TYPED_SEED = "ResourcesSeedSessionRequest"
RELAY = "legacy_agent"
SWITCHED_AGENT_TYPE = "simple_agent"
# Without a resources_server, the server binds its agent's, so it follows an `_inherit_from` agent rename.
# The timeouts are the relay's own defaults, which the relay did not enforce and this server does.
SWITCHED = """\
{server}:
  environment_servers:
    single_agent_turn_legacy:
      entrypoint: app.py
      agent_server:
        type: responses_api_agents
        name: {agent}
      resources_tool_transports:
      - direct_http
      default_episode_timeout_seconds: 21600
      cleanup_timeout_seconds: 180
"""
# A relay that inherits another keeps inheriting it, so the parent stays a template that composition consumes.
# The parent switches in the same run, so the merge holds one environment server type.
SWITCHED_CHILD = """\
{server}:
  _inherit_from: {parent}
  environment_servers:
    single_agent_turn_legacy:
      agent_server:
        name: {agent}
"""
# The keys a switched stanza writes; a relay that sets any other key stays, so nothing it configures is dropped.
SWITCHED_RELAY_KEYS = {"entrypoint", AGENT_SERVER}
SWITCHED_INSTANCE_KEYS = {"environment_servers", "_inherit_from"}
# Request-model aliases for the row's rollout coordinates, which an Environment Server does not forward.
ROLLOUT_COORDINATES = {
    "ROLLOUT_INDEX_KEY_NAME",
    "GROUP_ID_KEY_NAME",
    "'_ng_rollout_index'",
    "'_ng_group_id'",
    "'_ng_group_attempt'",
}


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

    Unbound templates leave `resources_server.name` unset for composition to fill, or
    explicitly select the agent session interface without Resources. A shared overlay names
    several benchmarks' agents to override one field on each; without an entrypoint or an
    `_inherit_from` supplying one, that name is not a server
    a run can start, and declaring a server for it strands the reference in every run that merges
    the overlay without the agent.
    """
    agent_type = agent_type_of(instance)
    agent = instance.get("responses_api_agents", {}).get(agent_type) if agent_type else None
    if not isinstance(agent, dict):
        return False
    resources_server = agent.get("resources_server", {})
    if (resources_server or {}).get("name") == "???":
        return False
    # Explicitly unbound templates for agent sessions do not use the compatibility /run route.
    # A legacy relay would conflict with the Environment Server supplied by composition.
    # An omitted binding can inherit Resources and must still migrate.
    if (
        agent_type
        in {
            "hermes_agent",
            "pi_agent",
            "codex_agent",
            "openclaw_agent",
            "opencode_agent",
            "kilocode_agent",
            "claude_code_agent",
            "harbor_oracle_agent",
            "terminus_2_sandboxed_agent",
            "miniswe_sandboxed_agent",
        }
        and resources_server is None
    ):
        return False
    # Stirrup runs only in agent sessions and has no /run for a legacy relay to call.
    if agent_type == "stirrup_agent":
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


def agent_references(server: dict) -> list[tuple[str, str]]:
    """Return each ``(field, agent)`` that one environment server's config references.

    Matches `environment_server_agent_refs` in `nemo_gym/config/global_config.py`.
    """
    return [
        (field, value["name"])
        for field, value in server.items()
        if isinstance(value, dict)
        and value.get("name")
        and (value.get("type") == AGENT_SERVER_TYPE or (field == AGENT_SERVER and value.get("type") is None))
    ]


def fronted_agents(document: dict) -> set[str]:
    """Return agents referenced by an environment server in the document."""
    references: set[str] = set()
    for instance in document.values():
        servers = instance.get("environment_servers") if isinstance(instance, dict) else None
        for server in servers.values() if isinstance(servers, dict) else ():
            if isinstance(server, dict):
                references.update(agent for _, agent in agent_references(server))
    return references


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
    known_types = known_types or {}
    stems: dict[str, str] = {}
    for name, agent_type in agent_types_in(document, known_types).items():
        instance = document[name]
        source = instance.get("_inherit_from")
        if isinstance(source, str) and source in known_types:
            continue
        if needs_environment_server(instance):
            stems[name] = server_name(name, agent_type)
    taken = list(stems.values())
    return {name: stem if taken.count(stem) == 1 else f"{name}{SUFFIX}" for name, stem in stems.items()}


def index(documents: list[dict]) -> tuple[dict[str, str], set[str]]:
    """Index agent types and environment servers across documents.

    Returns ``(agent_types, references)``.
    ``agent_types`` maps each agent instance to its type.
    ``references`` contains every agent an environment server references.
    """
    agent_types: dict[str, str] = {}
    for document in documents:
        agent_types.update(agent_types_in(document, agent_types))
    # A rename may precede the document defining its source.
    for document in documents:
        agent_types.update(agent_types_in(document, agent_types))
    references: set[str] = set()
    for document in documents:
        references.update(fronted_agents(document))
    return agent_types, references


def stanzas_for(
    document: dict,
    references: set[str],
    known_types: dict[str, str] | None = None,
) -> list[str]:
    """Return declarations for bound agents that no environment server references."""
    blocks: list[str] = []
    declared: set[str] = set()
    servers = server_names_for(document, known_types)
    for name, server in servers.items():
        if name in references:
            continue
        if server in document or server in declared:
            continue
        declared.add(server)
        blocks.append(DECLARED.format(server=server, agent=name))
    return blocks


def append_blocks(text: str, blocks: list[str]) -> str:
    """Return ``text`` with ``blocks`` appended, leaving the existing content, comments included, unchanged."""
    return (text if text.endswith("\n") or not text else text + "\n") + "".join(blocks)


def _merged(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        merged[key] = (
            _merged(merged[key], value) if isinstance(value, dict) and isinstance(merged.get(key), dict) else value
        )
    return merged


def run_documents(path: Path, loaded: dict[Path, dict | None]) -> list[tuple[Path, dict]]:
    """Return the files one run of ``path`` merges, in Gym's merge order.

    Includes come before the file that pulls them in, so the including file wins.
    Instance names are unique only within a run, so each file is resolved with its own includes.
    """
    order: list[tuple[Path, dict]] = []
    seen: set[Path] = set()

    def visit(file: Path) -> None:
        file = file.resolve()
        if file in seen:
            return
        seen.add(file)
        if file not in loaded:
            try:
                loaded[file] = load(file)
            except UnreadableConfigError:
                loaded[file] = None
        document = loaded[file] or {}
        for entry in document.get("config_paths") or []:
            entry = Path(entry)
            # Gym looks under the working directory, then its install root.
            candidates = [entry] if entry.is_absolute() else [Path.cwd() / entry, REPO / entry]
            visit(next((candidate for candidate in candidates if candidate.exists()), candidates[-1]))
        order.append((file, document))

    visit(path)
    return order


def _without_deleted_keys(instance: dict) -> tuple[dict, set[str]]:
    """Remove merged keys before inheritance; return absent targets for deletion after inheritance."""
    body = dict(instance)
    deleted = {key.strip() for key in body.pop("_delete_key", "").split(",") if key.strip()}
    deferred = deleted - body.keys()
    for key in deleted:
        body.pop(key, None)
    for key, value in body.items():
        if isinstance(value, dict):
            body[key] = _without_deleted_keys(value)[0]
    return body, deferred


def resolved_instances(documents: list[dict]) -> dict[str, dict]:
    """Map each top-level instance of one run to its body, merged across files and through `_inherit_from`."""
    raw: dict[str, dict] = {}
    for document in documents:
        for name, instance in document.items():
            if isinstance(instance, dict):
                raw[name] = _merged(raw.get(name, {}), instance)
            elif instance is None:
                # A later file removes an instance by setting it to null.
                raw.pop(name, None)

    resolved: dict[str, dict] = {}

    def resolve(name: str, seen: tuple[str, ...] = ()) -> dict:
        if name in resolved:
            return resolved[name]
        # Gym applies deletions after all files merge, before following inheritance. In particular,
        # deleting _inherit_from detaches a same-name harness override from the benchmark template.
        instance, deferred = _without_deleted_keys(raw.get(name, {}))
        source = instance.get("_inherit_from")
        own = {key: value for key, value in instance.items() if key != "_inherit_from"}
        base = resolve(source, (*seen, name)) if isinstance(source, str) and source not in seen else {}
        resolved[name] = _merged(base, own)
        if isinstance(source, str):
            for key in deferred:
                resolved[name].pop(key)
        return resolved[name]

    for name in raw:
        resolve(name)
    return resolved


def relayed_agent(instance: dict) -> str | None:
    """The agent a `legacy_agent` relay instance fronts, or None for any other instance."""
    servers = instance.get("environment_servers") if isinstance(instance, dict) else None
    if not isinstance(servers, dict) or set(servers) != {RELAY} or not isinstance(servers[RELAY], dict):
        return None
    return (servers[RELAY].get(AGENT_SERVER) or {}).get("name")


def _seed_overrides(path: Path) -> list[ast.AsyncFunctionDef]:
    try:
        with warnings.catch_warnings():
            # Some servers' sources have invalid escape sequences, which only the parser warns about.
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text())
    except (OSError, SyntaxError, ValueError):
        return []
    return [node for node in ast.walk(tree) if isinstance(node, ast.AsyncFunctionDef) and node.name == "seed_session"]


def _reads_rollout_coordinates(path: Path) -> bool:
    """True when a model in ``path`` aliases a field to one of the row's rollout coordinates."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text())
    except (OSError, SyntaxError, ValueError):
        return False
    return any(
        keyword.arg == "alias" and ast.unparse(keyword.value) in ROLLOUT_COORDINATES
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        for keyword in node.keywords
    )


def resources_blocker(resources: dict) -> str | None:
    """Why an Environment Server cannot drive this Resources Server instance, or None when it can.

    An Environment Server seeds with a `ResourcesSeedSessionRequest` and expects its `resources_session_id` back.
    The base `seed_session` does that. An override must accept the typed body, or read `resources_session_id`
    from the raw request; one that takes only the legacy body would fail every seed.
    It verifies with the task's data, without the row's `_ng_*` rollout coordinates, so a server whose request
    models read them, such as a cohort comparison, would fail every verify.
    The checks read the source, so a server passes them once it migrates; a server whose source is not under
    `resources_servers/` cannot be checked, so it stays.
    """
    servers = resources.get("resources_servers") or {}
    if len(servers) != 1:
        return None
    implementation, config = next(iter(servers.items()))
    directory = RESOURCES_SERVERS_ROOT / str(implementation)
    if not directory.is_dir():
        return f"cannot find resources server {implementation} under {display(RESOURCES_SERVERS_ROOT)}"
    entrypoint = directory / str((config or {}).get("entrypoint") or "app.py")
    sources = [
        path
        for path in sorted(directory.rglob("*.py"))
        if path != entrypoint and not {"tests", ".venv"} & set(path.parts)
    ]
    overrides = _seed_overrides(entrypoint) or [override for path in sources for override in _seed_overrides(path)]
    for override in overrides:
        source = ast.unparse(override)
        if TYPED_SEED not in source and "resources_session_id" not in source:
            return f"resources server {implementation} seeds only the legacy body, not {TYPED_SEED}"
    if any(_reads_rollout_coordinates(path) for path in [entrypoint, *sources]):
        return f"resources server {implementation} reads the row's rollout coordinates, which are not forwarded"
    return None


def switch_blocker(agent: dict, instances: dict[str, dict]) -> str | None:
    """Why a `simple_agent` cannot leave the relay, or None when it can.

    The switched server seeds the agent's Resources Server and grants direct HTTP access to it.
    `simple_agent` sessions reject a required MCP grant and more than one direct HTTP grant,
    counting the accesses its own config declares.
    """
    body = (agent.get("responses_api_agents") or {}).get(SWITCHED_AGENT_TYPE) or {}
    resources = (body.get("resources_server") or {}).get("name")
    if not resources or resources == "???":
        return "the agent binds no resources server"
    if (blocker := resources_blocker(instances.get(resources, {}))) is not None:
        return blocker
    accesses = [access for access in body.get("tool_accesses") or [] if isinstance(access, dict)]
    mcp = [access.get("name") for access in accesses if access.get("kind") == "mcp" and access.get("required")]
    if mcp:
        return f"required MCP tool access: {', '.join(map(str, mcp))}"
    direct = {f"{resources}.direct_http"} | {
        str(access.get("name")) for access in accesses if access.get("kind", "direct_http") == "direct_http"
    }
    if len(direct) > 1:
        return f"more than one direct HTTP tool access: {', '.join(sorted(direct))}"
    return None


def extra_keys(instance: dict) -> list[str]:
    """The keys a relay instance sets that its switched stanza would not keep."""
    relay = (instance.get("environment_servers") or {}).get(RELAY)
    own = set(instance) - SWITCHED_INSTANCE_KEYS
    body = {f"{RELAY}.{key}" for key in relay} - {f"{RELAY}.{key}" for key in SWITCHED_RELAY_KEYS} if relay else set()
    return sorted(own | body)


def switch_plan(
    documents: dict[Path, dict], loaded: dict[Path, dict | None]
) -> tuple[dict[Path, list[tuple[str, str, str | None]]], list[tuple[Path, str, str]]]:
    """Return ``(switches, kept)`` for the `simple_agent` relays declared in ``documents``.

    ``switches`` maps each file to its ``(relay, agent, parent)`` replacements,
    where ``parent`` is the relay's own `_inherit_from`, if any.
    ``kept`` lists each ``(file, relay, reason)`` that stays on the relay.
    """
    switches: dict[Path, list[tuple[str, str, str | None]]] = {}
    kept: list[tuple[Path, str, str]] = []
    for path, document in documents.items():
        run = run_documents(path, loaded)
        instances = resolved_instances([run_document for _, run_document in run])
        # A run that cannot load an include cannot be resolved, so none of its relays switch.
        missing = [display(file) for file, _ in run if loaded[file] is None]
        for name in document:
            agent_name = relayed_agent(instances.get(name, {}))
            agent = instances.get(agent_name or "", {})
            if agent_name is None or not (missing or SWITCHED_AGENT_TYPE in (agent.get("responses_api_agents") or {})):
                continue
            blocker = (
                f"cannot load included config: {', '.join(missing)}" if missing else switch_blocker(agent, instances)
            )
            if blocker is None and (keys := extra_keys(document[name] if isinstance(document[name], dict) else {})):
                blocker = f"the relay sets {', '.join(keys)}, which the switched server would not keep"
            if blocker is not None:
                kept.append((path, name, blocker))
                continue
            parent = document[name].get("_inherit_from") if isinstance(document[name], dict) else None
            switches.setdefault(path.resolve(), []).append((name, agent_name, parent))
    return switches, kept


def unpaired_relays(
    switched: set[tuple[Path, str]], paths: list[Path], loaded: dict[Path, dict | None]
) -> list[tuple[Path, str, str]]:
    """Return each ``(file, child, parent)`` relay pair where only one of the two would switch.

    Gym merges a child over its parent, so the pair would end up with two environment server types.
    """
    unpaired = []
    for path in paths:
        run = run_documents(path, loaded)
        file, own = run[-1]
        for name, instance in own.items():
            parent = instance.get("_inherit_from") if isinstance(instance, dict) else None
            if not isinstance(parent, str) or RELAY not in (instance.get("environment_servers") or {}):
                continue
            defined_in = [run_file for run_file, document in run if parent in document]
            parent_switched = bool(defined_in) and (defined_in[-1], parent) in switched
            if parent_switched != ((file, name) in switched):
                unpaired.append((path, name, parent))
    return unpaired


def stanza(relay: str, agent: str, parent: str | None) -> str:
    """The block that replaces a relay, still inheriting its parent when it had one."""
    return (SWITCHED if parent is None else SWITCHED_CHILD).format(server=relay, agent=agent, parent=parent)


def _inheritance(name: str, documents: list[dict]) -> list[str]:
    """``name`` followed by the instances it inherits, nearest first, as the run's last definition says."""
    chain = [name]
    while True:
        source = next(
            (
                document[chain[-1]].get("_inherit_from")
                for document in reversed(documents)
                if isinstance(document.get(chain[-1]), dict) and "_inherit_from" in document[chain[-1]]
            ),
            None,
        )
        if not isinstance(source, str) or source in chain:
            return chain
        chain.append(source)


def split_servers(
    planned: dict[tuple[Path, str], tuple[str, str | None]], paths: list[Path], loaded: dict[Path, dict | None]
) -> tuple[dict[tuple[Path, str], str], list[str]]:
    """Find the runs where the planned stanzas leave one instance with two environment server types.

    Another config can merge a relay with a `legacy_agent` server of the same name, such as a benchmark
    variant that points it at another harness, or inherit a switched relay without declaring a server.
    ``planned`` maps each ``(file, relay)`` to its ``(agent, parent)``.
    Returns ``(keep, refused)``: each planned relay that can stay on the relay, with the reason, and a message
    for each split that keeping a relay cannot fix, because a relay the instance inherits switches.
    """
    after = dict(loaded)
    for (file, relay), (agent, parent) in planned.items():
        document = dict(after.get(file) or {})
        document.update(yaml.safe_load(stanza(relay, agent, parent)))
        after[file] = document
    keep: dict[tuple[Path, str], str] = {}
    refused: list[str] = []
    for path in paths:
        run = run_documents(path, after)
        files = [file for file, _ in run]
        if not any((file, relay) in planned for file in files for relay in after.get(file) or {}):
            continue
        documents = [document for _, document in run]
        for name, instance in resolved_instances(documents).items():
            servers = instance.get("environment_servers")
            if not isinstance(servers, dict) or len(servers) < 2:
                continue
            switching = [
                (file, ancestor)
                for ancestor in _inheritance(name, documents)
                for file in reversed(files)
                if (file, ancestor) in planned
            ]
            # Several files of the run can declare the instance itself; they all stay with it.
            inherited = next((ancestor for _, ancestor in switching if ancestor != name), None)
            if inherited is not None:
                refused.append(
                    f"{name} in a run of {display(path)} would merge a {RELAY} server with {inherited}, "
                    f"which switches; keep the name there and replace its definition with a complete {RELAY} server, "
                    f"deleting _inherit_from and single_agent_turn_legacy, as benchmarks/hle/opencode.yaml does"
                )
                continue
            for declaration in switching:
                keep.setdefault(declaration, f"{display(path)} merges it with a {RELAY} server")
    return keep, refused


def replace_stanza(text: str, name: str, block: str) -> str:
    """Return ``text`` with top-level instance ``name`` replaced by ``block``.

    Comments and blank lines after the instance belong to whatever follows, so they stay.
    """
    lines = text.splitlines(keepends=True)
    start = next((i for i, line in enumerate(lines) if re.match(rf"{re.escape(name)}:\s*(#.*)?$", line)), None)
    if start is None:
        raise ValueError(f"cannot find the line that starts {name!r}")
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"[^\s#]", lines[i])), len(lines))
    while end > start + 1 and (not lines[end - 1].strip() or lines[end - 1].lstrip().startswith("#")):
        end -= 1
    return "".join(lines[:start]) + block + "".join(lines[end:])


def switch(documents: dict[Path, dict], *, check: bool) -> int:
    """Replace the `simple_agent` relays in ``documents``; refuse a set that splits a parent and child relay."""
    loaded: dict[Path, dict | None] = {path.resolve(): document for path, document in documents.items()}
    switches, kept = switch_plan(documents, loaded)
    switched = {(path, relay) for path, replacements in switches.items() for relay, *_ in replacements}
    paths = list(dict.fromkeys(path.resolve() for path in repo_config_files() + list(documents)))
    unpaired = unpaired_relays(switched, paths, loaded)
    if unpaired:
        for path, child, parent in unpaired:
            print(f"{child} in {display(path)} inherits {parent}; switch both in one run", file=sys.stderr)
        return 2

    planned = {
        (path, relay): (agent, parent)
        for path, replacements in switches.items()
        for relay, agent, parent in replacements
    }
    # Keeping one relay can split a relay that inherits it, so repeat until no run splits.
    while True:
        keep, refused = split_servers(planned, paths, loaded)
        if refused:
            for message in dict.fromkeys(refused):
                print(message, file=sys.stderr)
            return 2
        if not keep:
            break
        for (path, relay), reason in keep.items():
            del planned[(path, relay)]
            kept.append((path, relay, reason))

    # Every rewrite is built before any is written, so a failure leaves no file half switched.
    texts: dict[Path, str] = {}
    for (path, relay), (agent, parent) in planned.items():
        try:
            texts[path] = replace_stanza(texts.get(path) or path.read_text(), relay, stanza(relay, agent, parent))
        except ValueError as error:
            print(f"could not rewrite {display(path)}: {error}", file=sys.stderr)
            return 2
    for path, text in texts.items():
        if not check:
            path.write_text(text)
        count = sum(1 for file, _ in planned if file == path)
        print(f"{'would switch' if check else 'switched'} {count} in {display(path)}")
    for path, relay, reason in kept:
        print(f"stays on the relay: {relay} in {display(path)}: {reason}")
    print(f"\nfiles touched: {len(texts)}, relays switched: {len(planned)}, staying on the relay: {len(kept)}")
    return 1 if (check and texts) else 0


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
    parser.add_argument(
        "--switch-simple-agent",
        action="store_true",
        help="Replace legacy_agent relays in front of simple_agent with single_agent_turn_legacy servers.",
    )
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

    if args.switch_simple_agent:
        for message in unreadable:
            print(f"could not update {message}", file=sys.stderr)
        return 2 if unreadable else switch(documents, check=args.check)

    repo_documents = []
    for path in repo_config_files():
        try:
            document = load(path)
        except UnreadableConfigError:
            continue
        if document is not None:
            repo_documents.append(document)
    agent_types, references = index(repo_documents + list(documents.values()))

    changed, added = 0, 0
    for path, document in documents.items():
        blocks = stanzas_for(document, references, agent_types)
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

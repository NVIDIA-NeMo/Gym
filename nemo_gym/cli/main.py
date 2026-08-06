# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import importlib
import logging
import os
import re
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from nemo_gym import (
    NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME,
    _augment_sys_path,
    component_search_roots,
)
from nemo_gym.cli.utils import did_you_mean
from nemo_gym.config_types import Domain


logger = logging.getLogger(__name__)

VERSION_TARGET = "nemo_gym.cli.general:version"
_INTERNAL_CONFIG_TRANSPORT_ENV_VARS = ("NEMO_GYM_CONFIG_DICT", "NEMO_GYM_CONFIG_PATH")


class _GymArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that appends a difflib "did you mean?" hint to invalid-choice errors.

    Covers mistyped commands/groups and bad --flag choices (e.g. --storage), since argparse validates all of them
    as choices against the registry baked into the parser.
    """

    def error(self, message: str) -> None:
        match = re.search(r"invalid choice: '([^']+)' \(choose from (.+)\)", message)
        if match:
            typo = match.group(1)
            choices = re.findall(r"'([^']+)'", match.group(2))
            if not choices:
                choices = [choice.strip() for choice in match.group(2).split(",")]
            message += did_you_mean(typo, choices)
        super().error(message)


@dataclass(frozen=True)
class Flag:
    # Register this flag's argument(s) on a command's subparser.
    register: Callable[[argparse.ArgumentParser], None]
    # Turn the parsed value into leading Hydra override tokens (default: contributes nothing).
    translate_to_hydra: Callable[[argparse.Namespace], list[str]] = lambda args: []


@dataclass(frozen=True)
class Command:
    # What to run: either a "module:function" string (lazily imported and called with no args),
    # or a callable(args, overrides) that owns dispatch (e.g. picks the target from parsed flags).
    target: str | Callable[[argparse.Namespace, list[str]], None]
    # One-line help shown in the parent listing and atop this command's own --help.
    summary: str | None = None
    # Flags this command accepts; reusable ones (e.g. CONFIG) are shared across commands.
    flags: tuple[Flag, ...] = field(default_factory=tuple)


def dispatch(target: str, overrides: list[str]) -> None:
    module_path, func_name = target.split(":")
    # Drop the parsed command tokens so the downstream Hydra parsing only sees overrides.
    sys.argv = [sys.argv[0], *overrides]
    func = getattr(importlib.import_module(module_path), func_name)
    func()


def _value_flag(
    name: str, hydra_key: str, flag_help: str, *, aliases: tuple[str, ...] = (), choices: tuple[str, ...] | None = None
) -> Flag:
    """A `--name VALUE` flag that maps to the Hydra override `+<hydra_key>=VALUE` (omitted when unset)."""
    dest = name.replace("-", "_")
    return Flag(
        register=lambda p: p.add_argument(f"--{name}", *aliases, dest=dest, choices=choices, help=flag_help),
        translate_to_hydra=lambda args: (
            [f"+{hydra_key}={getattr(args, dest)}"] if getattr(args, dest) is not None else []
        ),
    )


def _bool_flag(name: str, hydra_key: str, flag_help: str) -> Flag:
    """A `--name` store_true flag that maps to the Hydra override `+<hydra_key>=true` when set."""
    dest = name.replace("-", "_")
    return Flag(
        register=lambda p: p.add_argument(f"--{name}", action="store_true", help=flag_help),
        translate_to_hydra=lambda args: [f"+{hydra_key}=true"] if getattr(args, dest) else [],
    )


def _quote_hydra_string(value: str) -> str:
    """Quote a literal string for Hydra without changing its backslashes."""
    escaped = re.sub(r'(\\*)"', lambda match: "\\" * (2 * len(match.group(1)) + 1) + '"', value)
    trailing_backslashes = len(escaped) - len(escaped.rstrip("\\"))
    if trailing_backslashes:
        escaped += "\\" * trailing_backslashes
    return f'"{escaped}"'


def _quoted_value_flag(
    name: str,
    hydra_key: str,
    flag_help: str,
    *,
    aliases: tuple[str, ...] = (),
    choices: tuple[str, ...] | None = None,
    dest: str | None = None,
) -> Flag:
    """A string flag encoded as a quoted Hydra value, including when it contains grammar characters."""
    dest = dest or name.replace("-", "_")
    return Flag(
        register=lambda p: p.add_argument(f"--{name}", *aliases, dest=dest, choices=choices, help=flag_help),
        translate_to_hydra=lambda args: (
            [f"+{hydra_key}={_quote_hydra_string(getattr(args, dest))}"] if getattr(args, dest) is not None else []
        ),
    )


def _repeated_quoted_flag(name: str, hydra_key: str, flag_help: str) -> Flag:
    """A repeatable string flag encoded as a Hydra list with each item safely quoted."""
    dest = name.replace("-", "_")
    return Flag(
        register=lambda p: p.add_argument(f"--{name}", dest=dest, action="append", metavar="VALUE", help=flag_help),
        translate_to_hydra=lambda args: (
            [f"+{hydra_key}=[{','.join(_quote_hydra_string(value) for value in getattr(args, dest))}]"]
            if getattr(args, dest)
            else []
        ),
    )


# Shared flag: load Gym config files. Reused by every command that reads server/benchmark configs.
CONFIG = Flag(
    register=lambda p: p.add_argument(
        "--config",
        action="append",
        metavar="PATH",
        help="Config file to load; repeatable. Maps to +config_paths=[...].",
    ),
    translate_to_hydra=lambda args: [f"+config_paths=[{','.join(args.config)}]"] if args.config else [],
)

# Shared flag: select the storage backend. Reused by `dataset upload` and `dataset download`.
STORAGE = Flag(
    register=lambda p: p.add_argument(
        "--storage", choices=("hf", "gitlab"), default="hf", help="Storage backend (default: hf)."
    )
)

# Shared model-server flags. Reused by commands that spin up / target a model server (`eval run`, `env start`).
# --model is the served model identifier across all backends: an API model name, an HF id, or a local checkpoint
# path, interpreted per --model-type (e.g. a path/HF id to serve with local_vllm_model).
MODEL = _value_flag(
    "model",
    "policy_model_name",
    "Model name, HF id, or local checkpoint path (interpreted per --model-type).",
    aliases=("-m",),
)
MODEL_URL = _value_flag("model-url", "policy_base_url", "Model server base URL.")
MODEL_API_KEY = _value_flag("model-api-key", "policy_api_key", "Model server API key.")
ROLLOUT_DRIVER = Flag(
    register=lambda p: p.add_argument(
        "--rollout-driver",
        metavar="MODULE:FUNCTION",
        help="Temporarily replace the rollout driver declared by a manifest-bound custom-driver recipe.",
    )
)


# Shared flag: select a single resources server by name. Reused by `env test`, `env init`, and `env packages`.
def _translate_resources_server(args: argparse.Namespace) -> list[str]:
    value = getattr(args, "resources_server", None)
    if not value:
        return []
    # With a manifest-bound recipe this is a real component replacement. The
    # component-only form retains the legacy source-directory test workflow.
    if getattr(args, "environment_ref", None):
        path = _asset_config_path("resources-server", value)
        setattr(args, "_resources_server_config_path", path)
        return [f"+config_paths=[{path}]"]
    return [f"+entrypoint=resources_servers/{value}"]


RESOURCES_SERVER = Flag(
    register=lambda p: p.add_argument("--resources-server", metavar="NAME", help="Name of the resources server."),
    translate_to_hydra=_translate_resources_server,
)

# Shared flag: emit machine-readable JSON instead of human output. Reused by reporting commands (version, list,
# env status). Each command reads the reserved `json` config key ad hoc via
# global_config_dict.get(JSON_OUTPUT_KEY_NAME) (see general.py, eval.py, env.py).
JSON = _bool_flag("json", "json", "Output as machine-readable JSON.")


def _register_init_target(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resources-server", metavar="NAME", help="Scaffold a legacy resources server.")
    group.add_argument("--environment", dest="init_environment", metavar="NAME", help="Scaffold an environment.")
    group.add_argument("--benchmark", dest="init_benchmark", metavar="NAME", help="Scaffold a benchmark.")


def _translate_init_target(args: argparse.Namespace) -> list[str]:
    if args.resources_server:
        return [f"+entrypoint=resources_servers/{args.resources_server}"]
    if args.init_environment:
        return [f"+init_name={_quote_hydra_string(args.init_environment)}", '+init_kind="environment"']
    if args.init_benchmark:
        return [f"+init_name={_quote_hydra_string(args.init_benchmark)}", '+init_kind="benchmark"']
    return []


ENV_INIT_TARGET = Flag(register=_register_init_target, translate_to_hydra=_translate_init_target)

_DOMAIN_CHOICES = tuple(domain.value for domain in Domain)

CATALOG_NAME = Flag(
    register=lambda p: p.add_argument(
        "name", nargs="?", metavar="NAME", help="Inspect one environment or benchmark by name."
    ),
    translate_to_hydra=lambda args: (
        [f"+component_name={_quote_hydra_string(args.name)}"] if getattr(args, "name", None) else []
    ),
)

ENVIRONMENT_REFERENCE = Flag(
    register=lambda p: p.add_argument(
        "environment_ref",
        nargs="?",
        metavar="NAME",
        help="Live catalog name; historical NAME@VERSION execution is not supported.",
    )
)

VALIDATION_REFERENCE = Flag(
    register=lambda p: p.add_argument(
        "environment_ref",
        nargs="?",
        metavar="NAME",
        help="Catalog name, optionally qualified as NAME@VERSION to check the current manifest version.",
    )
)

PUBLISH_REFERENCE = Flag(
    register=lambda p: p.add_argument(
        "environment_ref",
        metavar="NAME",
        help="Catalog name, optionally pinned as NAME@VERSION.",
    )
)

CATALOG_KIND = Flag(
    register=lambda p: p.add_argument(
        "--kind", dest="catalog_kind", choices=("environment", "benchmark"), help="Limit to one runnable-unit kind."
    ),
    translate_to_hydra=lambda args: (
        [f'+catalog_kind="{args.catalog_kind}"'] if getattr(args, "catalog_kind", None) else []
    ),
)

REFERENCE_KIND = Flag(
    register=lambda p: p.add_argument(
        "--kind", dest="catalog_kind", choices=("environment", "benchmark"), help="Disambiguate a catalog name."
    )
)

CATALOG_FILTERS = (
    _quoted_value_flag("domain", "catalog_domain", "Filter by evaluation domain.", choices=_DOMAIN_CHOICES),
    CATALOG_KIND,
    _quoted_value_flag("modality", "catalog_modality", "Filter by modality."),
    _quoted_value_flag("licensing", "catalog_licensing", "Filter by SPDX identifier or license class."),
    _quoted_value_flag(
        "status",
        "catalog_status",
        "Filter by derived validation status.",
        choices=("experimental", "no-manifest"),
    ),
    _quoted_value_flag(
        "lifecycle", "catalog_lifecycle", "Filter by authored lifecycle.", choices=("active", "deprecated")
    ),
    _repeated_quoted_flag("requires", "catalog_requires", "Require this capability; repeatable."),
    _bool_flag("facts", "catalog_facts", "Include the underlying manifest facts."),
)

MANIFEST_EDIT_FILTERS = (
    _repeated_quoted_flag(
        "environment",
        "manifest_names",
        "Select an environment or benchmark manifest by exact name; repeatable.",
    ),
    _quoted_value_flag("domain", "catalog_domain", "Select manifests by evaluation domain.", choices=_DOMAIN_CHOICES),
    CATALOG_KIND,
    _quoted_value_flag(
        "profile",
        "catalog_profile",
        "Select manifests by integration profile.",
        choices=("stock-loop", "measured-loop", "external-loop", "custom-driver"),
    ),
)

# `gym list <type> [<name>]`: an optional component name. When given, the listing command inspects that one
# component (surfaced as the reserved `component_name` config key) instead of listing all.
NAME = Flag(
    register=lambda p: p.add_argument(
        "name", nargs="?", metavar="NAME", help="Inspect a single component by name instead of listing all."
    ),
    translate_to_hydra=lambda args: [f"+component_name={args.name}"] if getattr(args, "name", None) else [],
)

# `gym search [<type>] <query>`: an optional component type plus the query. The query is surfaced to the
# chosen listing command as the reserved `query` config key; the type only picks which command to run
# (see `_search`). A lone positional preserves the historical benchmark search.
_SEARCHABLE_TYPES = {
    "benchmarks": "nemo_gym.cli.eval:list_benchmarks",
    "environments": "nemo_gym.cli.env:list_environments",
    "catalog": "nemo_gym.cli.catalog:list_environment_catalog",
    "agents": "nemo_gym.cli.agents:list_agents",
    "models": "nemo_gym.cli.models:list_models",
    "resources-servers": "nemo_gym.cli.resources_servers:list_resources_servers",
}

SEARCH_TERMS = Flag(
    register=lambda p: (
        p.add_argument(
            "component_type",
            nargs="?",
            choices=list(_SEARCHABLE_TYPES),
            help="Component type to search (default: benchmarks; use catalog for the manifest/legacy union).",
        ),
        p.add_argument(
            "query",
            metavar="QUERY",
            help="Text matched (substring or fuzzy) against a component's name, description, and key metadata.",
        ),
    ),
    translate_to_hydra=lambda args: (
        [f"+query={_quote_hydra_string(args.query)}"] if getattr(args, "query", None) else []
    ),
)


def _search(args: argparse.Namespace, overrides: list[str]) -> None:
    """Dispatch typed search while preserving the historical benchmark default."""
    target = _SEARCHABLE_TYPES.get(getattr(args, "component_type", None), "nemo_gym.cli.eval:list_benchmarks")
    dispatch(target, overrides)


# Asset selector flag -> (parent dir, configs subdir, default config flavor). All accept `name` or `name/flavor`,
# resolving to `<parent>/<server>/[<subdir>/]<flavor>.yaml`. A None default flavor falls back to the server name.
_ASSETS = {
    "benchmark": ("benchmarks", "", "config"),
    "environment": ("environments", "", "config"),
    "resources-server": ("resources_servers", "configs", None),
    "agent-server": ("responses_api_agents", "configs", None),
    "model-type": ("responses_api_models", "configs", None),
}


def _asset_config_path(flag: str, value: str) -> str:
    """Map a named asset (`name` or `name/flavor`) to its config path.

    Searches the roots from :func:`~nemo_gym.discovery.component_search_roots` (``NEMO_GYM_EXTRA_ROOTS`` +
    cwd + install root), the same helper that backs `gym list`/`gym search`, so config resolution and
    discovery agree on where components live. On a name collision across roots the highest-priority root wins
    (as in `gym list`), with a warning. ``--search-dir`` reaches here via ``NEMO_GYM_EXTRA_ROOTS`` (set
    in ``main``). Searching the install root is what lets built-ins resolve by name from an arbitrary cwd
    (e.g. a wheel install), not just inside the repo checkout.
    """
    parent, subdir, default_flavor = _ASSETS[flag]
    server_name, _, config_flavor = value.partition("/")
    config_flavor = config_flavor or default_flavor or server_name
    config_dir = f"{parent}/{server_name}/{subdir}".rstrip("/")
    path = f"{config_dir}/{config_flavor}.yaml"

    roots = component_search_roots()
    matches: list[Path] = []

    for root in roots:
        candidate = root / path
        if candidate.exists():
            resolved = candidate.resolve()
            if resolved not in matches:
                matches.append(resolved)

    if len(matches) > 1:
        shadowed = ", ".join(f"`{m}`" for m in matches[1:])
        logger.warning(
            f"`--{flag} {value}` matches multiple configs; using `{matches[0]}` from the highest-priority root "
            f"and ignoring {shadowed}. Pass `--config <path>` to select a different one."
        )
    if matches:
        return str(matches[0])

    # No match: build a "did you mean?" hint and the roots searched
    if flag == "benchmark":
        # Benchmarks need special handling because some use non-standard config paths (arbitrary nesting), so
        # the generic one-level flavor/sibling search below can't see them.
        # Enumerate their real config names (the same values `gym list benchmarks` prints) instead.
        from nemo_gym.benchmarks import _benchmark_config_name, _benchmark_config_paths

        config_names = {
            _benchmark_config_name(p.relative_to(root / parent))
            for root in roots
            for p in _benchmark_config_paths(root / parent)
        }
        # A bare directory that only groups benchmarks (e.g. `livecodebench`) is not itself selectable, so point
        # at the config names under it; otherwise fall back to a fuzzy match across every token.
        under_dir = sorted(config_name for config_name in config_names if config_name.startswith(f"{value}/"))
        hint = f" Did you mean `{min(under_dir, key=len)}`?" if under_dir else did_you_mean(value, config_names)
        available = ", ".join(sorted(f"`{(root / parent).resolve()}`" for root in roots if (root / parent).is_dir()))
    else:
        # Suggest the closest real name across all roots: a config flavor when the server exists, else a server
        # name, reporting the full paths that were searched in each case.
        available = ", ".join(
            set(f"`{(root / config_dir).resolve()}`" for root in roots if (root / config_dir).is_dir())
        )
        typo = config_flavor
        candidates = [p.stem for root in roots for p in (root / config_dir).glob("*.yaml")]

        if len(candidates) == 0:
            available = ", ".join(set(f"`{(root / parent).resolve()}`" for root in roots if (root / parent).is_dir()))
            typo = server_name
            candidates = [
                child.name
                for root in roots
                if (root / parent).is_dir()
                for child in (root / parent).iterdir()
                if child.is_dir()
            ]

        hint = did_you_mean(typo, candidates)

    raise ValueError(
        f"`--{flag} {value}` was specified which implies config `{path}`, which does not exist.{hint} "
        f"See available {flag} configs in {available}."
    )


def _asset_selector(flag: str) -> Flag:
    """A `--<flag> NAME` selector that resolves the named asset to a config and adds it to +config_paths."""
    dest = flag.replace("-", "_")

    def translate(args: argparse.Namespace) -> list[str]:
        value = getattr(args, dest)
        if not value:
            return []
        path = _asset_config_path(flag, value)
        setattr(args, f"_{dest}_config_path", path)
        return [f"+config_paths=[{path}]"]

    return Flag(
        register=lambda p: p.add_argument(f"--{flag}", metavar="NAME", help=f"Load the named {flag} config."),
        translate_to_hydra=translate,
    )


BENCHMARK = _asset_selector("benchmark")
ENVIRONMENT = _asset_selector("environment")
RESOURCES_SERVER_CONFIG = _asset_selector("resources-server")
AGENT_SERVER = _asset_selector("agent-server")
MODEL_TYPE = _asset_selector("model-type")

# `--search-dir`: extra component-search roots. `main()` folds these into the `NEMO_GYM_EXTRA_ROOTS` env
# var before dispatch (see there), so a single register-only flag suffices for every command — the roots
# reach discovery, the `--<component> NAME` selectors, deep path resolution, and spawned servers alike.
SEARCH_DIR = Flag(
    register=lambda p: p.add_argument(
        "--search-dir",
        action="append",
        metavar="DIR",
        help="Extra root directory to search for components; repeatable.",
    ),
)


def _merge_config_paths(overrides: list[str]) -> list[str]:
    """Coalesce all `+config_paths=[...]` tokens (from --config and asset selectors) into one (Hydra rejects dupes)."""
    prefix = "+config_paths=["
    paths: list[str] = []
    rest: list[str] = []
    for token in overrides:
        if token.startswith(prefix) and token.endswith("]"):
            paths.extend(p for p in token[len(prefix) : -1].split(",") if p)
        else:
            rest.append(token)
    return ([f"+config_paths=[{','.join(paths)}]"] if paths else []) + rest


def _manifest_binding_overrides(args: argparse.Namespace, overrides: list[str]) -> list[str]:
    """Bind one valid manifest adjacent to a selected recipe config."""

    if _override_has_key(overrides, "manifest_path"):
        return overrides

    config_paths = [
        path.strip().strip("\"'")
        for override in overrides
        if override.startswith("+config_paths=[") and override.endswith("]")
        for path in override[len("+config_paths=[") : -1].split(",")
        if path.strip()
    ]
    if not config_paths:
        return overrides

    from nemo_gym.environment_execution import resolve_manifest_for_validation
    from nemo_gym.environment_manifest import ManifestError

    try:
        selected = resolve_manifest_for_validation({"config_paths": config_paths})
    except ManifestError as error:
        args._parser.error(str(error))
    if selected is None:
        return overrides
    manifest_path, _manifest = selected
    return [*overrides, f"+manifest_path={_quote_hydra_string(str(manifest_path))}"]


def _recipe_execution_overrides(
    args: argparse.Namespace, overrides: list[str], *, allow_version: bool = False
) -> list[str]:
    reference = getattr(args, "environment_ref", None)
    rollout_driver = getattr(args, "rollout_driver", None)
    if rollout_driver and _override_mutates_tree(overrides, "rollout_collection_driver"):
        args._parser.error("--rollout-driver cannot be combined with a rollout_collection_driver override.")
    if reference and ("=" in reference or reference.startswith(("+", "~"))):
        overrides = [reference, *overrides]
        args.environment_ref = None
        reference = None
    selected_component_paths = {
        str(path)
        for name in ("resources_server", "agent_server", "model_type")
        if (path := getattr(args, f"_{name}_config_path", None)) is not None
    }
    unclassified_config_paths = [
        path.strip("\"'")
        for override in overrides
        if override.startswith("+config_paths=[") and override.endswith("]")
        for path in override[len("+config_paths=[") : -1].split(",")
        if path
    ]
    has_recipe_target = bool(
        reference
        or getattr(args, "environment", None)
        or getattr(args, "benchmark", None)
        or getattr(args, "config", None)
        or any(path not in selected_component_paths for path in unclassified_config_paths)
    )
    if reference and (getattr(args, "environment", None) or getattr(args, "benchmark", None)):
        args._parser.error("NAME cannot be combined with --environment or --benchmark.")
    if reference:
        overrides = _catalog_reference_overrides(
            args,
            overrides,
            include_reference=True,
            include_unpublished=True,
            allow_version=allow_version,
        )
    else:
        overrides = _manifest_binding_overrides(args, overrides)
    if not has_recipe_target or not _override_has_key(overrides, "manifest_path"):
        if rollout_driver:
            args._parser.error("--rollout-driver requires a manifest-bound environment or benchmark recipe.")
        return overrides

    selections = []
    for role, flag, dest in (
        ("resources_server", "resources-server", "resources_server"),
        ("agent_server", "agent-server", "agent_server"),
        ("model_server", "model-type", "model_type"),
    ):
        value = getattr(args, dest, None)
        if not value:
            continue
        path = getattr(args, f"_{dest}_config_path", None) or _asset_config_path(flag, value)
        selections.append(f"{role}:{_quote_hydra_string(str(Path(path).resolve()))}")
    if rollout_driver:
        overrides = [
            *overrides,
            f"+rollout_collection_driver={_quote_hydra_string(rollout_driver)}",
        ]
        selections.append(f"rollout_driver:{_quote_hydra_string(rollout_driver)}")
    if selections:
        overrides = [*overrides, "+environment_component_swaps={" + ",".join(selections) + "}"]
    return overrides


def _eval_run(args: argparse.Namespace, overrides: list[str]) -> None:
    overrides = _with_cli_override_metadata(_recipe_execution_overrides(args, overrides))
    target = "nemo_gym.cli.eval:collect_rollouts" if args.no_serve else "nemo_gym.cli.eval:e2e_rollout_collection"
    dispatch(target, overrides)


def _eval_reverify(args: argparse.Namespace, overrides: list[str]) -> None:
    dispatch(
        "nemo_gym.cli.eval:reverify_rollouts",
        _with_cli_override_metadata(_recipe_execution_overrides(args, overrides)),
    )


def _env_start(args: argparse.Namespace, overrides: list[str]) -> None:
    dispatch("nemo_gym.cli.env:run", _with_cli_override_metadata(_recipe_execution_overrides(args, overrides)))


def _env_test(args: argparse.Namespace, overrides: list[str]) -> None:
    reference = getattr(args, "environment_ref", None)
    # An optional positional makes argparse consume a leading Hydra ``+key=value``
    # token. Put it back into passthrough overrides to retain the legacy syntax.
    if reference and ("=" in reference or reference.startswith(("+", "~"))):
        overrides = [reference, *overrides]
        args.environment_ref = None
        reference = None
    has_entrypoint = any(override.lstrip("+").split("=", 1)[0] == "entrypoint" for override in overrides)
    has_replay = getattr(args, "replay", None) is not None or any(
        override.lstrip("+").split("=", 1)[0] == "replay_rollouts_path" for override in overrides
    )
    if reference and has_entrypoint:
        args._parser.error("NAME cannot be combined with --resources-server or +entrypoint.")
    if has_replay:
        if not reference:
            replay_path = getattr(args, "replay", None)
            if replay_path is None:
                args._parser.error(
                    "Replay passed as a Hydra override requires an environment NAME; "
                    "use --replay with a trajectory bundle to infer it."
                )
            from nemo_gym import _resolve_under_cwd_or_install
            from nemo_gym.config_types import ConfigError
            from nemo_gym.trajectory_bundle import bundle_path_for, read_trajectory_bundle

            selected = _resolve_under_cwd_or_install(Path(replay_path).expanduser()).resolve()
            bundle_path = selected if selected.name.endswith(".bundle.json") else bundle_path_for(selected)
            try:
                bundle = read_trajectory_bundle(bundle_path)
            except ConfigError as error:
                args._parser.error(
                    f"--replay without NAME requires a valid trajectory bundle at '{bundle_path}': {error}"
                )
            if bundle.environment is None:
                args._parser.error(
                    f"Trajectory bundle '{bundle_path}' has no captured environment; pass NAME explicitly."
                )
            args.environment_ref = bundle.environment.name
            args.catalog_kind = bundle.environment.kind
            reference = args.environment_ref
        if getattr(args, "update_expected", False):
            args._parser.error("--replay and --update-expected are separate workflows and cannot be combined.")
        dispatch("nemo_gym.cli.env:replay_rollouts", _recipe_execution_overrides(args, overrides))
        return
    replay_only_options = ("force", "limit", "concurrency", "output", "failures")
    supplied_replay_options = [
        option for option in replay_only_options if getattr(args, option, None) not in (None, False)
    ]
    if supplied_replay_options:
        args._parser.error(
            "Replay option(s) " + ", ".join(f"--{option}" for option in supplied_replay_options) + " require --replay."
        )
    if reference:
        dispatch("nemo_gym.cli.env:test", _recipe_execution_overrides(args, overrides))
        return
    if getattr(args, "update_expected", False) and not has_entrypoint:
        args._parser.error("--update-expected requires NAME, --resources-server, or +entrypoint.")
    dispatch("nemo_gym.cli.env:test" if has_entrypoint else "nemo_gym.cli.env:test_all", overrides)


def _override_has_key(overrides: list[str], key: str) -> bool:
    return any(override.lstrip("+~").split("=", 1)[0] == key for override in overrides)


def _override_mutates_tree(overrides: list[str], key: str) -> bool:
    for override in overrides:
        path = override.lstrip("+~").split("=", 1)[0]
        if path == key or path.startswith((key + ".", key + "[")):
            return True
    return False


_NON_SCORE_OVERRIDE_ROOTS = frozenset(
    {
        "append",
        "catalog_domain",
        "catalog_facts",
        "catalog_kind",
        "catalog_licensing",
        "catalog_lifecycle",
        "catalog_modality",
        "catalog_profile",
        "catalog_requires",
        "catalog_status",
        "component_name",
        "config_paths",
        "disable_aggregation",
        "dry_run",
        "environment_cli_override_paths",
        "environment_component_swaps",
        "environment_component_provenance",
        "environment_ref",
        "force",
        "json",
        "judge_failed_only",
        "manifest_path",
        "materialized_inputs_jsonl_fpath",
        "materialized_jsonl_fpath",
        "model_call_capture_dir",
        "nemo_gym_log_dir",
        "num_samples_in_parallel",
        "output_jsonl_fpath",
        "overwrite",
        "policy_base_url",
        "publish_dry_run",
        "publish_owner",
        "replay_rollouts_path",
        "resume_from_cache",
        "rollouts_jsonl_fpath",
        "sync_manifest",
        "verbose",
        "wandb_name",
        "wandb_project",
    }
)


def _with_cli_override_metadata(overrides: list[str]) -> list[str]:
    """Attach score-affecting CLI paths without serializing their raw values."""

    from nemo_gym.credential_keys import is_credential_key

    paths: list[str] = []
    retained: list[str] = []
    for override in overrides:
        raw = override.lstrip("+")
        if raw.startswith("~"):
            raw = raw[1:]
        path = raw.split("=", 1)[0]
        root = path.split(".", 1)[0]
        if root == "environment_cli_override_paths":
            continue
        retained.append(override)
        if (
            "=" in raw
            and root not in _NON_SCORE_OVERRIDE_ROOTS
            and not is_credential_key(path.rsplit(".", 1)[-1])
            and path not in paths
        ):
            paths.append(path)
    if paths:
        retained.append(
            "+environment_cli_override_paths=[" + ",".join(_quote_hydra_string(path) for path in paths) + "]"
        )
    return retained


def _env_init(args: argparse.Namespace, overrides: list[str]) -> None:
    """Keep the resources-server scaffold compatible while routing guided init to onboarding."""
    has_guided_init = any(
        _override_has_key(overrides, key)
        for key in (
            "init_name",
            "init_kind",
            "init_profile",
            "init_reuse_verifier",
            "init_version",
            "init_domain",
            "init_description",
            "init_modality",
            "init_licensing",
            "init_authors",
            "init_canonical_split",
        )
    )
    if getattr(args, "resources_server", None) and has_guided_init:
        args._parser.error("--resources-server cannot be combined with guided environment metadata.")
    target = (
        "nemo_gym.cli.onboarding:init_environment" if has_guided_init else "nemo_gym.cli.env:init_resources_server"
    )
    dispatch(target, overrides)


def _catalog_reference_overrides(
    args: argparse.Namespace,
    overrides: list[str],
    *,
    include_reference: bool,
    include_unpublished: bool = False,
    allow_version: bool = False,
) -> list[str]:
    reference = getattr(args, "environment_ref", None)
    if not reference:
        return overrides

    from nemo_gym.cli.onboarding import resolve_catalog_reference

    try:
        entry = resolve_catalog_reference(
            reference,
            getattr(args, "catalog_kind", None),
            include_unpublished=include_unpublished,
            allow_version=allow_version,
        )
    except ValueError as exc:
        args._parser.error(str(exc))

    if _override_has_key(overrides, "manifest_path"):
        args._parser.error("manifest_path is fixed by a catalog environment reference.")
    additions = [f"+config_paths=[{_quote_hydra_string(str(entry.config_path))}]"]
    if entry.manifest_path is not None:
        additions.append(f"+manifest_path={_quote_hydra_string(str(entry.manifest_path))}")
    if include_reference and not _override_has_key(overrides, "environment_ref"):
        additions.append(f"+environment_ref={_quote_hydra_string(reference)}")
    return _merge_config_paths([*additions, *overrides])


def _env_validate(args: argparse.Namespace, overrides: list[str]) -> None:
    resolved = _recipe_execution_overrides(args, overrides, allow_version=True)
    resolved = [override for override in resolved if not override.startswith("+environment_ref=")]
    dispatch("nemo_gym.cli.onboarding:validate_environment", _with_cli_override_metadata(resolved))


def _env_publish(args: argparse.Namespace, overrides: list[str]) -> None:
    allowed_roots = {
        "catalog_kind",
        "environment_ref",
        "json",
        "publish_dry_run",
        "publish_owner",
    }
    unsupported = sorted(
        {
            override.lstrip("+~").split("=", 1)[0].split(".", 1)[0]
            for override in overrides
            if override.lstrip("+~").split("=", 1)[0].split(".", 1)[0] not in allowed_roots
        }
    )
    if unsupported:
        args._parser.error(
            "gym env publish locks the canonical checked-in config and does not accept temporary composition "
            f"overrides: {', '.join(unsupported)}"
        )
    dispatch(
        "nemo_gym.cli.onboarding:publish_environment",
        _with_cli_override_metadata(
            _catalog_reference_overrides(
                args,
                overrides,
                include_reference=True,
                include_unpublished=True,
                allow_version=True,
            )
        ),
    )


def _dataset_upload(args: argparse.Namespace, overrides: list[str]) -> None:
    targets = {
        "hf": "nemo_gym.cli.dataset:upload_jsonl_dataset_to_hf_cli",
        "gitlab": "nemo_gym.cli.dataset:upload_jsonl_dataset_cli",
    }
    dispatch(targets[args.storage], overrides)


def _dataset_download(args: argparse.Namespace, overrides: list[str]) -> None:
    targets = {
        "hf": "nemo_gym.cli.dataset:download_jsonl_dataset_from_hf_cli",
        "gitlab": "nemo_gym.cli.dataset:download_jsonl_dataset_cli",
    }
    dispatch(targets[args.storage], overrides)


# One-line help for each command group, shown in `gym --help`.
GROUPS = {
    "list": "List available catalog entries and components.",
    "dataset": "Manage datasets.",
    "env": "Develop and run environments.",
    "eval": "Run evaluations.",
    "dev": "Contributor helpers.",
}


# NOTE: none of the flags are argparse-required (every value can also be supplied as a Hydra `+key=value` override).
COMMANDS = {
    "list benchmarks": Command(
        target="nemo_gym.cli.eval:list_benchmarks",
        summary="List or inspect available benchmarks.",
        flags=(NAME, JSON, SEARCH_DIR),
    ),
    "list environments": Command(
        target="nemo_gym.cli.env:list_environments",
        summary="List or inspect environments (backward-compatible legacy view).",
        flags=(NAME, JSON, SEARCH_DIR),
    ),
    "list catalog": Command(
        target="nemo_gym.cli.catalog:list_environment_catalog",
        summary="List or filter the manifest registry and legacy runnable-unit union.",
        flags=(CATALOG_NAME, *CATALOG_FILTERS, JSON, SEARCH_DIR),
    ),
    "list components": Command(
        target="nemo_gym.cli.onboarding:list_components",
        summary="List reusable deployable components by declared capability.",
        flags=(
            _quoted_value_flag("provides", "component_provides", "Filter by a capability the component provides."),
            JSON,
            SEARCH_DIR,
        ),
    ),
    "list agents": Command(
        target="nemo_gym.cli.agents:list_agents",
        summary="List or inspect available agent harnesses.",
        flags=(NAME, JSON, SEARCH_DIR),
    ),
    "list models": Command(
        target="nemo_gym.cli.models:list_models",
        summary="List or inspect available model servers.",
        flags=(NAME, JSON, SEARCH_DIR),
    ),
    "list resources-servers": Command(
        target="nemo_gym.cli.resources_servers:list_resources_servers",
        summary="List or inspect available resources servers.",
        flags=(NAME, JSON, SEARCH_DIR),
    ),
    "search": Command(
        target=_search,
        summary="Search benchmarks by default, or select catalog or another component type.",
        flags=(SEARCH_TERMS, JSON, SEARCH_DIR),
    ),
    "dataset upload": Command(
        target=_dataset_upload,
        summary="Upload a prepared dataset to HF (default) or GitLab.",
        flags=(
            STORAGE,
            _value_flag("input", "input_jsonl_fpath", "Local JSONL file to upload.", aliases=("-i",)),
            _value_flag("name", "dataset_name", "Dataset name."),
            # GitLab stores it as `version`, HF as `revision`; emit both and let each backend keep its own.
            Flag(
                register=lambda p: p.add_argument(
                    "--revision", dest="revision", help="Dataset revision (version) to upload."
                ),
                translate_to_hydra=lambda args: (
                    # we set both version and revision because GitLab and HF use different keys
                    # and we extra="ignore" so it's safe to set both
                    [f"+version={args.revision}", f"+revision={args.revision}"] if args.revision is not None else []
                ),
            ),
            _value_flag("split", "split", "Dataset split (HF only)."),
            _bool_flag("create-pr", "create_pr", "Open a pull request instead of committing directly (HF only)."),
        ),
    ),
    "dataset download": Command(
        target=_dataset_download,
        summary="Download a dataset from HF (default) or GitLab.",
        flags=(
            STORAGE,
            _value_flag("repo-id", "repo_id", "HF repo id, e.g. org/dataset (HF only)."),
            _value_flag("name", "dataset_name", "Dataset name (GitLab only)."),
            # NOTE(martas): HF download does not allow to specify revision
            _value_flag("revision", "version", "Dataset version (GitLab only)."),
            _value_flag(
                "artifact", "artifact_fpath", "Remote file to fetch (GitLab: required; HF: optional raw file)."
            ),
            _value_flag("output", "output_fpath", "Local destination file.", aliases=("-o",)),
            _value_flag(
                "output-dir", "output_dirpath", "Local destination directory; needed for all splits (HF only)."
            ),
            _value_flag("split", "split", "Dataset split (HF only)."),
        ),
    ),
    "dataset rm": Command(
        target="nemo_gym.cli.dataset:delete_jsonl_dataset_from_gitlab_cli",
        summary="Delete a dataset from GitLab.",
        flags=(_value_flag("name", "dataset_name", "Name of the dataset to delete."),),
    ),
    "dataset migrate": Command(
        target="nemo_gym.cli.dataset:upload_jsonl_dataset_to_hf_and_delete_gitlab_cli",
        summary="Transfer a dataset from GitLab to HF.",
        flags=(
            _value_flag("input", "input_jsonl_fpath", "Local JSONL file to upload to HF.", aliases=("-i",)),
            _value_flag("name", "dataset_name", "Dataset name."),
            _value_flag("revision", "revision", "Dataset revision (HF)."),
            _value_flag("split", "split", "Dataset split."),
            _bool_flag("create-pr", "create_pr", "Open a pull request instead of committing directly."),
        ),
    ),
    "dataset render": Command(
        target="nemo_gym.cli.dataset:materialize_prompts_cli",
        summary="Generate a dataset preview.",
        flags=(
            _value_flag("input", "input_jsonl_fpath", "Raw input JSONL file.", aliases=("-i",)),
            _value_flag("prompt-config", "prompt_config", "Prompt template YAML to apply."),
            _value_flag("output", "output_jsonl_fpath", "Output JSONL file.", aliases=("-o",)),
            SEARCH_DIR,
        ),
    ),
    "dataset collate": Command(
        target="nemo_gym.cli.dataset:prepare_data",
        summary="Validate and collate the dataset.",
        flags=(
            CONFIG,
            RESOURCES_SERVER_CONFIG,
            MODEL_TYPE,
            SEARCH_DIR,
            _value_flag("mode", "mode", "Data preparation mode.", choices=("train_preparation", "example_validation")),
            _value_flag("output-dir", "output_dirpath", "Output directory for the prepared data."),
            _bool_flag("download", "should_download", "Download source datasets before collating."),
        ),
    ),
    "env init": Command(
        target=_env_init,
        summary="Scaffold a profile-aware environment or benchmark (or a legacy resources server).",
        flags=(
            ENV_INIT_TARGET,
            _quoted_value_flag(
                "profile",
                "init_profile",
                "Select who drives the episode.",
                choices=("stock-loop", "measured-loop", "external-loop", "custom-driver"),
            ),
            _quoted_value_flag("reuse-verifier", "init_reuse_verifier", "Reuse an existing verifier by name."),
            _quoted_value_flag("version", "init_version", "Initial semantic version.", dest="init_version"),
            _quoted_value_flag("domain", "init_domain", "Primary evaluation domain.", choices=_DOMAIN_CHOICES),
            _quoted_value_flag("description", "init_description", "Searchable description."),
            _quoted_value_flag("modality", "init_modality", "Input/output modality."),
            _quoted_value_flag("licensing", "init_licensing", "SPDX identifier or license class."),
            _repeated_quoted_flag("author", "init_authors", "Manifest author; repeatable."),
            _quoted_value_flag(
                "canonical-split",
                "init_canonical_split",
                "Benchmark split used for canonical evaluation.",
            ),
        ),
    ),
    "env resolve": Command(
        target="nemo_gym.cli.env:dump_config",
        summary="Resolve the final config from configs, flags, and overrides.",
        flags=(CONFIG, SEARCH_DIR),
    ),
    "env validate": Command(
        target=_env_validate,
        summary="Resolve and validate an environment, its manifest, components, and datasets without Ray.",
        flags=(
            VALIDATION_REFERENCE,
            CONFIG,
            BENCHMARK,
            ENVIRONMENT,
            RESOURCES_SERVER_CONFIG,
            AGENT_SERVER,
            MODEL_TYPE,
            SEARCH_DIR,
            MODEL,
            MODEL_URL,
            MODEL_API_KEY,
            ROLLOUT_DRIVER,
            REFERENCE_KIND,
            _quoted_value_flag("manifest", "manifest_path", "Explicit manifest.yaml path."),
            _bool_flag("sync", "sync_manifest", "Update the manifest's composition mirror from resolved config."),
            JSON,
        ),
    ),
    "env manifest": Command(
        target="nemo_gym.cli.onboarding:edit_environment_manifests",
        summary="Bulk-edit manifest-authoritative fields over an explicit catalog selection.",
        flags=(
            *MANIFEST_EDIT_FILTERS,
            _repeated_quoted_flag("set", "manifest_set", "Assignment as field.path=YAML_VALUE; repeatable."),
            _bool_flag("dry-run", "dry_run", "Validate and show the selected files without writing."),
            JSON,
            SEARCH_DIR,
        ),
    ),
    "env publish": Command(
        target=_env_publish,
        summary="Validate and publish an environment manifest into the in-repo registry.",
        flags=(
            PUBLISH_REFERENCE,
            REFERENCE_KIND,
            _repeated_quoted_flag("owner", "publish_owner", "CODEOWNER handle; repeatable."),
            _bool_flag("dry-run", "publish_dry_run", "Show the manifest and CODEOWNERS changes without writing."),
            JSON,
            SEARCH_DIR,
        ),
    ),
    "env packages": Command(
        target="nemo_gym.cli.env:pip_list",
        summary="Print pip packages for the selected resources server.",
        flags=(
            RESOURCES_SERVER,
            SEARCH_DIR,
            _bool_flag("outdated", "outdated", "List only outdated packages."),
            Flag(
                register=lambda p: p.add_argument(
                    "--json", action="store_true", help="Output the package list as JSON."
                ),
                translate_to_hydra=lambda args: ["+format=json"] if args.json else [],
            ),
        ),
    ),
    "env test": Command(
        target=_env_test,
        summary="Test a scorer fixture, replay saved trajectories, or run legacy server tests.",
        flags=(
            ENVIRONMENT_REFERENCE,
            REFERENCE_KIND,
            RESOURCES_SERVER,
            _bool_flag(
                "update-expected",
                "update_expected",
                "Regenerate expected statuses and rewards from the in-process scorer.",
            ),
            _quoted_value_flag(
                "replay",
                "replay_rollouts_path",
                "Re-score a trajectory bundle or captured rollout JSONL without rerunning the policy.",
            ),
            _bool_flag(
                "force",
                "force",
                "Allow an explicit verifier replacement or a verifier that declares reverification unsafe.",
            ),
            _value_flag(
                "failures",
                "failure_trajectories",
                "Captured failures to replay (default: latest-replayable).",
                choices=("exclude", "judge-failed", "latest-replayable"),
            ),
            _value_flag("limit", "limit", "Maximum number of captured trajectories to re-score."),
            _value_flag("concurrency", "num_samples_in_parallel", "Maximum concurrent verifier requests."),
            _quoted_value_flag(
                "output",
                "output_jsonl_fpath",
                "Fresh output path (default: RUN_replayed.jsonl); existing files are never overwritten.",
                aliases=("-o",),
            ),
            SEARCH_DIR,
        ),
    ),
    "env start": Command(
        target=_env_start,
        summary="Start the servers for a catalog environment or explicit config.",
        flags=(
            ENVIRONMENT_REFERENCE,
            REFERENCE_KIND,
            CONFIG,
            BENCHMARK,
            ENVIRONMENT,
            RESOURCES_SERVER_CONFIG,
            AGENT_SERVER,
            MODEL_TYPE,
            SEARCH_DIR,
            MODEL,
            MODEL_URL,
            MODEL_API_KEY,
            ROLLOUT_DRIVER,
        ),
    ),
    "env status": Command(target="nemo_gym.cli.env:status", summary="Print the server status.", flags=(JSON,)),
    "eval prepare": Command(
        target="nemo_gym.cli.eval:prepare_benchmark",
        summary="Prepare benchmark data and dump it to disk.",
        flags=(CONFIG, BENCHMARK, SEARCH_DIR),
    ),
    "eval run": Command(
        target=_eval_run,
        summary="Run a catalog environment (or explicit config), collect rollouts, and optionally start servers.",
        flags=(
            ENVIRONMENT_REFERENCE,
            REFERENCE_KIND,
            CONFIG,
            BENCHMARK,
            ENVIRONMENT,
            RESOURCES_SERVER_CONFIG,
            AGENT_SERVER,
            MODEL_TYPE,
            SEARCH_DIR,
            Flag(
                register=lambda p: p.add_argument(
                    "--no-serve",
                    action="store_true",
                    help="Collect against already-running servers instead of starting them.",
                )
            ),
            _bool_flag("resume", "resume_from_cache", "Resume from cached rollouts instead of recollecting."),
            _value_flag("agent", "agent_name", "Agent to collect rollouts with.", aliases=("-a",)),
            _value_flag("input", "input_jsonl_fpath", "Input tasks JSONL file.", aliases=("-i",)),
            _value_flag(
                "output",
                "output_jsonl_fpath",
                "Output rollouts JSONL file (also available as --save-trajectories).",
                aliases=("-o", "--save-trajectories"),
            ),
            _value_flag("limit", "limit", "Maximum number of tasks to run."),
            _value_flag("num-repeats", "num_repeats", "Number of rollouts per task."),
            _value_flag("prompt-config", "prompt_config", "Prompt template YAML to apply."),
            _value_flag("concurrency", "num_samples_in_parallel", "Maximum number of concurrent samples."),
            _value_flag("split", "split", "Dataset split to use (train, validation, or benchmark)."),
            MODEL,
            MODEL_URL,
            MODEL_API_KEY,
            ROLLOUT_DRIVER,
            _value_flag("temperature", "responses_create_params.temperature", "Sampling temperature."),
            _value_flag("top-p", "responses_create_params.top_p", "Nucleus sampling top-p."),
            _value_flag("max-output-tokens", "responses_create_params.max_output_tokens", "Maximum output tokens."),
            _bool_flag(
                "disable-aggregation",
                "disable_aggregation",
                "Skip post-run aggregate-metrics computation. Use with gym eval aggregate for sharded jobs.",
            ),
        ),
    ),
    "eval aggregate": Command(
        target="nemo_gym.cli.eval:aggregate_rollouts",
        summary="Aggregate sharded rollout results.",
        flags=(
            CONFIG,
            _value_flag(
                "input-glob",
                "input_glob",
                "Glob (or comma-separated globs) matching the rollout shards to aggregate.",
                aliases=("-i",),
            ),
            _value_flag(
                "output",
                "output_jsonl_fpath",
                "Path for the merged rollouts and aggregate-metrics file.",
                aliases=("-o",),
            ),
        ),
    ),
    "eval reverify": Command(
        target=_eval_reverify,
        summary="Re-verify existing rollouts to recompute rewards with an updated resources server",
        flags=(
            ENVIRONMENT_REFERENCE,
            REFERENCE_KIND,
            CONFIG,
            BENCHMARK,
            ENVIRONMENT,
            RESOURCES_SERVER_CONFIG,
            MODEL_TYPE,
            SEARCH_DIR,
            _value_flag("inputs", "materialized_inputs_jsonl_fpath", "Materialized inputs JSONL."),
            _value_flag("rollouts", "rollouts_jsonl_fpath", "Rollouts JSONL to re-verify."),
            _value_flag("output", "output_jsonl_fpath", "Output JSONL with recomputed rewards.", aliases=("-o",)),
            _value_flag("concurrency", "num_samples_in_parallel", "Maximum number of concurrent samples."),
            _value_flag("limit", "limit", "Maximum number of examples to re-verify."),
            _bool_flag("force", "force", "Override UNSUPPORTED reverify_mode guard (output prefixed with unsafe_)."),
            _bool_flag(
                "overwrite", "overwrite", "Delete existing output file before writing, instead of raising an error."
            ),
            _bool_flag(
                "resume",
                "resume_from_cache",
                "Resume from a partially-completed output file: skip rows already done and re-verify only the rest.",
            ),
            _bool_flag(
                "disable-aggregation",
                "disable_aggregation",
                "Skip the post-reverification aggregate-metrics computation and file write.",
            ),
            _bool_flag(
                "judge-failed-only",
                "judge_failed_only",
                "Failure-recovery: carry successful rollouts through unchanged and re-verify only the run's "
                "previously judge-failed rollouts (auto-read from <rollouts_stem>_failures.jsonl).",
            ),
            _bool_flag(
                "append",
                "append",
                "With --judge-failed-only: append recovered results to an existing --output (never cleared) "
                "instead of seeding successes into a fresh file; captured bundle artifacts remain immutable.",
            ),
        ),
    ),
    "eval profile": Command(
        target="nemo_gym.cli.eval:reward_profile",
        summary="Compute a reward profile from rollouts.",
        flags=(
            _value_flag(
                "inputs",
                "materialized_inputs_jsonl_fpath",
                "Materialized inputs JSONL fed to rollout collection.",
            ),
            _value_flag("rollouts", "rollouts_jsonl_fpath", "Rollouts JSONL produced by collection."),
        ),
    ),
    "dev test": Command(target="nemo_gym.cli.dev:dev_test", summary="Run NeMo Gym's unit tests."),
}


def _add_leaf(subparsers: argparse._SubParsersAction, name: str, command: Command) -> None:
    leaf = subparsers.add_parser(name, help=command.summary, description=command.summary)
    # `_parser=leaf` so error reporting (and flag "did you mean?" hints) uses this command's own options/prog.
    leaf.set_defaults(_command=command, _parser=leaf)
    leaf.add_argument("-v", "--verbose", action="store_true", help="Set logging level to DEBUG.")
    for flag in command.flags:
        flag.register(leaf)


def build_parser() -> argparse.ArgumentParser:
    # _GymArgumentParser propagates to every subparser (argparse defaults parser_class to type(self)).
    parser = _GymArgumentParser(prog="gym", add_help=True)
    parser.add_argument("--version", action="store_true", help="Show the NeMo Gym version and exit.")
    parser.add_argument("--json", action="store_true", help="With --version, output as JSON.")
    parser.set_defaults(_parser=parser)

    subparsers = parser.add_subparsers()
    groups: dict[str, argparse._SubParsersAction] = {}

    for command_name, command in COMMANDS.items():
        parts = command_name.split()
        if len(parts) == 1:
            _add_leaf(subparsers, parts[0], command)
            continue

        group_name, action_name = parts
        if group_name not in groups:
            group_parser = subparsers.add_parser(
                group_name, help=GROUPS.get(group_name), description=GROUPS.get(group_name)
            )
            group_parser.set_defaults(_parser=group_parser)
            groups[group_name] = group_parser.add_subparsers()
        _add_leaf(groups[group_name], action_name, command)

    return parser


def _handle_pydantic_validation_error(exc, parser: argparse.ArgumentParser) -> None:
    # ckeck if the error is coming from a BaseNeMoGymCLIConfig subclass
    # pydantic sets ValidationError.title to the validated
    # model's name, so we match it against the CLI config classes.
    from nemo_gym.config_types import BaseNeMoGymCLIConfig

    config_names = {BaseNeMoGymCLIConfig.__name__}
    stack = [BaseNeMoGymCLIConfig]
    while stack:
        cls = stack.pop()
        for sub in cls.__subclasses__():
            if sub.__name__ not in config_names:
                config_names.add(sub.__name__)
                stack.append(sub)
    if exc.title not in config_names:
        # if this is not a user's config validation error, raise the original error
        raise

    # For user's config validation, raise a descriptive error message
    missing: list[str] = []
    invalid: list[str] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<config>"
        if error["type"] == "missing":
            missing.append(location)
        else:
            invalid.append(f"{location} ({error['msg']})")

    parts: list[str] = []
    if missing:
        parts.append(
            f"missing required configuration: {', '.join(missing)}. "
            f"Provide each via its flag (see --help) or as a +{missing[0]}=<value> override."
        )
    if invalid:
        parts.append(f"invalid configuration: {'; '.join(invalid)}.")
    parser.error(" ".join(parts) if parts else str(exc))


@contextmanager
def _extra_roots_from_search_dir(search_dirs: list[str] | None):
    """Prepend ``--search-dir`` roots to ``NEMO_GYM_EXTRA_ROOTS`` for the duration of the command, then restore.

    Setting the env var lets the roots reach every resolver (discovery, the --<component> selectors,
    deep config/prompt/rollout resolution) and inherit into spawned server subprocesses.
    The original value is restored (or the var unset) on exit so main() leaves no global side effect.
    (sys.path is augmented with the new roots for plugin ``prepare.py`` imports and left as-is on exit.)
    """
    if not search_dirs:
        yield
        return
    original = os.environ.get(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME)
    value = os.pathsep.join([*search_dirs, *([original] if original else [])])
    os.environ[NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME] = value
    _augment_sys_path()  # re-read env so --search-dir roots are importable (e.g. a benchmark prepare.py)
    logger.debug(f"Set {NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME}={value} from --search-dir")
    try:
        yield
    finally:
        if original is None:
            os.environ.pop(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, None)
            logger.debug(f"Unset {NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME}")
        else:
            os.environ[NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME] = original
            logger.debug(f"Restored {NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME}={original}")


@contextmanager
def _without_internal_config_transport():
    """Keep child-server config transport from becoming root CLI input."""

    inherited = {name: os.environ.pop(name) for name in _INTERNAL_CONFIG_TRANSPORT_ENV_VARS if name in os.environ}
    try:
        yield
    finally:
        os.environ.update(inherited)


def _main() -> None:
    parser = build_parser()
    args, overrides = parser.parse_known_args()

    if getattr(args, "verbose", False):
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    # Hydra overrides never start with "-" so we treat them as unknown flags.
    unknown_flags = [token for token in overrides if token.startswith("-")]
    if unknown_flags:
        error_parser = getattr(args, "_parser", parser)
        known_options = [opt for action in error_parser._actions for opt in action.option_strings]
        hints = "".join(did_you_mean(flag.split("=", 1)[0], known_options) for flag in unknown_flags)
        error_parser.error(f"unrecognized arguments: {' '.join(unknown_flags)}{hints}")

    # set NEMO_GYM_EXTRA_ROOTS from --search-dir for the duration of the command
    with _extra_roots_from_search_dir(getattr(args, "search_dir", None)):
        if args.version:
            dispatch(VERSION_TARGET, ["+json=true", *overrides] if args.json else overrides)
            return

        command = getattr(args, "_command", None)
        if command is None:
            args._parser.print_help()
            sys.exit(1)

        try:
            translated = [token for flag in command.flags for token in flag.translate_to_hydra(args)]
        except ValueError as exc:
            getattr(args, "_parser", parser).error(str(exc))

        # --config and the asset selectors all emit +config_paths; coalesce them into one token.
        overrides = _merge_config_paths(translated + overrides)
        # --verbose flows through the config (as +verbose=true) so it reaches spun-up servers, not just this process.
        if getattr(args, "verbose", False):
            overrides = ["+verbose=true", *overrides]

        # Local import keeps `gym --help` (which returns before this point) free of pydantic's import cost;
        # any real command loads pydantic anyway via its config's model_validate.
        from pydantic import ValidationError

        try:
            if callable(command.target):
                command.target(args, overrides)
            else:
                dispatch(command.target, overrides)
        except ValidationError as exc:
            _handle_pydantic_validation_error(exc, getattr(args, "_parser", parser))


def main() -> None:
    with _without_internal_config_transport():
        _main()

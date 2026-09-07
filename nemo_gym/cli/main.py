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
import json
import logging
import os
import re
import sys
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, _augment_sys_path, component_search_roots
from nemo_gym._config_aliases import LEGACY_ENVIRONMENT_ALIASES, legacy_config_path_alias
from nemo_gym.cli.utils import did_you_mean


logger = logging.getLogger(__name__)

VERSION_TARGET = "nemo_gym.cli.general:version"


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
    name: str,
    hydra_key: str,
    flag_help: str,
    *,
    aliases: tuple[str, ...] = (),
    choices: tuple[str, ...] | None = None,
    quote: bool = False,
) -> Flag:
    """A `--name VALUE` flag that maps to the Hydra override `+<hydra_key>=VALUE` (omitted when unset)."""
    dest = name.replace("-", "_")
    return Flag(
        register=lambda p: p.add_argument(f"--{name}", *aliases, dest=dest, choices=choices, help=flag_help),
        translate_to_hydra=lambda args: (
            [f"+{hydra_key}={json.dumps(getattr(args, dest)) if quote else getattr(args, dest)}"]
            if getattr(args, dest) is not None
            else []
        ),
    )


def _bool_flag(name: str, hydra_key: str, flag_help: str) -> Flag:
    """A `--name` store_true flag that maps to the Hydra override `+<hydra_key>=true` when set."""
    dest = name.replace("-", "_")
    return Flag(
        register=lambda p: p.add_argument(f"--{name}", action="store_true", help=flag_help),
        translate_to_hydra=lambda args: [f"+{hydra_key}=true"] if getattr(args, dest) else [],
    )


def _split_comma_separated(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",")]
    if not all(values):
        raise argparse.ArgumentTypeError("check IDs must be a comma-separated list without empty entries")
    return values


def _csv_list_flag(name: str, hydra_key: str, flag_help: str) -> Flag:
    """A comma-separated CLI value translated to a Hydra list override."""
    dest = name.replace("-", "_")
    return Flag(
        register=lambda p: p.add_argument(
            f"--{name}",
            dest=dest,
            type=_split_comma_separated,
            metavar="CHECK[,CHECK...]",
            help=flag_help,
        ),
        translate_to_hydra=lambda args: (
            [f"+{hydra_key}={json.dumps(getattr(args, dest), separators=(',', ':'))}"]
            if getattr(args, dest) is not None
            else []
        ),
    )


def _comma_list_flag(name: str, hydra_key: str, flag_help: str, *, metavar: str) -> Flag:
    """A `--name "A,B"` flag that maps to the Hydra override `+<hydra_key>=["A","B"]`"""
    dest = name.replace("-", "_")

    def to_hydra(args: argparse.Namespace) -> list[str]:
        raw = getattr(args, dest)
        if raw is None:
            return []
        items = [item.strip() for item in raw.split(",") if item.strip()]
        return [f"+{hydra_key}=[{','.join(json.dumps(item) for item in items)}]"]

    return Flag(
        register=lambda p: p.add_argument(f"--{name}", dest=dest, metavar=metavar, help=flag_help),
        translate_to_hydra=to_hydra,
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


# Shared flag: select a single resources server by name. Reused by `env test`, `env init`, and `env packages`.
RESOURCES_SERVER = Flag(
    register=lambda p: p.add_argument("--resources-server", metavar="NAME", help="Name of the resources server."),
    translate_to_hydra=lambda args: (
        [f"+entrypoint=resources_servers/{args.resources_server}"] if args.resources_server else []
    ),
)

# Shared flag: emit machine-readable JSON instead of human output. Reused by reporting commands (version, list,
# env status). Each command reads the reserved `json` config key ad hoc via
# global_config_dict.get(JSON_OUTPUT_KEY_NAME) (see general.py, eval.py, env.py).
# SUPPRESS so an absent trailing `--json` can't overwrite one given before the subcommand (see build_parser).
JSON = Flag(
    register=lambda p: p.add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="Output as machine-readable JSON."
    ),
    translate_to_hydra=lambda args: ["+json=true"] if getattr(args, "json", False) else [],
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
# (see `_search`). A lone positional is the query, defaulting to the unified environment catalog.
_SEARCHABLE_TYPES = {
    "benchmarks": "nemo_gym.cli.eval:list_benchmarks",
    "environments": "nemo_gym.cli.env:list_environments",
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
            help="Component type to search (default: environments).",
        ),
        p.add_argument(
            "query",
            metavar="QUERY",
            help="Text matched (substring or fuzzy) against a component's name, description, and key metadata.",
        ),
    ),
    translate_to_hydra=lambda args: [f"+query={json.dumps(args.query)}"] if getattr(args, "query", None) else [],
)


def _search(args: argparse.Namespace, overrides: list[str]) -> None:
    """`gym search [<type>] <query>`: dispatch to the chosen type's listing command (default environments),
    which filters itself to the `query` config key already in `overrides`."""
    component_type = getattr(args, "component_type", None)
    if component_type in {"environments", "benchmarks"}:
        kind = component_type.removesuffix("s")
        dispatch("nemo_gym.cli.env:list_environments", [f"+catalog_kind={kind}", *overrides])
        return
    dispatch(_SEARCHABLE_TYPES[component_type or "environments"], overrides)


def _list_benchmarks(_args: argparse.Namespace, overrides: list[str]) -> None:
    """List the benchmark view of the unified workload catalog."""
    dispatch("nemo_gym.cli.env:list_environments", ["+catalog_kind=benchmark", *overrides])


def _parsed_onboarding_name(args: argparse.Namespace) -> str | None:
    value = getattr(args, "onboarding_name", None)
    return value if value and "=" not in value else None


ONBOARDING_NAME = Flag(
    register=lambda p: p.add_argument("onboarding_name", nargs="?", metavar="NAME", help="Workload name."),
    translate_to_hydra=lambda args: (
        [f"+onboarding_name={json.dumps(name)}"]
        if (name := _parsed_onboarding_name(args))
        else ([args.onboarding_name] if getattr(args, "onboarding_name", None) else [])
    ),
)

CATALOG_KIND = _value_flag(
    "kind",
    "catalog_kind",
    "Workload kind.",
    choices=("environment", "benchmark"),
)


def _register_init_target(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resources-server", metavar="NAME", help="Name of the resources server.")
    group.add_argument("--environment", dest="scaffold_environment", metavar="NAME", help="Environment name.")
    group.add_argument("--benchmark", dest="scaffold_benchmark", metavar="NAME", help="Benchmark name.")


def _translate_init_target(args: argparse.Namespace) -> list[str]:
    if args.scaffold_environment:
        return ["+scaffold_kind=environment", f"+scaffold_name={json.dumps(args.scaffold_environment)}"]
    if args.scaffold_benchmark:
        return ["+scaffold_kind=benchmark", f"+scaffold_name={json.dumps(args.scaffold_benchmark)}"]
    return [f"+entrypoint=resources_servers/{args.resources_server}"] if args.resources_server else []


INIT_TARGET = Flag(register=_register_init_target, translate_to_hydra=_translate_init_target)

REWARD_RANGE = Flag(
    register=lambda p: p.add_argument(
        "--reward-range",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        help="Declared inclusive reward range.",
    ),
    translate_to_hydra=lambda args: (
        [f"+reward_range=[{args.reward_range[0]},{args.reward_range[1]}]"] if args.reward_range else []
    ),
)


def _register_reward_direction(parser: argparse.ArgumentParser) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--higher-is-better",
        dest="higher_is_better",
        action="store_const",
        const=True,
        help="Declare that larger rewards are better.",
    )
    group.add_argument(
        "--lower-is-better",
        dest="higher_is_better",
        action="store_const",
        const=False,
        help="Declare that smaller rewards are better.",
    )


REWARD_DIRECTION = Flag(
    register=_register_reward_direction,
    translate_to_hydra=lambda args: (
        [f"+higher_is_better={str(args.higher_is_better).lower()}"] if args.higher_is_better is not None else []
    ),
)


# Asset selector flag -> (parent dir, configs subdir, default config flavor). All accept `name` or `name/flavor`,
# resolving to `<parent>/<server>/[<subdir>/]<flavor>.yaml`. A None default flavor falls back to the server name.
_ASSETS = {
    "benchmark": ("benchmarks", "", "config"),
    "environment": ("environments", "", "config"),
    "resources-server": ("resources_servers", "configs", None),
    "model-type": ("responses_api_models", "configs", None),
    "agent-type": ("responses_api_agents", "configs", None),
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

    if "/" in value and flag in {"benchmark", "environment"}:
        nested_matches: list[Path] = []
        for root in roots:
            candidate = root / parent / value / "config.yaml"
            resolved = candidate.resolve()
            if candidate.is_file() and resolved not in nested_matches:
                nested_matches.append(resolved)
        if nested_matches:
            if len(nested_matches) > 1:
                shadowed = ", ".join(f"`{path}`" for path in nested_matches[1:])
                logger.warning(
                    f"`--{flag} {value}` matches multiple configs; using `{nested_matches[0]}` from the "
                    f"highest-priority root and ignoring {shadowed}. Pass `--config <path>` to select a different one."
                )
            return str(nested_matches[0])

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

    if flag == "environment" and value in LEGACY_ENVIRONMENT_ALIASES:
        canonical = LEGACY_ENVIRONMENT_ALIASES[value]
        resolved = _asset_config_path(flag, canonical)
        logger.warning(f"`--environment {value}` is deprecated; use `--environment {canonical}`.")
        return resolved

    if canonical_path := legacy_config_path_alias(path):
        for root in roots:
            candidate = root / canonical_path
            if candidate.exists():
                logger.warning(f"Config path `{path}` is deprecated; use `{canonical_path}`.")
                return str(candidate.resolve())

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


def _asset_selector(flag: str, *, repeatable: bool = False) -> Flag:
    """A `--<flag> NAME` selector that resolves the named asset to a config and adds it to +config_paths.

    Repeatable selectors accumulate, composing several components the way `--config` does.
    """
    dest = flag.replace("-", "_")

    def translate_to_hydra(args: argparse.Namespace) -> list[str]:
        selected = getattr(args, dest)
        if not selected:
            return []
        names = selected if isinstance(selected, list) else [selected]
        return [f"+config_paths=[{','.join(_asset_config_path(flag, name) for name in names)}]"]

    return Flag(
        register=lambda p: p.add_argument(
            f"--{flag}",
            metavar="NAME",
            action="append" if repeatable else "store",
            help=f"Load the named {flag} config." + (" Repeatable." if repeatable else ""),
        ),
        translate_to_hydra=translate_to_hydra,
    )


# Selecting several environments or benchmarks composes their configs, so these accumulate. A model type
# names the one policy model a run serves, so it does not.
BENCHMARK = _asset_selector("benchmark", repeatable=True)
ENVIRONMENT = _asset_selector("environment", repeatable=True)
RESOURCES_SERVER_CONFIG = _asset_selector("resources-server", repeatable=True)
MODEL_TYPE = _asset_selector("model-type")

# Override for verifier-side agent/model compatibility guards. Offered wherever runtime configs compose.
ALLOW_UNSUPPORTED_PAIRING = _bool_flag(
    "allow-unsupported-pairing",
    "allow_unsupported_pairing",
    "Run even if the environment's resources server does not declare support for the selected agent or model type.",
)

AGENT_TYPE = Flag(
    register=lambda p: p.add_argument(
        "--agent-type",
        metavar="NAME",
        help="Agent (NAME or NAME/FLAVOR) to run the selected environment or benchmark.",
    ),
    translate_to_hydra=lambda args: _translate_agent_type(args),
)


def _translate_agent_type(args: argparse.Namespace) -> list[str]:
    """`--agent-type` picks the harness to compose, which is already fixed once the servers are up."""
    name = getattr(args, "agent_type", None)
    if not name:
        return []
    if getattr(args, "no_serve", False):
        raise ValueError(
            "`--agent-type` chooses which agent runs the task, which is settled once the servers are "
            "running, so it cannot be combined with `--no-serve`. Use `--agent` to name one of them."
        )
    return [f"+config_paths=[{_asset_config_path('agent-type', name)}]"]


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


def _eval_submit(args: argparse.Namespace, overrides: list[str]) -> None:
    from omegaconf import OmegaConf

    from nemo_gym.orchestration.api import SubmitConfig
    from nemo_gym.orchestration.submit import submit

    merged = OmegaConf.merge(
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist([t.lstrip("+") for t in overrides]) if overrides else OmegaConf.create(),
    )
    config = SubmitConfig.model_validate(OmegaConf.to_container(merged, resolve=True))
    submit(config, dry_run=args.dry_run)


def _eval_run(args: argparse.Namespace, overrides: list[str]) -> None:
    target = "nemo_gym.cli.eval:collect_rollouts" if args.no_serve else "nemo_gym.cli.eval:e2e_rollout_collection"
    dispatch(target, overrides)


def _eval_health_check(args: argparse.Namespace, overrides: list[str]) -> None:
    expected_overrides = ["+verbose=true"] if args.verbose else []
    if args.json:
        expected_overrides.append("+json=true")
    if overrides != expected_overrides:
        args._parser.error("health-check does not accept Hydra overrides")
    from nemo_gym.cli.eval import health_check_rollouts

    try:
        health_check_rollouts(
            args.run_dir,
            rollout_file=args.rollout_file,
            workers=args.workers,
            ignored_checks=args.ignore_checks or (),
            json_output=args.json,
        )
    except (FileNotFoundError, ValueError) as exc:
        args._parser.error(str(exc))


def _has_override(overrides: list[str], key: str) -> bool:
    return any(override.lstrip("+").split("=", 1)[0] == key for override in overrides)


def _env_init(args: argparse.Namespace, overrides: list[str]) -> None:
    scaffold_selected = bool(
        args.scaffold_environment or args.scaffold_benchmark or _has_override(overrides, "scaffold_kind")
    )
    legacy_selected = _has_override(overrides, "entrypoint")
    scaffold_options = any(
        _has_override(overrides, key) for key in ("profile", "reuse_verifier", "reward_range", "higher_is_better")
    )
    if scaffold_selected and legacy_selected:
        args._parser.error("select a resources server or a manifest-backed workload, not both")
    if scaffold_options and not scaffold_selected:
        args._parser.error("--profile, --reuse-verifier, and reward options require --environment or --benchmark")
    target = "nemo_gym.cli.env:init_environment" if scaffold_selected else "nemo_gym.cli.env:init_resources_server"
    dispatch(target, overrides)


def _env_test(args: argparse.Namespace, overrides: list[str]) -> None:
    manifest_selected = bool(_parsed_onboarding_name(args) or _has_override(overrides, "onboarding_name"))
    legacy_selected = _has_override(overrides, "entrypoint")
    manifest_options = any(_has_override(overrides, key) for key in ("catalog_kind", "update_expected", "json"))
    if manifest_selected and legacy_selected:
        args._parser.error("select a workload name or --resources-server, not both")
    if args.all and (manifest_selected or legacy_selected):
        args._parser.error("--all tests every resources server, so it cannot be combined with a named target")
    if manifest_options and not manifest_selected:
        args._parser.error("--kind, --update-expected, and --json require a workload name")
    if manifest_selected:
        dispatch("nemo_gym.cli.env:test_environment_manifest", overrides)
        return

    # Run a single server's tests if +entrypoint was passed. No need to check for
    # --resources-server because it is translated to +entrypoint in the flag definition.
    if _has_override(overrides, "entrypoint"):
        dispatch("nemo_gym.cli.env:test", overrides)
        return

    if not args.all:
        args._parser.error(
            "requires a target: a workload name, --resources-server NAME, or --all to test every resources server."
        )
    dispatch("nemo_gym.cli.env:test_all", overrides)


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
    "list": "List available components (benchmarks, environments, agents, models, resources-servers).",
    "dataset": "Manage datasets.",
    "env": "Develop and run environments.",
    "eval": "Run evaluations.",
    "dev": "Contributor helpers.",
}


# NOTE: none of the flags are argparse-required (every value can also be supplied as a Hydra `+key=value` override).
COMMANDS = {
    "list benchmarks": Command(
        target=_list_benchmarks,
        summary="List or inspect available benchmarks.",
        flags=(NAME, JSON, SEARCH_DIR),
    ),
    "list environments": Command(
        target="nemo_gym.cli.env:list_environments",
        summary="List or inspect the environment and benchmark catalog.",
        flags=(
            NAME,
            _value_flag("domain", "domain", "Filter by domain."),
            CATALOG_KIND,
            _value_flag("modality", "modality", "Filter by modality.", quote=True),
            _value_flag("licensing", "licensing", "Filter by licensing.", quote=True),
            _value_flag(
                "status",
                "status",
                "Filter by validation status.",
                choices=("experimental", "no-manifest"),
            ),
            _value_flag("lifecycle", "lifecycle", "Filter by lifecycle.", choices=("active", "deprecated")),
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
        summary="Search a component type (default environments) by name; like `list` filtered to a query.",
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
        summary="Scaffold a resources server, environment, or benchmark.",
        flags=(
            INIT_TARGET,
            _value_flag(
                "profile",
                "profile",
                "Integration profile for a manifest-backed scaffold.",
                choices=(
                    "custom-gym-verifier",
                    "custom-gym-agent-loop",
                    "external-agent-loop",
                    "external-rollout-driver",
                ),
            ),
            _value_flag("reuse-verifier", "reuse_verifier", "Existing verifier to reuse.", quote=True),
            REWARD_RANGE,
            REWARD_DIRECTION,
        ),
    ),
    "env resolve": Command(
        target="nemo_gym.cli.env:dump_config",
        summary="Resolve the final config from configs, flags, and overrides.",
        flags=(CONFIG, SEARCH_DIR),
    ),
    "env schema": Command(
        target="nemo_gym.cli.env:show_schema",
        summary="Print a resources server's task-data schema as JSON Schema.",
        flags=(RESOURCES_SERVER, SEARCH_DIR),
    ),
    "env validate": Command(
        target="nemo_gym.cli.env:validate",
        summary="Validate a manifest-backed workload or config without starting services.",
        flags=(
            ONBOARDING_NAME,
            CATALOG_KIND,
            _value_flag("manifest", "manifest_path", "Manifest path to validate.", quote=True),
            _bool_flag("sync", "sync", "Synchronize composition mirrors after validation."),
            _bool_flag("strict", "strict", "Fail instead of warning when model values are left undefined."),
            JSON,
            CONFIG,
            BENCHMARK,
            ENVIRONMENT,
            RESOURCES_SERVER_CONFIG,
            MODEL_TYPE,
            AGENT_TYPE,
            ALLOW_UNSUPPORTED_PAIRING,
            SEARCH_DIR,
            MODEL,
            MODEL_URL,
            MODEL_API_KEY,
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
                    "--json", action="store_true", default=argparse.SUPPRESS, help="Output the package list as JSON."
                ),
                translate_to_hydra=lambda args: ["+format=json"] if getattr(args, "json", False) else [],
            ),
        ),
    ),
    "env test": Command(
        target=_env_test,
        summary="Test a manifest-backed workload or resources server.",
        flags=(
            ONBOARDING_NAME,
            CATALOG_KIND,
            _bool_flag("update-expected", "update_expected", "Update verifier fixture expectations."),
            JSON,
            RESOURCES_SERVER,
            SEARCH_DIR,
            Flag(
                register=lambda p: p.add_argument(
                    "--all",
                    action="store_true",
                    help="Test every resources server. Slow: builds one venv per server.",
                )
            ),
        ),
    ),
    "env publish": Command(
        target="nemo_gym.cli.env:publish_environment_manifest",
        summary="Run local publication checks and confirm catalog discovery.",
        flags=(ONBOARDING_NAME, CATALOG_KIND, JSON, SEARCH_DIR),
    ),
    "env start": Command(
        target="nemo_gym.cli.env:run",
        summary="Start the servers.",
        flags=(
            CONFIG,
            BENCHMARK,
            ENVIRONMENT,
            RESOURCES_SERVER_CONFIG,
            MODEL_TYPE,
            AGENT_TYPE,
            ALLOW_UNSUPPORTED_PAIRING,
            SEARCH_DIR,
            MODEL,
            MODEL_URL,
            MODEL_API_KEY,
        ),
    ),
    "env prefetch": Command(
        target="nemo_gym.cli.env:prefetch",
        summary="Pre-warm per-server venvs without starting servers.",
        flags=(
            CONFIG,
            BENCHMARK,
            ENVIRONMENT,
            RESOURCES_SERVER_CONFIG,
            MODEL_TYPE,
            AGENT_TYPE,
            ALLOW_UNSUPPORTED_PAIRING,
            SEARCH_DIR,
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
        summary="Collate data, start servers, and collect rollouts.",
        flags=(
            CONFIG,
            BENCHMARK,
            ENVIRONMENT,
            RESOURCES_SERVER_CONFIG,
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
            AGENT_TYPE,
            ALLOW_UNSUPPORTED_PAIRING,
            _value_flag("agent", "agent_name", "Agent to collect rollouts with.", aliases=("-a",)),
            _value_flag("input", "input_jsonl_fpath", "Input tasks JSONL file.", aliases=("-i",)),
            _value_flag("output", "output_jsonl_fpath", "Output rollouts JSONL file.", aliases=("-o",)),
            _value_flag("limit", "limit", "Maximum number of tasks to run."),
            _value_flag("num-repeats", "num_repeats", "Number of rollouts per task."),
            _value_flag("prompt-config", "prompt_config", "Prompt template YAML to apply."),
            _value_flag("concurrency", "num_samples_in_parallel", "Maximum number of concurrent samples."),
            _value_flag("split", "split", "Dataset split to use (train, validation, or benchmark)."),
            MODEL,
            MODEL_URL,
            MODEL_API_KEY,
            _value_flag("temperature", "responses_create_params.temperature", "Sampling temperature."),
            _value_flag("top-p", "responses_create_params.top_p", "Nucleus sampling top-p."),
            _value_flag("max-output-tokens", "responses_create_params.max_output_tokens", "Maximum output tokens."),
            _bool_flag(
                "disable-aggregation",
                "disable_aggregation",
                "Skip post-run aggregate-metrics computation. Use with gym eval aggregate for sharded jobs.",
            ),
            _bool_flag("no-health-check", "disable_health_check", "Skip post-run rollout health checks."),
            _value_flag("health-check-workers", "health_check_workers", "Number of rollout-health worker processes."),
            _csv_list_flag(
                "health-check-ignore",
                "health_check_ignored_checks",
                "Comma-separated rollout-health check IDs to exclude from verdict derivation.",
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
            _bool_flag("no-health-check", "disable_health_check", "Skip post-aggregation rollout health checks."),
            _value_flag("health-check-workers", "health_check_workers", "Number of rollout-health worker processes."),
            _csv_list_flag(
                "health-check-ignore",
                "health_check_ignored_checks",
                "Comma-separated rollout-health check IDs to exclude from verdict derivation.",
            ),
        ),
    ),
    "eval health-check": Command(
        target=_eval_health_check,
        summary="Verify rollout quality for an existing run directory.",
        flags=(
            Flag(register=lambda p: p.add_argument("run_dir", metavar="RUN_DIR")),
            Flag(
                register=lambda p: p.add_argument(
                    "--rollouts-file",
                    dest="rollout_file",
                    type=Path,
                    metavar="PATH",
                    help="Rollout JSONL path; relative paths resolve under RUN_DIR (default: rollouts.jsonl).",
                )
            ),
            Flag(register=lambda p: p.add_argument("--workers", type=int, help="Number of worker processes.")),
            Flag(
                register=lambda p: p.add_argument(
                    "--ignore-checks",
                    "--ignore",
                    type=_split_comma_separated,
                    metavar="CHECK[,CHECK...]",
                    help="Comma-separated check IDs to exclude from verdict derivation.",
                )
            ),
            JSON,
        ),
    ),
    "eval reverify": Command(
        target="nemo_gym.cli.eval:reverify_rollouts",
        summary="Re-verify existing rollouts to recompute rewards with an updated resources server",
        flags=(
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
                "instead of seeding successes into a fresh file.",
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
    "eval submit": Command(
        target=_eval_submit,
        summary="Submit a job.",
        flags=(
            Flag(
                register=lambda p: p.add_argument(
                    "--config", "-c", required=True, metavar="PATH", help="Submit config YAML file."
                ),
            ),
            Flag(
                register=lambda p: p.add_argument(
                    "--dry-run", action="store_true", help="Print generated job scripts without submitting."
                ),
            ),
        ),
    ),
    "eval compare": Command(
        target="nemo_gym.cli.eval:compare",
        summary="Compare a baseline eval run against candidate runs.",
        flags=(
            _value_flag(
                "baseline",
                "baseline_rollouts_jsonl_fpath",
                "Baseline run's rollouts JSONL (its *_aggregate_metrics.json sibling is what gets read today).",
                quote=True,
            ),
            _comma_list_flag(
                "candidates",
                "candidate_rollouts_jsonl_fpaths",
                "Candidate run's rollouts JSONL. Comma-separated list; one candidate is supported today.",
                metavar="PATH[,PATH...]",
            ),
            _value_flag(
                "baseline-agg-metrics",
                "baseline_aggregate_metrics_fpath",
                "Baseline's aggregate-metrics JSON, when it is not the sibling of --baseline.",
                quote=True,
            ),
            _comma_list_flag(
                "candidates-agg-metrics",
                "candidate_aggregate_metrics_fpaths",
                "Candidates' aggregate-metrics JSON, in --candidates order, when not siblings of --candidates.",
                metavar="PATH[,PATH...]",
            ),
            _value_flag("agent", "agent_name", "Agent to compare on both sides (default: all shared agents)."),
            _value_flag("baseline-agent", "baseline_agent_name", "Agent to read from the baseline's metrics."),
            _comma_list_flag(
                "candidate-agents",
                "candidate_agent_names",
                "Agent to read from each candidate's metrics, in --candidates order.",
                metavar="NAME[,NAME...]",
            ),
            _value_flag(
                "output-dir",
                "output_dirpath",
                "Where to write the report (default: the candidate run's own directory).",
                aliases=("-o",),
                quote=True,
            ),
            _value_flag(
                "report-format",
                "report_format",
                "Report artifacts to write (default: both).",
                choices=("md", "json", "both"),
            ),
        ),
    ),
    "dev test": Command(target="nemo_gym.cli.dev:dev_test", summary="Run NeMo Gym's unit tests."),
}


def _accepts_json(parser: argparse.ArgumentParser) -> bool:
    return any("--json" in action.option_strings for action in parser._actions)


def _add_leaf(subparsers: argparse._SubParsersAction, name: str, command: Command) -> None:
    leaf = subparsers.add_parser(name, help=command.summary, description=command.summary)
    # `_parser=leaf` so error reporting (and flag "did you mean?" hints) uses this command's own options/prog.
    leaf.set_defaults(_command=command, _parser=leaf)
    # SUPPRESS so an absent trailing `-v` can't overwrite one given before the subcommand (see build_parser).
    leaf.add_argument(
        "-v", "--verbose", action="store_true", default=argparse.SUPPRESS, help="Set logging level to DEBUG."
    )
    for flag in command.flags:
        flag.register(leaf)


def build_parser() -> argparse.ArgumentParser:
    # _GymArgumentParser propagates to every subparser (argparse defaults parser_class to type(self)).
    parser = _GymArgumentParser(prog="gym", add_help=True)
    parser.add_argument("--version", action="store_true", help="Show the NeMo Gym version and exit.")
    # Also registered on every command that emits JSON, so `--json` is accepted before or after the subcommand.
    parser.add_argument(
        "--json", action="store_true", help="Output as machine-readable JSON (--version and reporting commands)."
    )
    # Also registered on every leaf, so `-v` is accepted before or after the subcommand.
    parser.add_argument("-v", "--verbose", action="store_true", help="Set logging level to DEBUG.")
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


def main() -> None:
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
        # A flag rejected for its position (not for being unknown) is still in known_options, so exclude it
        # from its own candidate set — otherwise it matches itself and is suggested as its own correction.
        hints = "".join(
            did_you_mean(name, [opt for opt in known_options if opt != name])
            for name in (flag.split("=", 1)[0] for flag in unknown_flags)
        )
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

        # A pre-subcommand `--json` reaches commands that emit no JSON, where it would otherwise be dropped
        # without a word. Reject it instead of pretending it applied.
        if getattr(args, "json", False) and not _accepts_json(args._parser):
            args._parser.error(
                "--json is not supported by this command; it applies to --version and to reporting commands "
                "such as `gym list benchmarks`, `gym search`, and `gym env status`"
            )

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

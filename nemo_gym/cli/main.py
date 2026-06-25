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
import difflib
import importlib
import re
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from nemo_gym import WORKING_DIR


VERSION_TARGET = "nemo_gym.cli.general:version"


def _did_you_mean(value: str, candidates: Iterable[str]) -> str:
    """A ` Did you mean \\`X\\`?` fragment for the closest candidate to `value`, or `""` if none is close enough."""
    matches = difflib.get_close_matches(value, list(candidates), n=1)
    return f" Did you mean `{matches[0]}`?" if matches else ""


class _GymArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that appends a difflib "did you mean?" hint to invalid-choice errors.

    Covers mistyped commands/groups and bad --flag choices (e.g. --storage), since argparse validates all of them
    as choices against the registry baked into the parser.
    """

    def error(self, message: str) -> None:
        match = re.search(r"invalid choice: '([^']+)' \(choose from (.+)\)", message)
        if match:
            typo, choices = match.group(1), re.findall(r"'([^']+)'", match.group(2))
            message += _did_you_mean(typo, choices)
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
JSON = _bool_flag("json", "json", "Output as JSON for programmatic use.")

# Positional search query for `gym search`; surfaced to the listing command as the `query` config key.
QUERY = Flag(
    register=lambda p: p.add_argument("query", metavar="QUERY", help="Substring to match against component names."),
    translate_to_hydra=lambda args: [f"+query={args.query}"] if getattr(args, "query", None) else [],
)


# Asset selector flag -> (parent dir, configs subdir, default config flavor). All accept `name` or `name/flavor`,
# resolving to `<parent>/<server>/[<subdir>/]<flavor>.yaml`. A None default flavor falls back to the server name.
_ASSETS = {
    "benchmark": ("benchmarks", "", "config"),
    "environment": ("environments", "", "config"),
    "resources-server": ("resources_servers", "configs", None),
    "model-type": ("responses_api_models", "configs", None),
}


def _asset_config_path(flag: str, value: str, search_dirs: tuple[str, ...] = ()) -> str:
    """Map a named asset (`name` or `name/flavor`) to its config path.

    Searches WORKING_DIR (built-ins) first, then any user-registered --search-dir roots.
    """
    parent, subdir, default_flavor = _ASSETS[flag]
    server_name, _, config_flavor = value.partition("/")
    config_flavor = config_flavor or default_flavor or server_name
    config_dir = f"{parent}/{server_name}/{subdir}".rstrip("/")
    path = f"{config_dir}/{config_flavor}.yaml"

    # Match in WORKING_DIR (built-ins) and every --search-dir root; dedupe roots that resolve to the same file.
    roots = [WORKING_DIR, *(Path(d) for d in search_dirs)]
    matches: list[Path] = []

    for root in roots:
        candidate = root / path
        if candidate.exists():
            matches.append(candidate.resolve())

    if len(matches) > 1:
        matches_str = ", ".join(f"`{m}`" for m in matches)
        raise ValueError(
            f"`--{flag} {value}` is ambiguous: it matches multiple configs ({matches_str}). "
            f"Pass the intended config directly with `--config <path>` instead."
        )
    if matches:
        return str(matches[0])

    # No match: suggest the closest real name across all roots (a config flavor when the server exists, else a
    # server name) and report the full paths that were searched.
    available = ", ".join(set(f"`{(root / config_dir).resolve()}`" for root in roots if (root / config_dir).is_dir()))
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

    raise ValueError(
        f"`--{flag} {value}` was specified which implies config `{path}`, which does not exist.{_did_you_mean(typo, candidates)} "
        f"See available {flag} configs in {available}."
    )


def _asset_selector(flag: str) -> Flag:
    """A `--<flag> NAME` selector that resolves the named asset to a config and adds it to +config_paths."""
    dest = flag.replace("-", "_")
    return Flag(
        register=lambda p: p.add_argument(f"--{flag}", metavar="NAME", help=f"Load the named {flag} config."),
        translate_to_hydra=lambda args: (
            [
                f"+config_paths=[{_asset_config_path(flag, getattr(args, dest), tuple(getattr(args, 'search_dir', None) or ()))}]"
            ]
            if getattr(args, dest)
            else []
        ),
    )


BENCHMARK = _asset_selector("benchmark")
ENVIRONMENT = _asset_selector("environment")
RESOURCES_SERVER_CONFIG = _asset_selector("resources-server")
MODEL_TYPE = _asset_selector("model-type")

# Shared flag: register extra root dirs to search for named components. Consumed by the asset selectors above
# (not emitted as a Hydra override). Reused by every command that accepts a --<component type> NAME selector.
SEARCH_DIR = Flag(
    register=lambda p: p.add_argument(
        "--search-dir",
        action="append",
        metavar="DIR",
        help="Extra root directory to search for named components; repeatable.",
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


def _eval_run(args: argparse.Namespace, overrides: list[str]) -> None:
    target = "nemo_gym.cli.eval:collect_rollouts" if args.no_serve else "nemo_gym.cli.eval:e2e_rollout_collection"
    dispatch(target, overrides)


def _env_test(args: argparse.Namespace, overrides: list[str]) -> None:
    # Run a single server's tests if +entrypoint was passed. No need to check for
    # --resources-server because it is translated to +entrypoint in the flag definition.

    has_entrypoint = any(override.lstrip("+").split("=", 1)[0] == "entrypoint" for override in overrides)
    dispatch("nemo_gym.cli.env:test" if has_entrypoint else "nemo_gym.cli.env:test_all", overrides)


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
    "list": "List available components (benchmarks, agents, environments).",
    "dataset": "Manage datasets.",
    "env": "Develop and run environments.",
    "eval": "Run evaluations.",
    "dev": "Contributor helpers.",
}

COMMANDS = {
    "list benchmarks": Command(
        target="nemo_gym.cli.eval:list_benchmarks", summary="List available benchmarks.", flags=(JSON,)
    ),
    "list environments": Command(
        target="nemo_gym.cli.env:list_environments", summary="List available environments by name.", flags=(JSON,)
    ),
    "list agents": Command(
        target="nemo_gym.cli.agents:list_agents",
        summary="List agent harnesses and how each composes (Pattern A vs self-contained B).",
        flags=(JSON,),
    ),
    "search": Command(
        target="nemo_gym.cli.eval:list_benchmarks",
        summary="Search available components (currently benchmarks) by name; like `list` filtered to a query.",
        flags=(QUERY, JSON),
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
        ),
    ),
    "dataset collate": Command(
        target="nemo_gym.cli.dataset:prepare_data",
        summary="Validate and collate the dataset.",
        flags=(
            CONFIG,
            RESOURCES_SERVER_CONFIG,
            SEARCH_DIR,
            _value_flag("mode", "mode", "Data preparation mode.", choices=("train_preparation", "example_validation")),
            _value_flag("output-dir", "output_dirpath", "Output directory for the prepared data."),
            _bool_flag("download", "should_download", "Download source datasets before collating."),
        ),
    ),
    "env init": Command(
        target="nemo_gym.cli.env:init_resources_server",
        summary="Scaffold config for a new server, benchmark, or agent.",
        flags=(RESOURCES_SERVER,),
    ),
    "env resolve": Command(
        target="nemo_gym.cli.env:dump_config",
        summary="Resolve the final config from configs, flags, and overrides.",
        flags=(CONFIG,),
    ),
    "env validate": Command(
        target="nemo_gym.cli.env:validate",
        summary="Validate a config (paths, cross-refs, ??? values, servers) fast — no Ray, no servers.",
        flags=(
            CONFIG,
            BENCHMARK,
            ENVIRONMENT,
            RESOURCES_SERVER_CONFIG,
            MODEL_TYPE,
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
        summary="Test the resources server(s); runs all if no resources server is given.",
        flags=(RESOURCES_SERVER,),
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
            SEARCH_DIR,
            MODEL,
            MODEL_URL,
            MODEL_API_KEY,
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
    "eval profile": Command(
        target="nemo_gym.cli.eval:reward_profile",
        summary="Compute a reward profile from rollouts.",
        flags=(
            _value_flag(
                "inputs", "materialized_inputs_jsonl_fpath", "Materialized inputs JSONL fed to rollout collection."
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


def main() -> None:
    parser = build_parser()
    args, overrides = parser.parse_known_args()

    # Hydra overrides never start with "-" so we treat them as unknown flags.
    unknown_flags = [token for token in overrides if token.startswith("-")]
    if unknown_flags:
        error_parser = getattr(args, "_parser", parser)
        known_options = [opt for action in error_parser._actions for opt in action.option_strings]
        hints = "".join(_did_you_mean(flag.split("=", 1)[0], known_options) for flag in unknown_flags)
        error_parser.error(f"unrecognized arguments: {' '.join(unknown_flags)}{hints}")

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
        sys.stderr.write(f"{getattr(args, '_parser', parser).prog}: error: {exc}\n")
        sys.exit(2)

    # --config and the asset selectors all emit +config_paths; coalesce them into one token.
    overrides = _merge_config_paths(translated + overrides)
    # --verbose flows through the config (as +verbose=true) so it reaches spun-up servers, not just this process.
    if getattr(args, "verbose", False):
        overrides = ["+verbose=true", *overrides]
    if callable(command.target):
        command.target(args, overrides)
    else:
        dispatch(command.target, overrides)

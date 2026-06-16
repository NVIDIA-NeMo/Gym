# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
from collections.abc import Callable
from dataclasses import dataclass, field


VERSION_TARGET = "nemo_gym.cli.general:version"


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

# Command-specific flags read by the corresponding target callable below.
NO_SERVE = Flag(
    register=lambda p: p.add_argument(
        "--no-serve",
        action="store_true",
        help="Collect against already-running servers instead of starting them.",
    )
)

STORAGE = Flag(
    register=lambda p: p.add_argument(
        "--storage", choices=("hf", "gitlab"), default="hf", help="Storage backend (default: hf)."
    )
)

ENTRYPOINT = Flag(
    register=lambda p: p.add_argument(
        "--resource_server", metavar="NAME", help="Name of the resource server to test. Tests all servers if omitted."
    ),
    translate_to_hydra=lambda args: (
        [f"+entrypoint=resources_servers/{args.resource_server}"] if args.resource_server else []
    ),
)


def _eval_run(args: argparse.Namespace, overrides: list[str]) -> None:
    target = "nemo_gym.cli.eval:collect_rollouts" if args.no_serve else "nemo_gym.cli.eval:e2e_rollout_collection"
    dispatch(target, overrides)


def _env_test(args: argparse.Namespace, overrides: list[str]) -> None:
    # Run a single server's tests if one was named (--resource_server is translated to +entrypoint above)
    # or an +entrypoint override was passed directly; otherwise run the whole suite.
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
    "list": "List available components. As of now, only benchmarks are available.",
    "dataset": "Manage datasets.",
    "env": "Develop and run environments.",
    "eval": "Run evaluations.",
    "dev": "Contributor helpers.",
}

COMMANDS = {
    "list benchmarks": Command(target="nemo_gym.cli.eval:list_benchmarks", summary="List available benchmarks."),
    "dataset upload": Command(
        target=_dataset_upload,
        summary="Upload a prepared dataset to HF (default) or GitLab.",
        flags=(STORAGE,),
    ),
    "dataset download": Command(
        target=_dataset_download,
        summary="Download a dataset from HF (default) or GitLab.",
        flags=(STORAGE,),
    ),
    "dataset rm": Command(
        target="nemo_gym.cli.dataset:delete_jsonl_dataset_from_gitlab_cli",
        summary="Delete a dataset from GitLab.",
    ),
    "dataset migrate": Command(
        target="nemo_gym.cli.dataset:upload_jsonl_dataset_to_hf_and_delete_gitlab_cli",
        summary="Transfer a dataset from GitLab to HF.",
    ),
    "dataset render": Command(
        target="nemo_gym.cli.dataset:materialize_prompts_cli",
        summary="Generate a dataset preview.",
    ),
    "dataset collate": Command(
        target="nemo_gym.cli.dataset:prepare_data",
        summary="Validate and collate the dataset.",
        flags=(CONFIG,),
    ),
    "env init": Command(
        target="nemo_gym.cli.env:init_resources_server",
        summary="Scaffold config for a new server, benchmark, or agent.",
    ),
    "env resolve": Command(
        target="nemo_gym.cli.env:dump_config",
        summary="Resolve the final config from configs, flags, and overrides.",
        flags=(CONFIG,),
    ),
    "env packages": Command(
        target="nemo_gym.cli.env:pip_list",
        summary="Print pip packages for the selected resource server.",
    ),
    "env test": Command(
        target=_env_test,
        summary="Test the resource server(s); runs all if no resource server is given.",
        flags=(ENTRYPOINT,),
    ),
    "env run": Command(target="nemo_gym.cli.env:run", summary="Start the servers.", flags=(CONFIG,)),
    "env status": Command(target="nemo_gym.cli.env:status", summary="Print the server status."),
    "eval prepare": Command(
        target="nemo_gym.cli.eval:prepare_benchmark",
        summary="Prepare benchmark data and dump it to disk.",
        flags=(CONFIG,),
    ),
    "eval run": Command(
        target=_eval_run,
        summary="Collate data, start servers, and collect rollouts.",
        flags=(CONFIG, NO_SERVE),
    ),
    "eval aggregate": Command(
        target="nemo_gym.cli.eval:aggregate_rollouts",
        summary="Aggregate sharded rollout results.",
        flags=(CONFIG,),
    ),
    "eval profile": Command(
        target="nemo_gym.cli.eval:reward_profile",
        summary="Compute a reward profile from rollouts.",
    ),
    "dev test": Command(target="nemo_gym.cli.dev:dev_test", summary="Run NeMo Gym's unit tests."),
}


def _add_leaf(subparsers: argparse._SubParsersAction, name: str, command: Command) -> None:
    leaf = subparsers.add_parser(name, help=command.summary, description=command.summary)
    leaf.set_defaults(_command=command)
    for flag in command.flags:
        flag.register(leaf)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gym", add_help=True)
    parser.add_argument("--version", action="store_true", help="Show the NeMo Gym version and exit.")
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
        getattr(args, "_parser", parser).error(f"unrecognized arguments: {' '.join(unknown_flags)}")

    if args.version:
        dispatch(VERSION_TARGET, overrides)
        return

    command = getattr(args, "_command", None)
    if command is None:
        args._parser.print_help()
        sys.exit(1)

    overrides = [token for flag in command.flags for token in flag.translate_to_hydra(args)] + overrides
    if callable(command.target):
        command.target(args, overrides)
    else:
        dispatch(command.target, overrides)

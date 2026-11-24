# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Repository check: a taskset names the catalog entry its tasks belong to, as `<catalog name>[:<label>]`.

Results carry the catalog name through `task_id.taskset`. A config that a manifest points at may only declare tasksets
named after that manifest. A run preset (a config with no manifest) is held to the manifest whose config sits beside
it, or else to the nearest manifest above it: no config names its catalog entry yet, so its place is all there is.
The optional label is run-local, such as a split (`:train`) or a harness route (`:opencode`). Configs outside the
catalog trees, such as a resources server's example config, name no catalog entry, so only the renamed names are
refused there.
"""

import re
import subprocess
from functools import cache
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

from nemo_gym.config.config_types import ConfigPathNotFoundError
from nemo_gym.config.global_config import GlobalConfigDictParser
from nemo_gym.environment.catalog_renames import RENAMED_CATALOG_NAMES, RENAMED_TASKSETS
from nemo_gym.environment.discovery import iter_server_configs
from nemo_gym.environment.registry import discover_environment_catalog


REPO_ROOT = Path(__file__).resolve().parents[2]
CATALOG_TREES = ("benchmarks", "environments")
ROUTES_KEY = "environment_server_routes"
# A label is one name with no `/` or `:`, so `<catalog name>:<label>` splits at the only `:`.
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def taskset_problem(taskset: str, catalog_name: str) -> str | None:
    """Why ``taskset`` does not name the catalog entry ``catalog_name``, or None when it does."""
    if taskset == catalog_name:
        return None
    if not taskset.startswith(f"{catalog_name}:"):
        return f"must be `{catalog_name}` or `{catalog_name}:<label>`"
    if not LABEL.fullmatch(taskset.removeprefix(f"{catalog_name}:")):
        return "its label must be one name of letters, digits, `_`, `.` or `-`"
    return None


def renamed_problem(taskset: str) -> str | None:
    """Why ``taskset`` is a renamed name, or None when it is not."""
    if taskset in RENAMED_TASKSETS:
        return f"was renamed to {', '.join(f'`{new}`' for new in RENAMED_TASKSETS[taskset])}"
    catalog_name = taskset.partition(":")[0]
    if catalog_name in RENAMED_CATALOG_NAMES:
        return f"names `{catalog_name}`, which was renamed to `{RENAMED_CATALOG_NAMES[catalog_name]}`"
    return None


def _tracked_yaml() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "*.yaml", "*.yml"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    return [REPO_ROOT / line for line in listed.stdout.splitlines()]


def _load(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError):
        return None


def _route_keys(config: object) -> set[str]:
    routes = config.get(ROUTES_KEY) if isinstance(config, (dict, DictConfig)) else None
    return {str(key) for key in routes} if isinstance(routes, (dict, DictConfig)) else set()


def _dataset_tasksets(config: object) -> set[str]:
    tasksets = set()
    for _group, _name, server in iter_server_configs(config):
        datasets = server.get("datasets")
        for dataset in datasets if isinstance(datasets, (list, ListConfig)) else ():
            if isinstance(dataset, (dict, DictConfig)) and dataset.get("taskset") is not None:
                tasksets.add(str(dataset["taskset"]))
    return tasksets


@cache
def _composed(config_path: Path) -> dict:
    """The config as a run merges it from its `config_paths`, without resolving any value."""
    _paths, configs = GlobalConfigDictParser().load_extra_config_paths([str(config_path)])
    return OmegaConf.to_container(OmegaConf.merge(*configs), resolve=False, throw_on_missing=False)


@cache
def _catalog_names_by_config() -> dict[Path, str]:
    """The catalog name each manifest gives its config."""
    return {
        entry.config_path.resolve(): entry.name
        for entry in discover_environment_catalog()
        if entry.manifest_path is not None and REPO_ROOT in entry.manifest_path.resolve().parents
    }


@cache
def _manifest_folders() -> dict[Path, str]:
    return {
        entry.manifest_path.resolve().parent: entry.name
        for entry in discover_environment_catalog()
        if entry.manifest_path is not None and REPO_ROOT in entry.manifest_path.resolve().parents
    }


def _owning_catalog_name(config_path: Path) -> str | None:
    """The catalog name a config's tasksets must carry, or None when no catalog entry owns the config.

    No config names its catalog entry yet, so a run preset is matched by where it sits: beside the config of exactly
    one manifest (`deepsearchqa/configs/deepsearchqa_codex.yaml` beside `deepsearchqa/long_run`'s Claude Code copy),
    else under the nearest manifest above it.
    """
    config_path = config_path.resolve()
    names = _catalog_names_by_config()
    if (name := names.get(config_path)) is not None:
        return name
    if config_path.relative_to(REPO_ROOT).parts[0] not in CATALOG_TREES:
        return None
    beside = {name for config, name in names.items() if config.parent == config_path.parent}
    if len(beside) == 1:
        return beside.pop()
    folders = _manifest_folders()
    return next((folders[folder] for folder in config_path.parents if folder in folders), None)


def test_every_catalog_config_names_its_catalog_entry_in_its_tasksets() -> None:
    problems = []
    for path in _tracked_yaml():
        if path.name == "manifest.yaml" or (catalog_name := _owning_catalog_name(path)) is None:
            continue
        own = _load(path)
        if not isinstance(own, dict):
            continue
        # A route the config takes from a component it includes belongs to that component; its own routes and every
        # dataset it runs, its own or included, name this entry.
        named = {("route", key) for key in _route_keys(own)}
        try:
            composed = _composed(path)
        except ConfigPathNotFoundError:
            composed = own  # a config that includes a missing file still declares its own datasets
        named |= {("taskset", taskset) for taskset in _dataset_tasksets(composed)}
        for kind, name in sorted(named):
            if (problem := taskset_problem(name, catalog_name)) is not None:
                problems.append(f"{path.relative_to(REPO_ROOT)}: {kind} `{name}` {problem}")
    assert not problems, "Tasksets that do not name their catalog entry:\n" + "\n".join(problems)


def test_no_config_uses_a_renamed_taskset() -> None:
    problems = []
    for path in _tracked_yaml():
        config = _load(path)
        for name in sorted(_route_keys(config) | _dataset_tasksets(config)):
            if (problem := renamed_problem(name)) is not None:
                problems.append(f"{path.relative_to(REPO_ROOT)}: `{name}` {problem}")
    assert not problems, "Configs that use a renamed taskset:\n" + "\n".join(problems)


@pytest.mark.parametrize(
    ("taskset", "problem"),
    [
        ("swebench/verified", None),
        ("swebench/verified:opencode", None),
        ("swebench/verified:train-2", None),
        ("swebench:verified", "must be `swebench/verified` or `swebench/verified:<label>`"),
        ("swebench/verified/opencode", "must be `swebench/verified` or `swebench/verified:<label>`"),
        ("swebench/verifiedx", "must be `swebench/verified` or `swebench/verified:<label>`"),
        ("swebench/verified:", "its label must be one name"),
        ("swebench/verified:a:b", "its label must be one name"),
        ("swebench/verified:a/b", "its label must be one name"),
    ],
)
def test_a_taskset_is_its_catalog_name_and_an_optional_label(taskset: str, problem: str | None) -> None:
    found = taskset_problem(taskset, "swebench/verified")
    assert found == problem if problem is None else found.startswith(problem)


def test_a_taskset_under_a_renamed_catalog_name_is_refused() -> None:
    # `bird_interact:c_interact` would otherwise read as the catalog name `bird_interact` and a label.
    assert renamed_problem("bird_interact:c_interact") is not None
    assert renamed_problem("bird_interact:anything") == (
        "names `bird_interact`, which was renamed to `bird_interact/lite/c_interact`"
    )
    assert renamed_problem("bird_interact/lite/c_interact") is None

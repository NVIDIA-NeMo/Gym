# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provisioning shared by the Harbor benchmarks: write rows for a benchmark and describe the images it needs.

Every Harbor benchmark config declares one harbor_tasks resources server whose `harbor_datasets` pin the Harbor
datasets and whose `image_template` names built environments. These helpers read those settings from the config,
so a benchmark's prepare script and build_images.py agree with the server that runs it.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

from nemo_gym import PARENT_DIR
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.sandbox.adapters.harbor_metadata import (
    HARBOR_AGENT_TIMEOUT_METADATA_KEY,
    HARBOR_AGENT_USER_METADATA_KEY,
)


HARBOR_REQUIREMENTS = PARENT_DIR / "resources_servers" / "harbor_tasks" / "requirements.txt"


@dataclass(frozen=True)
class HarborTasksSettings:
    """The harbor_tasks settings provisioning shares with the server that runs a benchmark."""

    resources_server: str
    datasets: dict[str, dict[str, Any]]
    image_template: Optional[str]
    allow_unenforced_network_policy: bool


def load_harbor_tasks_settings(config_path: Path, resources_server: Optional[str] = None) -> HarborTasksSettings:
    """Read the harbor_tasks server settings from a Gym config, resolving it as `gym env start` would."""
    initial = OmegaConf.merge(OmegaConf.load(config_path), GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT)
    resolved = GlobalConfigDictParser().parse_no_environment(initial_global_config_dict=initial)
    candidates = {}
    for name, instance in resolved.items():
        servers = instance.get("resources_servers") if isinstance(instance, DictConfig) else None
        for server in (servers or {}).values():
            if isinstance(server, DictConfig) and "harbor_datasets" in server:
                candidates[str(name)] = OmegaConf.to_container(server, resolve=True)
    if resources_server is not None:
        candidates = {name: server for name, server in candidates.items() if name == resources_server}
    if len(candidates) != 1:
        raise ValueError(
            f"{config_path} must resolve to exactly one harbor_tasks resources server"
            f"{f' named {resources_server!r}' if resources_server else ''}; found {sorted(candidates)}"
        )
    [(name, server)] = candidates.items()
    datasets = {}
    for alias, dataset in server["harbor_datasets"].items():
        dataset = dict(dataset)
        if dataset.get("path") and not Path(dataset["path"]).is_absolute():
            dataset["path"] = str(PARENT_DIR / dataset["path"])
        datasets[alias] = dataset
    return HarborTasksSettings(
        resources_server=name,
        datasets=datasets,
        image_template=server.get("image_template"),
        allow_unenforced_network_policy=bool(server.get("allow_unenforced_network_policy", False)),
    )


def harbor_requirement() -> str:
    """The Harbor pin the harbor_tasks server installs, so provisioning resolves tasks the same way."""
    for line in HARBOR_REQUIREMENTS.read_text().splitlines():
        if line.startswith("harbor"):
            return line.strip()
    raise ValueError(f"No harbor requirement in {HARBOR_REQUIREMENTS}")


def describe_tasks(
    settings: HarborTasksSettings, *, task_names: Optional[list[str]] = None, limit: Optional[int] = None
) -> list[dict[str, Any]]:
    """Download the configured datasets with Harbor and describe each task (see harbor_side.py).

    Harbor runs in an isolated `uv` environment because its dependencies conflict with Gym's.
    """
    datasets = {
        alias: dataset | ({"task_names": task_names} if task_names else {}) | ({"n_tasks": limit} if limit else {})
        for alias, dataset in settings.datasets.items()
    }
    request = {
        "datasets": datasets,
        "image_template": settings.image_template,
        "allow_unenforced_network_policy": settings.allow_unenforced_network_policy,
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        request_path, output_path = Path(tmp_dir) / "request.json", Path(tmp_dir) / "tasks.jsonl"
        request_path.write_text(json.dumps(request))
        command = [
            "uv",
            "run",
            "--isolated",
            "--no-project",
            "--with",
            harbor_requirement(),
            "python",
            "-m",
            "benchmarks.harbor.prepare_utils.harbor_side",
            str(request_path),
            str(output_path),
        ]
        # The repository root is the import root for benchmarks.* and resources_servers.harbor_tasks.tasks.
        env = os.environ | {"PYTHONPATH": str(PARENT_DIR)}
        subprocess.run(command, cwd=PARENT_DIR, env=env, check=True)
        return [json.loads(line) for line in output_path.read_text().splitlines()]


def task_row(record: dict[str, Any]) -> dict[str, Any]:
    """The harbor_tasks row for one described task: its name, and its instruction as the user message."""
    metadata = {}
    if record["agent_timeout_sec"] is not None:
        metadata[HARBOR_AGENT_TIMEOUT_METADATA_KEY] = str(record["agent_timeout_sec"])
    if record["agent_user"] is not None:
        metadata[HARBOR_AGENT_USER_METADATA_KEY] = str(record["agent_user"])
    responses_create_params: dict[str, Any] = {"input": [{"role": "user", "content": record["instruction"]}]}
    if metadata:
        responses_create_params["metadata"] = metadata
    return {
        "task_id": record["task_name"],
        "harbor_dataset": record["harbor_dataset"],
        "task_name": record["task_name"],
        "responses_create_params": responses_create_params,
    }


def prepare_rows(config_path: Path, output_fpath: Path, *, skip_unsupported: bool = False) -> Path:
    """Write the rows for a Harbor benchmark config and report the images its tasks need.

    A task harbor_tasks cannot run raises, unless `skip_unsupported`. Images are only reported, never built here:
    run build_images.py with the same config to build the environments the rows need.
    """
    records = describe_tasks(load_harbor_tasks_settings(config_path))
    rows = []
    for record in records:
        if record["unsupported"]:
            message = (
                f"Harbor task {record['task_name']!r} uses unsupported features: {', '.join(record['unsupported'])}"
            )
            if not skip_unsupported:
                raise ValueError(message)
            print(f"Skipping: {message}")
            continue
        rows.append(task_row(record))
    output_fpath.parent.mkdir(parents=True, exist_ok=True)
    output_fpath.write_text("".join(json.dumps(row) + "\n" for row in rows))

    built = {record["image"] for record in records if record["builds_image"] and not record["unsupported"]}
    print(f"Wrote {len(rows)} Harbor task rows to {output_fpath}")
    if built:
        print(
            f"{len(built)} environment image(s) are built from environment/Dockerfile. Build them before running:\n"
            f"  python -m benchmarks.harbor.prepare_utils.build_images --config {config_path} --load"
        )
    return output_fpath

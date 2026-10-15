# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn loaded tasks into single-agent-turn rows and the run config that serves them."""

import json
from pathlib import Path
from typing import Any

from nemo_gym.episode_types import MaterializedTask, TaskId
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput
from nemo_gym.tasks.harbor.digest import DATASET_DIGEST_KEY, DIGEST_KEY
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.task import HarborTask, is_task_folder


AGENT_TIMEOUT_METADATA_KEY = "agent_timeout_sec"

# Instance names in the generated run config, per taskset.
ENVIRONMENT_SERVER_IMPL = "harbor"
ENVIRONMENT_SERVER_INSTANCE = "harbor_environment_server"


def _instance_taskset(taskset: str) -> str:
    # Hydra keys cannot contain arbitrary folder-name characters. Escape UTF-8 bytes, including
    # the escape marker itself, so e.g. "2.0" and "2_2e0" cannot produce the same instance name.
    safe = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"  # pragma: allowlist secret
    return "".join(chr(byte) if byte in safe else f"_{byte:02x}" for byte in taskset.encode("utf-8"))


def agent_instance(taskset: str) -> str:
    """Return an override-compatible agent instance name for the taskset."""
    return f"harbor_{_instance_taskset(taskset)}_agent"


def environment_server_instance(taskset: str) -> str:
    """Return an override-compatible environment instance name for the taskset."""
    return f"harbor_{_instance_taskset(taskset)}_environment"


def materialize_task(
    task: HarborTask, taskset: str, *, dataset_digest: str | None = None
) -> MaterializedTask[SingleAgentTurnTaskInput]:
    """One row: the instruction as a user message, the folder digest in ``task_data``.

    With ``dataset_digest``, the row also carries the dataset's digest (``manifest.toml``'s
    ``[dataset].digest``). Rollout collection copies the ``_ng_*`` keys of ``task_data`` onto every
    results row it writes, so each row then says which task folder and which dataset it ran.
    """
    task_data = {DIGEST_KEY: task.digest}
    if dataset_digest is not None:
        task_data[DATASET_DIGEST_KEY] = dataset_digest
    return MaterializedTask(
        task_id=TaskId(taskset=taskset, task_id=task.task_id),
        task_input=SingleAgentTurnTaskInput(
            responses_create_params={
                "input": [{"role": "user", "content": task.instruction}],
                "metadata": {AGENT_TIMEOUT_METADATA_KEY: str(task.config.agent.timeout_sec)},
            },
            task_data=task_data,
        ),
    )


def _validate_task_ids(tasks: list[HarborTask]) -> None:
    seen: set[str] = set()
    for task in tasks:
        if task.task_id in seen:
            raise HarborTaskError(f"Duplicate Harbor task ID {task.task_id!r} in one taskset")
        seen.add(task.task_id)


def write_rows(
    tasks: list[HarborTask], taskset: str, path: Path, *, dataset_digest: str | None = None
) -> list[dict[str, Any]]:
    """Write one materialized row per task as JSONL and return the rows."""
    _validate_task_ids(tasks)
    rows = [
        materialize_task(task, taskset, dataset_digest=dataset_digest).model_dump(mode="json", exclude_unset=True)
        for task in tasks
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return rows


def taskset_mapping(tasks: list[HarborTask], folder: Path) -> dict[str, Any]:
    """Where a taskset's folders live and which digest each task had at materialization.

    Servers resolve a task as ``<folder>/<task_id>``, or ``<folder>/<paths[task_id]>`` for a task
    found below the first level (see :func:`nemo_gym.tasks.harbor.task.discover_tasks`). For a
    folder of task folders ``folder`` is the folder itself; for a single task folder it is the
    task's parent, so ``folder`` is always the root the tasks were discovered under.
    """
    _validate_task_ids(tasks)
    root = Path(folder).resolve()
    if is_task_folder(root):
        root = root.parent
    paths: dict[str, str] = {}
    for task in tasks:
        try:
            relative = task.path.relative_to(root)
        except ValueError:
            raise ValueError(f"Task {task.path} is not under the taskset folder {root}") from None
        if relative.as_posix() != task.task_id:
            paths[task.task_id] = relative.as_posix()
    mapping: dict[str, Any] = {"folder": str(root), "tasks": {task.task_id: task.digest for task in tasks}}
    if paths:
        mapping["paths"] = paths
    return mapping


def run_config(
    *,
    taskset: str,
    folder: Path,
    tasks: list[HarborTask],
    rows_path: Path,
    agent_instance_name: str,
    agent_impl: str,
    agent_overrides: dict[str, Any] | None = None,
    sandbox_provider: str = "sandbox",
    dataset_digest: str | None = None,
    dataset_toml_digest: str | None = None,
) -> dict[str, Any]:
    """The run-config blocks that bind one taskset to the ``harbor`` environment server.

    The environment server block inherits the shipped ``harbor_environment_server`` config and
    names the taskset's agent. The agent block inherits the selected harness config and declares
    the taskset's rows as its validation dataset, which is where rollout collection looks for
    datasets. Rows route to the environment server by taskset. The taskset mapping records the
    dataset digest and the ``dataset.toml`` digest, so the run config says which dataset ran and
    whether the Gym layer or upstream moved between two runs.
    """
    agent_name = agent_instance(taskset)
    environment_name = environment_server_instance(taskset)
    mapping = taskset_mapping(tasks, folder)
    if dataset_digest is not None:
        mapping["dataset_digest"] = dataset_digest
    if dataset_toml_digest is not None:
        mapping["dataset_toml_digest"] = dataset_toml_digest
    return {
        "environment_routing_mode": "taskset",
        "environment_server_routes": {taskset: environment_name},
        agent_name: {
            "_inherit_from": agent_instance_name,
            "responses_api_agents": {
                agent_impl: {
                    "datasets": [
                        {
                            "name": taskset,
                            "type": "validation",
                            # Harbor task folders carry no dataset-level license; the run config
                            # needs one for a validation split, so it is left to be determined.
                            "license": "TBD",
                            "jsonl_fpath": str(Path(rows_path).resolve()),
                        }
                    ],
                    **(agent_overrides or {}),
                }
            },
        },
        environment_name: {
            "_inherit_from": ENVIRONMENT_SERVER_INSTANCE,
            "environment_servers": {
                ENVIRONMENT_SERVER_IMPL: {
                    "sandbox_provider": sandbox_provider,
                    # Verifier logs sit beside the run's rollouts, as an absolute path: the server
                    # process runs from its own folder, so a relative default would land there.
                    "artifacts_dir": str(Path(rows_path).resolve().parent / "verifier"),
                    "tasksets": {taskset: mapping},
                    "agent_server": {"type": "responses_api_agents", "name": agent_name},
                }
            },
        },
    }

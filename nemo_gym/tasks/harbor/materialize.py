# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn loaded tasks into single-agent-turn rows and the run config that serves them."""

import json
from pathlib import Path
from typing import Any

from nemo_gym.episode_types import MaterializedTask, TaskId
from nemo_gym.single_agent_turn_types import SingleAgentTurnTaskInput
from nemo_gym.tasks.harbor.digest import DIGEST_KEY
from nemo_gym.tasks.harbor.task import HarborTask


AGENT_TIMEOUT_METADATA_KEY = "agent_timeout_sec"

# Instance names in the generated run config, per taskset.
RESOURCES_SERVER_IMPL = "harbor"
RESOURCES_SERVER_INSTANCE = "harbor_resources_server"
ENVIRONMENT_SERVER_IMPL = "single_agent_turn"


def resources_server_instance(taskset: str) -> str:
    return f"harbor_{taskset}_resources_server"


def agent_instance(taskset: str) -> str:
    return f"harbor_{taskset}_agent"


def environment_server_instance(taskset: str) -> str:
    return f"harbor_{taskset}_environment"


def materialize_task(task: HarborTask, taskset: str) -> MaterializedTask[SingleAgentTurnTaskInput]:
    """One row: the instruction as a user message, the folder digest in ``task_data``."""
    return MaterializedTask(
        task_id=TaskId(taskset=taskset, task_id=task.task_id),
        task_input=SingleAgentTurnTaskInput(
            responses_create_params={
                "input": [{"role": "user", "content": task.instruction}],
                "metadata": {AGENT_TIMEOUT_METADATA_KEY: str(task.config.agent.timeout_sec)},
            },
            task_data={DIGEST_KEY: task.digest},
        ),
    )


def write_rows(tasks: list[HarborTask], taskset: str, path: Path) -> list[dict[str, Any]]:
    """Write one materialized row per task as JSONL and return the rows."""
    rows = [materialize_task(task, taskset).model_dump(mode="json", exclude_unset=True) for task in tasks]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    return rows


def taskset_mapping(tasks: list[HarborTask], folder: Path) -> dict[str, Any]:
    """Where a taskset's folders live and which digest each task had at materialization.

    Servers resolve a task as ``<folder>/<task_id>``. For a folder of task folders that is
    the folder itself; for a single task folder it is the task's parent, so ``folder`` is
    always the tasks' common parent directory.
    """
    parents = {task.path.parent for task in tasks}
    if len(parents) != 1:
        raise ValueError(f"Tasks of one taskset must share a parent folder, got {sorted(map(str, parents))}")
    return {"folder": str(parents.pop().resolve()), "tasks": {task.task_id: task.digest for task in tasks}}


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
) -> dict[str, Any]:
    """The run-config blocks that bind one taskset to the ``harbor`` resources server.

    The resources server block inherits the shipped ``harbor_resources_server`` config. The
    agent block inherits the selected harness config and points it at the taskset's
    resources server. The environment server is a ``single_agent_turn`` instance. Rows
    route to it by taskset.
    """
    resources_name = resources_server_instance(taskset)
    agent_name = agent_instance(taskset)
    environment_name = environment_server_instance(taskset)
    return {
        "environment_routing_mode": "taskset",
        "environment_server_routes": {taskset: environment_name},
        resources_name: {
            "_inherit_from": RESOURCES_SERVER_INSTANCE,
            "resources_servers": {
                RESOURCES_SERVER_IMPL: {
                    "sandbox_provider": sandbox_provider,
                    "tasksets": {taskset: taskset_mapping(tasks, folder)},
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
                }
            },
        },
        agent_name: {
            "_inherit_from": agent_instance_name,
            "responses_api_agents": {
                agent_impl: {
                    "resources_server": {"type": "resources_servers", "name": resources_name},
                    **(agent_overrides or {}),
                }
            },
        },
        environment_name: {
            "_inherit_from": ENVIRONMENT_SERVER_IMPL,
            "environment_servers": {
                ENVIRONMENT_SERVER_IMPL: {
                    "resources_server": {"type": "resources_servers", "name": resources_name},
                    "agent_server": {"type": "responses_api_agents", "name": agent_name},
                }
            },
        },
    }

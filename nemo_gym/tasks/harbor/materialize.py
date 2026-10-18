# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn loaded tasks into episode rows and the run config that serves them."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from typing_extensions import Self

from nemo_gym.episode_types import MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.tasks.harbor.digest import DATASET_DIGEST_KEY, DIGEST_KEY
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.task import HarborTask, is_task_folder


AGENT_TIMEOUT_METADATA_KEY = "agent_timeout_sec"
# The Harbor step a turn's request belongs to, in its `responses_create_params.metadata`.
STEP_METADATA_KEY = "harbor_step"


class HarborTurnInput(BaseModel):
    """One turn of a multi-step task: the Harbor step's name and the request the agent receives."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class HarborTaskInput(BaseModel):
    """A Harbor row's input: one ``responses_create_params`` for a single-turn task, or one per turn.

    ``task_data`` carries the folder digest; the server reads everything else from the pinned folder.
    """

    model_config = ConfigDict(extra="forbid")

    responses_create_params: NeMoGymResponseCreateParamsNonStreaming | None = None
    task_data: dict[str, JsonValue]
    turns: list[HarborTurnInput] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def accept_flat_task_input(cls, value: object) -> object:
        """Fold flat task fields into ``task_data``, keeping both request-envelope fields where they are.

        The flat single-turn input (the task fields beside ``responses_create_params``, as
        ``SingleAgentTurnTaskInput`` accepts them) is the canonical row once validated, so a client that sends
        ``{"responses_create_params": ..., "_ng_digest": ...}`` runs as the row does. Only the keys the input set
        are passed on, so a row that names no ``responses_create_params`` (a multi-step row) keeps the field unset
        and its serialized form (``exclude_unset``) carries no ``null``.
        """
        if not isinstance(value, Mapping):
            return value
        fields = dict(value)
        envelope = {key: fields.pop(key) for key in ("responses_create_params", "turns") if key in fields}
        task_data = fields.pop("task_data", {})
        if not isinstance(task_data, Mapping):
            return value  # Let the field validator report the malformed canonical container.
        for key in fields.keys() & task_data.keys():
            if fields[key] != task_data[key]:
                raise ValueError(f"Conflicting task field {key!r} inside and outside task_data")
        return {**envelope, "task_data": dict(task_data) | fields}

    @model_validator(mode="after")
    def one_shape(self) -> Self:
        if (self.responses_create_params is None) == (not self.turns):
            raise ValueError("A Harbor row has either responses_create_params or turns, not both")
        return self


def _turn_params(instruction: str, timeout_sec: float, step: str | None = None) -> dict[str, Any]:
    metadata = {AGENT_TIMEOUT_METADATA_KEY: str(timeout_sec)}
    if step is not None:
        metadata[STEP_METADATA_KEY] = step
    return {"input": [{"role": "user", "content": instruction}], "metadata": metadata}


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
) -> MaterializedTask[HarborTaskInput]:
    """One row: the instruction as a user message, the folder digest in ``task_data``.

    A multi-step task gets one turn per step, each with that step's instruction and agent budget. The
    top-level ``instruction.md`` is never sent for such a task.

    With ``dataset_digest``, the row also carries the dataset's digest (``manifest.toml``'s
    ``[dataset].digest``). Rollout collection copies the ``_ng_*`` keys of ``task_data`` onto every
    results row it writes, so each row then says which task folder and which dataset it ran.
    """
    task_data: dict[str, JsonValue] = {DIGEST_KEY: task.digest}
    if dataset_digest is not None:
        task_data[DATASET_DIGEST_KEY] = dataset_digest
    if task.turns:
        task_input = HarborTaskInput(
            task_data=task_data,
            turns=[
                HarborTurnInput(
                    name=turn.name,
                    responses_create_params=_turn_params(
                        turn.instruction, task.config.step_agent_timeout_sec(turn.config), step=turn.name
                    ),
                )
                for turn in task.turns
            ],
        )
    else:
        task_input = HarborTaskInput(
            responses_create_params=_turn_params(task.instruction, task.config.agent.timeout_sec),
            task_data=task_data,
        )
    return MaterializedTask(task_id=TaskId(taskset=taskset, task_id=task.task_id), task_input=task_input)


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

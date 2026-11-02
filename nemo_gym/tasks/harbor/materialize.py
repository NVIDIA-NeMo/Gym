# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Turn loaded tasks into episode rows and the run config that serves them."""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from typing_extensions import Self

from nemo_gym.core.episode import TaskId
from nemo_gym.core.responses_api_types import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime.episode_types import MaterializedTask
from nemo_gym.tasks.harbor.dataset_config import GymDatasetConfig
from nemo_gym.tasks.harbor.digest import DATASET_DIGEST_KEY, DIGEST_KEY
from nemo_gym.tasks.harbor.errors import HarborTaskError
from nemo_gym.tasks.harbor.task import HarborTask, is_task_folder


AGENT_TIMEOUT_METADATA_KEY = "agent_timeout_sec"
# The Harbor step a turn's request belongs to, in its `responses_create_params.metadata`.
STEP_METADATA_KEY = "harbor_step"
# The dataset a row came from, on every row: the dataset folder's name (the taskset, unless the target is a single
# task folder, whose dataset is its parent). The same string as `nemo_gym.config.global_config.TASK_SOURCE_KEY_NAME`,
# which this module does not import; rollout collection copies the key from a row onto the record it writes, so
# every results row names its dataset.
TASK_SOURCE_KEY = "task_source"


class HarborTurnInput(BaseModel):
    """One turn of a multi-step task: the Harbor step's name and the request the agent receives."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming


class HarborTaskInput(BaseModel):
    """A Harbor row's input: one ``responses_create_params`` for a single-turn task, or one per turn.

    ``task_data`` carries the folder digest and the task's own fields; the server reads everything else
    from the pinned folder.
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
    """One row: the instruction as a user message, the folder digest and the task's fields in ``task_data``.

    A multi-step task gets one turn per step, each with that step's instruction and agent budget. The
    top-level ``instruction.md`` is never sent for such a task.

    ``task_data`` holds the task's own field values (``task_data.json`` in its folder, validated against
    the dataset's ``task_data.py`` before any row is written) under the reserved ``_ng_*`` keys. The values
    are the file's as written, not the validated model's dump: the model validates strictly, as JSON, so an
    accepted value already has its field's type (:meth:`~nemo_gym.tasks.harbor.task_data.TaskDataSchema.validate`),
    and the same fields are what the seed compares with the folder's file and what the in-sandbox verifier
    reads as ``task_data.json``. A dump would drift from both (defaults filled in, aliases and ``AliasPath``
    shapes rewritten) without changing what any verifier sees.

    With ``dataset_digest``, the row also carries the dataset's digest (``manifest.toml``'s
    ``[dataset].digest``). Rollout collection copies the ``_ng_*`` keys of ``task_data`` onto every
    results row it writes, so each row then says which task folder and which dataset it ran.
    """
    task_data: dict[str, JsonValue] = {**task.task_data, DIGEST_KEY: task.digest}
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
    tasks: list[HarborTask],
    taskset: str,
    path: Path,
    *,
    dataset_digest: str | None = None,
    task_source: str | None = None,
) -> list[dict[str, Any]]:
    """Write one materialized row per task as JSONL and return the rows.

    Beside the task's identity and input, every row carries ``task_source``: the dataset folder's name,
    ``task_source`` when given (a single task folder target inside a dataset passes its parent's name; a
    standalone task folder, whose parent holds no ``dataset.toml`` or ``manifest.toml``, passes its own) and
    the taskset otherwise (a dataset folder target's taskset is that name). It is the dataset label a results row and
    a trainer read, and nothing the collector routes by: a materialized row reaches its environment server
    by its taskset, ``agent_map`` and ``fan_out`` do not apply to it, and a per-dataset ``num_repeats`` dict
    keys on the taskset.

    The metrics sidecars the data processor leaves beside the rows (:func:`rows_metrics_sidecars`) go with the
    rows they described: they are derived from one selection of the task folder, and the processor would refuse
    a changed selection against them ("Found conflicting aggregate metrics"), a check meant for a committed
    source dataset, not for rows regenerated on every command. Removed here, they are rewritten by the run.
    """
    _validate_task_ids(tasks)
    label = task_source or taskset
    rows = [
        materialize_task(task, taskset, dataset_digest=dataset_digest).model_dump(mode="json", exclude_unset=True)
        | {TASK_SOURCE_KEY: label}
        for task in tasks
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")
    for sidecar in rows_metrics_sidecars(path):
        sidecar.unlink(missing_ok=True)
    return rows


def rows_metrics_sidecars(rows_path: Path) -> tuple[Path, Path]:
    """The data processor's per-dataset metrics beside the rows and the conflict file it writes on a mismatch:
    ``<rows stem>_metrics.json`` and ``<rows stem>_metrics_conflict.json`` (``nemo_gym.train_data_utils``)."""
    metrics = rows_path.with_name(f"{rows_path.stem}_metrics.json")
    return metrics, metrics.with_name(f"{metrics.stem}_conflict.json")


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


def dataset_entry(taskset: str, rows_path: Path, dataset: GymDatasetConfig | None = None) -> dict[str, Any]:
    """The ``datasets:`` entry that declares a taskset's rows, from the dataset's ``[gym]`` table.

    ``type`` is the split the rows enter (``GymDatasetConfig.effective_type``: the declared one, else
    ``validation`` with a license, else ``benchmark``) and ``license`` the declared one, so a hub dataset
    without a license is a benchmark that evaluates and never trains, and a ``train`` or ``validation``
    dataset carries the license Gym requires of one. ``verified`` is the dataset's declaration (``false``
    unless it says otherwise, as the pre-commit hook starts a new resources server), and ``domain`` and
    ``allowed_agents`` are carried when declared. The entry sets no ``num_repeats``: collate would duplicate
    the rows by it and the collector's run-level count would multiply on top, and the metrics sidecar beside
    the rows records the entry, so a count that moved with the command line would conflict run to run.
    ``[gym.defaults].num_repeats`` is the run config's top-level ``num_repeats`` instead (see
    ``nemo_gym.tasks.harbor.cli.build_run``). A benchmark entry names no ``prepare_script``: the rows are
    written by the preparation that generates this entry.
    """
    dataset = dataset or GymDatasetConfig()
    entry: dict[str, Any] = {"name": taskset, "type": dataset.effective_type}
    if dataset.license is not None:
        entry["license"] = dataset.license
    if dataset.domain is not None:
        entry["domain"] = dataset.domain.value
    entry["verified"] = dataset.verified
    if dataset.allowed_agents is not None:
        entry["allowed_agents"] = list(dataset.allowed_agents)
    entry["jsonl_fpath"] = str(Path(rows_path).resolve())
    return entry


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
    dataset: GymDatasetConfig | None = None,
) -> dict[str, Any]:
    """The run-config blocks that bind one taskset to the ``harbor`` environment server.

    The environment server block inherits the shipped ``harbor_environment_server`` config and
    names the taskset's agent. The agent block inherits the selected harness config and declares
    the taskset's rows as its dataset (:func:`dataset_entry`, from the ``[gym]`` table of
    ``dataset.toml``: split, license, repeats and catalog metadata), which is where rollout collection
    looks for datasets. Rows route to the environment server by taskset. The taskset mapping records the
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
                    "datasets": [dataset_entry(taskset, rows_path, dataset)],
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

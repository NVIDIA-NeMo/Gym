# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build a benchmark's evaluation plan: every task once, with a stable id and a content digest.

The plan file format is versioned by its schema label and only changes additively within a version.
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from nemo_gym.config.config_types import ConfigError
from nemo_gym.core.episode_types import TaskId, is_materialized_task_row
from nemo_gym.data.task_materialization import TASK_ID_FIELDS, materialize_task, require_row_in_taskset
from nemo_gym.data.task_selection import row_task_id


EVAL_PLAN_SCHEMA = "nemo-gym.eval-plan.v1"


class PlanGym(BaseModel):
    """The Gym that produced the plan."""

    model_config = ConfigDict(extra="forbid")

    version: str
    commit: str | None = Field(description="Git commit of the running Gym, '-dirty' suffixed; None if unknown.")


class PlanBenchmark(BaseModel):
    """The benchmark the plan was made for."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="The benchmark dataset name.")
    config_paths: list[str] = Field(description="Config files the benchmark was selected with.")


class PlanTask(BaseModel):
    """One task: its identity and a digest of its prepared input."""

    model_config = ConfigDict(extra="forbid")

    task_id: TaskId
    digest: str = Field(description="'sha256:' + SHA-256 of the task's input (task_input) as canonical JSON.")


class EvalPlan(BaseModel):
    """A benchmark's tasks, each listed once, sorted by task id."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    schema_name: Literal["nemo-gym.eval-plan.v1"] = Field(default=EVAL_PLAN_SCHEMA, alias="schema")
    gym: PlanGym
    benchmark: PlanBenchmark
    selected_task_ids: list[str] | None = Field(
        default=None, description="The requested subset of task ids, or None when the plan covers every task."
    )
    tasks: list[PlanTask]


def canonical_json(value: JsonValue) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _task_of_row(
    row: Mapping[str, JsonValue], *, taskset: str | None, default_taskset: str
) -> tuple[TaskId, JsonValue]:
    """A row's identity and task input, whether the row is flat or already materialized.

    A flat row is materialized into ``taskset``, or ``default_taskset`` when the dataset declares none. A materialized
    row keeps the identity and task input it carries; when the dataset declares a taskset, the row must name it, as
    dataset collation requires.
    """
    if is_materialized_task_row(row):
        if taskset is not None:
            require_row_in_taskset(row, taskset)
        return TaskId.model_validate(row["task_id"]), row["task_input"]
    materialized = materialize_task(row, taskset=taskset or default_taskset)
    return TaskId.model_validate(materialized["task_id"]), materialized["task_input"]


def build_eval_plan(
    rows: Sequence[Mapping[str, JsonValue]],
    *,
    benchmark: PlanBenchmark,
    taskset: str | None,
    gym: PlanGym,
    selected_task_ids: Sequence[str] | None = None,
) -> EvalPlan:
    """Turn prepared benchmark rows into a plan.

    A row is either flat or already materialized (a ``task_id`` of taskset and id, and a ``task_input``), as collation
    accepts both. A flat row must carry its own id in one of ``TASK_ID_FIELDS``. Unlike collation, there is no
    fallback to the row's position, because a position-based id names a different task once rows are reordered.
    A materialized row's id and task input are used as written, and its digest is that of its ``task_input``.
    ``taskset`` is the dataset's declared taskset, or None when it declares none: flat rows then take the benchmark's
    name, and materialized rows keep the taskset they name.
    """
    missing = [
        index for index, row in enumerate(rows) if not is_materialized_task_row(row) and row_task_id(row) is None
    ]
    if missing:
        raise ConfigError(
            f"Benchmark {benchmark.name!r}: {len(missing)} of {len(rows)} tasks have no task id "
            f"(none of {list(TASK_ID_FIELDS)}), e.g. rows {missing[:5]}. "
            "A plan needs ids that do not depend on row order; add a `task_id` field in the benchmark's prepare.py."
        )

    tasks: dict[str, PlanTask] = {}
    duplicates: set[str] = set()
    for row in rows:
        try:
            task_id, task_input = _task_of_row(row, taskset=taskset, default_taskset=benchmark.name)
        except ValueError as e:
            raise ConfigError(f"Benchmark {benchmark.name!r}: {e}") from e
        if task_id.task_id in tasks:
            duplicates.add(task_id.task_id)
            continue
        tasks[task_id.task_id] = PlanTask(
            task_id=task_id,
            digest="sha256:" + hashlib.sha256(canonical_json(task_input)).hexdigest(),
        )
    if duplicates:
        raise ConfigError(f"Benchmark {benchmark.name!r}: task ids are not unique: {sorted(duplicates)[:10]}")

    if selected_task_ids is not None:
        unknown = sorted(set(selected_task_ids) - tasks.keys())
        if unknown:
            raise ConfigError(f"Benchmark {benchmark.name!r} has no tasks with ids {unknown}")
        tasks = {task_id: tasks[task_id] for task_id in selected_task_ids}

    return EvalPlan(
        gym=gym,
        benchmark=benchmark,
        selected_task_ids=sorted(set(selected_task_ids)) if selected_task_ids is not None else None,
        tasks=[tasks[task_id] for task_id in sorted(tasks)],
    )


def eval_plan_json(plan: EvalPlan) -> str:
    """Serialize a plan deterministically: the same plan always gives the same bytes."""
    return json.dumps(plan.model_dump(mode="json", by_alias=True), sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def eval_plan_json_schema() -> str:
    return json.dumps(EvalPlan.model_json_schema(by_alias=True), indent=2, sort_keys=True) + "\n"


def write_eval_plan(plan: EvalPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(eval_plan_json(plan))

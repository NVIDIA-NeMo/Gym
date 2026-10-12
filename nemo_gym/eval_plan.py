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

from nemo_gym.config_types import ConfigError
from nemo_gym.episode_types import TaskId
from nemo_gym.task_materialization import TASK_ID_FIELDS, materialize_task


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
    """One task: its identity and a digest of the exact input a run receives."""

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


def _row_task_id(row: Mapping[str, JsonValue]) -> str | None:
    return next((str(row[key]) for key in TASK_ID_FIELDS if row.get(key) is not None), None)


def build_eval_plan(
    rows: Sequence[Mapping[str, JsonValue]],
    *,
    benchmark: PlanBenchmark,
    taskset: str,
    gym: PlanGym,
    selected_task_ids: Sequence[str] | None = None,
) -> EvalPlan:
    """Turn prepared benchmark rows into a plan.

    Every row must carry its own id in one of ``TASK_ID_FIELDS``. Unlike collation, there is no fallback to
    the row's position, because a position-based id names a different task once rows are reordered.
    ``taskset`` is the dataset's declared taskset, or the benchmark name when it declares none.
    """
    missing = [index for index, row in enumerate(rows) if _row_task_id(row) is None]
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
            materialized = materialize_task(row, taskset=taskset)
        except ValueError as e:
            raise ConfigError(f"Benchmark {benchmark.name!r}: {e}") from e
        task_id = TaskId.model_validate(materialized["task_id"])
        if task_id.task_id in tasks:
            duplicates.add(task_id.task_id)
            continue
        tasks[task_id.task_id] = PlanTask(
            task_id=task_id,
            digest="sha256:" + hashlib.sha256(canonical_json(materialized["task_input"])).hexdigest(),
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

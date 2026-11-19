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
"""Run context on the eval driver's spans: which benchmark, which served model, which attempt.

Each rollout is its own trace, linked to ``gym.job``, so a rollout trace cannot inherit the job's
context from a parent. ``gym.job`` and every ``gym.rollout.dispatch`` therefore carry it directly,
as span attributes. They are never metric labels.

Every value comes from the resolved config or the collector row; a value that cannot be read
from either is omitted rather than guessed:

* The served model name is ``policy_model_name``, the value Gym's model-server configs send as
  the model.
* A row's benchmark is the ``type: benchmark`` dataset it was prepared from. Rows record only
  their declaring instance (``task_source``) and, when materialized, their taskset, so the name
  is known only for a ``split=benchmark`` run and only when that instance or taskset maps to
  exactly one benchmark dataset.
* The attempt index is the driver's resume counter; the driver leaves it unstamped on the
  first attempt, which is attempt 0.

The run id is not repeated here: it is the ``nemo.run.id`` resource attribute of every span.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from omegaconf.errors import OmegaConfBaseException

from nemo_gym.config.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    POLICY_MODEL_NAME_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
    get_first_server_config_dict,
)
from nemo_gym.core.episode_types import is_materialized_task_row


#: Served model name of the policy model, on ``gym.job`` and ``gym.rollout.dispatch``.
MODEL_NAME_ATTRIBUTE = "nemo.gym.model.name"
#: Benchmark dataset a dispatched row was prepared from, on ``gym.rollout.dispatch``.
BENCHMARK_NAME_ATTRIBUTE = "nemo.gym.benchmark.name"
#: Sorted, distinct benchmark datasets the job dispatches rows from, on ``gym.job``.
BENCHMARK_NAMES_ATTRIBUTE = "nemo.gym.benchmark.names"
#: Resume attempt of a dispatched row, 0 on its first attempt, on ``gym.rollout.dispatch``.
ATTEMPT_INDEX_ATTRIBUTE = "nemo.gym.rollout.attempt.index"

#: Span attribute values this module produces.
SpanAttributes = dict[str, str | int | list[str]]

_BENCHMARK_SPLIT = "benchmark"
_SERVER_KINDS = ("responses_api_agents", "resources_servers")


@dataclass(frozen=True)
class RunContext:
    """What the driver can say about a run from its resolved config.

    Build it with :func:`run_context_from_config`, once per run, only when traces are exported.
    """

    model_name: str | None = None
    benchmark_by_taskset: Mapping[str, str] = field(default_factory=dict)
    benchmark_by_task_source: Mapping[str, str] = field(default_factory=dict)

    def benchmark_name(self, row: Mapping[str, object]) -> str | None:
        """The benchmark dataset *row* was prepared from, or None when it is not known."""
        taskset = _materialized_taskset(row)
        if taskset is not None and taskset in self.benchmark_by_taskset:
            return self.benchmark_by_taskset[taskset]
        task_source = row.get(TASK_SOURCE_KEY_NAME)
        return self.benchmark_by_task_source.get(task_source) if isinstance(task_source, str) else None

    def job_attributes(self, rows: Iterable[Mapping[str, object]]) -> SpanAttributes:
        """Attributes for ``gym.job``, given the rows the job dispatches."""
        attributes: SpanAttributes = {}
        if self.model_name is not None:
            attributes[MODEL_NAME_ATTRIBUTE] = self.model_name
        benchmarks = sorted({name for row in rows if (name := self.benchmark_name(row)) is not None})
        if benchmarks:
            attributes[BENCHMARK_NAMES_ATTRIBUTE] = benchmarks
        return attributes

    def dispatch_attributes(self, row: Mapping[str, object]) -> SpanAttributes:
        """Attributes for one ``gym.rollout.dispatch``."""
        attributes: SpanAttributes = {}
        if self.model_name is not None:
            attributes[MODEL_NAME_ATTRIBUTE] = self.model_name
        benchmark = self.benchmark_name(row)
        if benchmark is not None:
            attributes[BENCHMARK_NAME_ATTRIBUTE] = benchmark
        attempt = row.get(ATTEMPT_INDEX_KEY_NAME, 0)
        if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0:
            attributes[ATTEMPT_INDEX_ATTRIBUTE] = attempt
        return attributes


def run_context_from_config(global_config: Mapping[str, object]) -> RunContext:
    """Read the run context out of a resolved global config. Never raises on a malformed config."""
    model_name = _config_str(global_config, POLICY_MODEL_NAME_KEY_NAME)
    if _config_str(global_config, "split") != _BENCHMARK_SPLIT:
        return RunContext(model_name=model_name)

    by_taskset: dict[str, set[str]] = {}
    by_task_source: dict[str, set[str]] = {}
    for instance_name, name, taskset in _benchmark_datasets(global_config):
        by_task_source.setdefault(instance_name, set()).add(name)
        if taskset is not None:
            by_taskset.setdefault(taskset, set()).add(name)
    return RunContext(
        model_name=model_name,
        benchmark_by_taskset=_unambiguous(by_taskset),
        benchmark_by_task_source=_unambiguous(by_task_source),
    )


def _config_str(config: Mapping[str, object], key: str) -> str | None:
    try:
        value = config.get(key)
    except OmegaConfBaseException:
        # A missing mandatory value or an unresolvable interpolation is an unknown value.
        return None
    return value if isinstance(value, str) and value else None


def _benchmark_datasets(global_config: Mapping[str, object]) -> Iterator[tuple[str, str, str | None]]:
    """``(declaring instance, dataset name, taskset or None)`` for every ``type: benchmark`` dataset."""
    for instance_name in list(global_config.keys()):
        try:
            block = global_config[instance_name]
            if not isinstance(block, Mapping) or not any(kind in block for kind in _SERVER_KINDS):
                continue
            datasets = list(get_first_server_config_dict(global_config, instance_name).get("datasets") or [])
        except (AttributeError, IndexError, TypeError, OmegaConfBaseException):
            continue
        for dataset in datasets:
            if not isinstance(instance_name, str) or not isinstance(dataset, Mapping):
                continue
            name = _config_str(dataset, "name")
            if _config_str(dataset, "type") == _BENCHMARK_SPLIT and name is not None:
                yield instance_name, name, _config_str(dataset, "taskset")


def _unambiguous(names_by_key: Mapping[str, set[str]]) -> dict[str, str]:
    return {key: next(iter(names)) for key, names in names_by_key.items() if len(names) == 1}


def _materialized_taskset(row: Mapping[str, object]) -> str | None:
    if not is_materialized_task_row(row):
        return None
    taskset = row["task_id"].get("taskset")
    return taskset if isinstance(taskset, str) and taskset else None

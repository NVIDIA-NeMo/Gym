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

* A row's served model name is the ``model`` (or ``openai_model``) field of the model server its
  agent's ``model_server`` names, the value that server sends upstream. It is known only when
  the agent the row is dispatched to names one model server and that server sets exactly one
  model name; ``gym.job`` carries it only when every row the job dispatched has the same one.
* A row's benchmark is the ``type: benchmark`` dataset it was prepared from. Rows record only
  their declaring instance (``task_source``) and, when materialized, their taskset, so the name
  is known only for a ``split=benchmark`` run and only when that instance or taskset maps to
  exactly one benchmark dataset. An entry that sets no ``taskset`` declares the taskset of its
  own name, as rollout collection reads it.
* The attempt index is the execution index the rollout store reserved for the dispatch, ``0``
  on a fresh run and one more on each resumed attempt; a row with none is attempt 0.

The run id is not repeated here: it is the ``nemo.run.id`` resource attribute of every span.
"""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field

from omegaconf.errors import OmegaConfBaseException

from nemo_gym.config.global_config import (
    AGENT_REF_KEY_NAME,
    AGENT_SERVER_TYPE_KEY_NAME,
    ATTEMPT_INDEX_KEY_NAME,
    ENVIRONMENT_SERVER_STAMP_KEY_NAME,
    ENVIRONMENT_SERVER_TYPE_KEY_NAME,
    MODEL_SERVER_TYPE_KEY_NAME,
    TASK_SOURCE_KEY_NAME,
    environment_server_attributed_agent,
    get_first_server_config_dict,
)
from nemo_gym.core.episode_types import is_materialized_task_row


#: Served model name of the rollout's agent, on ``gym.rollout.dispatch``, and of the whole job on ``gym.job``.
MODEL_NAME_ATTRIBUTE = "nemo.gym.model.name"
#: Benchmark dataset a dispatched row was prepared from, on ``gym.rollout.dispatch``.
BENCHMARK_NAME_ATTRIBUTE = "nemo.gym.benchmark.name"
#: Sorted, distinct benchmark datasets the job dispatched rows from, on ``gym.job``.
BENCHMARK_NAMES_ATTRIBUTE = "nemo.gym.benchmark.names"
#: Execution index reserved for a dispatched row, 0 on its first attempt, on ``gym.rollout.dispatch``.
ATTEMPT_INDEX_ATTRIBUTE = "nemo.gym.rollout.attempt.index"

#: Span attribute values this module produces.
SpanAttributes = dict[str, str | int | list[str]]

_BENCHMARK_SPLIT = "benchmark"
_SERVER_KINDS = ("responses_api_agents", "resources_servers")
# The fields Gym's model servers send upstream as the model.
_MODEL_NAME_FIELDS = ("model", "openai_model")


@dataclass(frozen=True)
class RunContext:
    """What the driver can say about a run from its resolved config.

    Build it with :func:`run_context_from_config`, once per run, only when traces are exported.
    """

    model_by_agent: Mapping[str, str] = field(default_factory=dict)
    agent_by_environment_server: Mapping[str, str] = field(default_factory=dict)
    benchmark_by_taskset: Mapping[str, str] = field(default_factory=dict)
    benchmark_by_task_source: Mapping[str, str] = field(default_factory=dict)

    def model_name(self, row: Mapping[str, object]) -> str | None:
        """The model the agent *row* is dispatched to is served, or None when it is not known."""
        agent_ref = row.get(AGENT_REF_KEY_NAME)
        agent = agent_ref.get("name") if isinstance(agent_ref, Mapping) else None
        if agent is None:
            # A materialized row names no agent: it runs on the one its environment server attributes results to.
            environment_server = row.get(ENVIRONMENT_SERVER_STAMP_KEY_NAME)
            agent = (
                self.agent_by_environment_server.get(environment_server)
                if isinstance(environment_server, str)
                else None
            )
        return self.model_by_agent.get(agent) if isinstance(agent, str) else None

    def benchmark_name(self, row: Mapping[str, object]) -> str | None:
        """The benchmark dataset *row* was prepared from, or None when it is not known."""
        taskset = _materialized_taskset(row)
        if taskset is not None and taskset in self.benchmark_by_taskset:
            return self.benchmark_by_taskset[taskset]
        task_source = row.get(TASK_SOURCE_KEY_NAME)
        return self.benchmark_by_task_source.get(task_source) if isinstance(task_source, str) else None

    def dispatch_attributes(self, row: Mapping[str, object]) -> SpanAttributes:
        """Attributes for one ``gym.rollout.dispatch``."""
        attributes: SpanAttributes = {}
        model = self.model_name(row)
        if model is not None:
            attributes[MODEL_NAME_ATTRIBUTE] = model
        benchmark = self.benchmark_name(row)
        if benchmark is not None:
            attributes[BENCHMARK_NAME_ATTRIBUTE] = benchmark
        attempt = row.get(ATTEMPT_INDEX_KEY_NAME, 0)
        if isinstance(attempt, int) and not isinstance(attempt, bool) and attempt >= 0:
            attributes[ATTEMPT_INDEX_ATTRIBUTE] = attempt
        return attributes


@dataclass
class JobRunContext:
    """The run context of the rows a job has dispatched, for ``gym.job``: :meth:`add` every batch the job
    dispatches, then set :meth:`attributes` before the span ends. Keeps distinct values, not rows."""

    context: RunContext
    #: Distinct served models of the dispatched rows; None stands for a row whose model is not known.
    models: set[str | None] = field(default_factory=set)
    benchmarks: set[str] = field(default_factory=set)

    def add(self, rows: Iterable[Mapping[str, object]]) -> None:
        for row in rows:
            self.models.add(self.context.model_name(row))
            if (name := self.context.benchmark_name(row)) is not None:
                self.benchmarks.add(name)

    def attributes(self) -> SpanAttributes:
        attributes: SpanAttributes = {}
        if len(self.models) == 1 and (model := next(iter(self.models))) is not None:
            attributes[MODEL_NAME_ATTRIBUTE] = model
        if self.benchmarks:
            attributes[BENCHMARK_NAMES_ATTRIBUTE] = sorted(self.benchmarks)
        return attributes


def run_context_from_config(
    global_config: Mapping[str, object], *, collector_config: Mapping[str, object] | None = None
) -> RunContext:
    """Read the run context out of the running servers' resolved config. Never raises on a malformed config.

    ``collector_config`` is the collector's own config, whose ``split`` is the split the run dispatches: with
    ``--no-serve`` the servers were started with a config of their own. Without it the split is ``global_config``'s.
    """
    model_by_agent = _models_by_agent(global_config)
    agent_by_environment_server = _agents_by_environment_server(global_config)
    split_config = global_config if collector_config is None else collector_config
    if _config_str(split_config, "split") != _BENCHMARK_SPLIT:
        return RunContext(model_by_agent=model_by_agent, agent_by_environment_server=agent_by_environment_server)

    by_taskset: dict[str, set[str]] = {}
    by_task_source: dict[str, set[str]] = {}
    for instance_name, name, taskset in _benchmark_datasets(global_config):
        by_task_source.setdefault(instance_name, set()).add(name)
        # An entry that sets no taskset declares the one of its own name, as a Harbor dataset's generated entry does.
        by_taskset.setdefault(taskset or name, set()).add(name)
    return RunContext(
        model_by_agent=model_by_agent,
        agent_by_environment_server=agent_by_environment_server,
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


def _server_blocks(global_config: Mapping[str, object], kind: str) -> Iterator[tuple[str, Mapping[str, object]]]:
    """``(instance name, server config)`` for every instance of *kind* that holds exactly one server."""
    for instance_name in list(global_config.keys()):
        try:
            block = global_config[instance_name]
            servers = block.get(kind) if isinstance(block, Mapping) else None
            if not isinstance(instance_name, str) or not isinstance(servers, Mapping) or len(servers) != 1:
                continue
            server = next(iter(servers.values()))
        except (AttributeError, TypeError, OmegaConfBaseException):
            continue
        if isinstance(server, Mapping):
            yield instance_name, server


def _models_by_agent(global_config: Mapping[str, object]) -> dict[str, str]:
    """The served model of every agent whose ``model_server`` sets exactly one model name."""
    served = {}
    for name, server in _server_blocks(global_config, MODEL_SERVER_TYPE_KEY_NAME):
        models = {model for key in _MODEL_NAME_FIELDS if (model := _config_str(server, key)) is not None}
        if len(models) == 1:
            served[name] = next(iter(models))
    models_by_agent = {}
    for name, agent in _server_blocks(global_config, AGENT_SERVER_TYPE_KEY_NAME):
        try:
            reference = agent.get("model_server")
        except OmegaConfBaseException:
            continue
        if not isinstance(reference, Mapping) or _config_str(reference, "type") != MODEL_SERVER_TYPE_KEY_NAME:
            continue
        model = served.get(_config_str(reference, "name"))
        if model is not None:
            models_by_agent[name] = model
    return models_by_agent


def _agents_by_environment_server(global_config: Mapping[str, object]) -> dict[str, str]:
    agents = {}
    for name, server in _server_blocks(global_config, ENVIRONMENT_SERVER_TYPE_KEY_NAME):
        try:
            agent = environment_server_attributed_agent(server)
        except (AttributeError, TypeError, OmegaConfBaseException):
            continue
        if isinstance(agent, str) and agent:
            agents[name] = agent
    return agents


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

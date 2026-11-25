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
"""Run context read from a resolved config: known values become attributes, unknown ones are omitted.

Pure config and row logic, so no nemo-lens is needed. The spans that carry these attributes are
covered in ``test_call_sites.py``.
"""

import pytest
from omegaconf import OmegaConf

from nemo_gym.telemetry.run_context import JobRunContext, RunContext, run_context_from_config


def _agent(*datasets: dict) -> dict:
    return {"responses_api_agents": {"simple_agent": {"datasets": list(datasets)}}}


def _benchmark(name: str, taskset: str | None = None) -> dict:
    dataset = {"name": name, "type": "benchmark"}
    if taskset is not None:
        dataset["taskset"] = taskset
    return dataset


_FLAT_ROW = {"task_source": "aime_agent"}


def _materialized_row(taskset: str) -> dict:
    return {"task_source": "aime_agent", "task_id": {"taskset": taskset, "task_id": "t0"}, "task_input": {}}


@pytest.mark.parametrize(
    ("config", "row", "expected_benchmark"),
    [
        pytest.param(
            {"split": "benchmark", "aime_agent": _agent(_benchmark("aime24"))},
            _FLAT_ROW,
            "aime24",
            id="instance-with-one-benchmark",
        ),
        pytest.param(
            {
                "split": "benchmark",
                "aime_server": {"resources_servers": {"math": {"datasets": [_benchmark("aime24")]}}},
            },
            {"task_source": "aime_server"},
            "aime24",
            id="declared-by-a-resources-server",
        ),
        pytest.param(
            {"aime_agent": _agent(_benchmark("aime24"))},
            _FLAT_ROW,
            None,
            id="no-split-is-unknown",
        ),
        pytest.param(
            {"split": "train", "aime_agent": _agent(_benchmark("aime24"), {"name": "aime_train", "type": "train"})},
            _FLAT_ROW,
            None,
            id="train-split-is-not-a-benchmark",
        ),
        pytest.param(
            {"split": "benchmark", "aime_agent": _agent(_benchmark("aime24"), _benchmark("aime25"))},
            _FLAT_ROW,
            None,
            id="instance-with-two-benchmarks-is-ambiguous",
        ),
        pytest.param(
            {"split": "benchmark", "aime_agent": _agent({"name": "aime_train", "type": "train"})},
            _FLAT_ROW,
            None,
            id="instance-without-a-benchmark",
        ),
        pytest.param(
            {"split": "benchmark", "aime_agent": _agent(_benchmark("aime24"))},
            {"agent_ref": {"name": "aime_agent"}},
            None,
            id="row-without-task-source",
        ),
        pytest.param(
            {"split": "benchmark", "aime_agent": _agent(_benchmark("aime24", "aime"), _benchmark("aime25", "aime25"))},
            _materialized_row("aime25"),
            "aime25",
            id="materialized-row-by-taskset",
        ),
        pytest.param(
            {
                "split": "benchmark",
                "a": _agent(_benchmark("aime24", "aime")),
                "b": _agent(_benchmark("aime25", "aime")),
            },
            {"task_id": {"taskset": "aime", "task_id": "t0"}, "task_input": {}},
            None,
            id="taskset-shared-by-two-benchmarks-is-ambiguous",
        ),
        pytest.param(
            {"split": "benchmark", "aime_agent": {"responses_api_agents": {}}, "other": _agent(_benchmark("x"))},
            _FLAT_ROW,
            None,
            id="malformed-instance-is-skipped",
        ),
    ],
)
def test_benchmark_name_is_known_only_when_the_config_names_exactly_one(config, row, expected_benchmark):
    context = run_context_from_config(OmegaConf.create(config))

    assert context.benchmark_name(row) == expected_benchmark


def test_a_harbor_benchmark_is_named_by_its_generated_dataset_entry(tmp_path):
    """The entry `gym eval run` generates for a Harbor dataset is named after its taskset and sets no `taskset`."""
    from nemo_gym.data.tasks.harbor.materialize import agent_instance, dataset_entry

    taskset = "terminal_bench_2_1"
    config = {
        "split": "benchmark",
        agent_instance(taskset): _agent(dataset_entry(taskset, tmp_path / "rows.jsonl")),
    }
    # A Harbor row's task_source is its dataset folder's name, not an instance.
    row = {"task_source": taskset, "task_id": {"taskset": taskset, "task_id": "t0"}, "task_input": {}}

    assert run_context_from_config(OmegaConf.create(config)).benchmark_name(row) == taskset


@pytest.mark.parametrize(
    ("config", "expected_model"),
    [
        pytest.param({"policy_model_name": "gpt-oss-120b"}, "gpt-oss-120b", id="set"),
        pytest.param({}, None, id="absent"),
        pytest.param({"policy_model_name": ""}, None, id="empty"),
        pytest.param({"policy_model_name": "???"}, None, id="missing-mandatory-value"),
        pytest.param({"policy_model_name": "${nowhere}"}, None, id="unresolvable-interpolation"),
        pytest.param({"policy_model_name": 7}, None, id="not-a-string"),
    ],
)
def test_model_name_is_policy_model_name_when_it_resolves(config, expected_model):
    assert run_context_from_config(OmegaConf.create(config)).model_name == expected_model


_KNOWN = RunContext(model_name="m", benchmark_by_task_source={"aime_agent": "aime24"})


@pytest.mark.parametrize(
    ("context", "row", "expected"),
    [
        pytest.param(
            _KNOWN,
            _FLAT_ROW,
            {"nemo.gym.model.name": "m", "nemo.gym.benchmark.name": "aime24", "nemo.gym.rollout.attempt.index": 0},
            id="first-attempt-is-0",
        ),
        pytest.param(
            _KNOWN,
            {**_FLAT_ROW, "_ng_attempt_index": 2},
            {"nemo.gym.model.name": "m", "nemo.gym.benchmark.name": "aime24", "nemo.gym.rollout.attempt.index": 2},
            id="resumed-attempt",
        ),
        pytest.param(RunContext(), {}, {"nemo.gym.rollout.attempt.index": 0}, id="nothing-else-known"),
        pytest.param(RunContext(), {"_ng_attempt_index": -1}, {}, id="negative-attempt-omitted"),
        pytest.param(RunContext(), {"_ng_attempt_index": True}, {}, id="bool-attempt-omitted"),
        pytest.param(RunContext(), {"_ng_attempt_index": "1"}, {}, id="string-attempt-omitted"),
    ],
)
def test_dispatch_attributes(context, row, expected):
    assert context.dispatch_attributes(row) == expected


@pytest.mark.parametrize(
    ("context", "batches", "expected"),
    [
        pytest.param(
            RunContext(model_name="m", benchmark_by_task_source={"a": "aime24", "b": "aime25"}),
            [[{"task_source": "b"}, {"task_source": "a"}], [{"task_source": "b"}, {"task_source": "unknown"}]],
            {"nemo.gym.model.name": "m", "nemo.gym.benchmark.names": ["aime24", "aime25"]},
            id="sorted-distinct-benchmarks",
        ),
        pytest.param(RunContext(), [[{"task_source": "a"}]], {}, id="nothing-known"),
        pytest.param(_KNOWN, [], {"nemo.gym.model.name": "m"}, id="nothing-dispatched"),
    ],
)
def test_job_attributes_name_the_dispatched_rows(context, batches, expected):
    job = JobRunContext(context)
    for batch in batches:
        job.add(batch)

    assert job.attributes() == expected

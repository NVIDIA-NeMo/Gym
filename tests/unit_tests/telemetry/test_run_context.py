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

from pathlib import Path

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


_FLAT_ROW = {"agent_ref": {"name": "aime_agent"}, "task_source": "aime_agent"}


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


_REPO = Path(__file__).resolve().parents[3]


def _agent_calling(model_server: str = "policy_model", **reference) -> dict:
    model_ref = {"type": "responses_api_models", "name": model_server} | reference
    return {"aime_agent": {"responses_api_agents": {"simple_agent": {"model_server": model_ref}}}}


def _model_server(**fields) -> dict:
    return {"policy_model": {"responses_api_models": {"some_model": fields}}}


@pytest.mark.parametrize(
    ("config", "expected_model"),
    [
        pytest.param({**_agent_calling(), **_model_server(model="gpt-oss-120b")}, "gpt-oss-120b", id="model"),
        pytest.param(
            {**_agent_calling(), **_model_server(openai_model="gpt-4.1")}, "gpt-4.1", id="openai-model-field"
        ),
        pytest.param(
            {**_agent_calling(), **_model_server(model="${policy_model_name}"), "policy_model_name": "qwen3"},
            "qwen3",
            id="interpolated-policy-model-name",
        ),
        pytest.param({**_agent_calling(), **_model_server(model="m", openai_model="m")}, "m", id="both-fields-agree"),
        pytest.param({**_agent_calling(), **_model_server(model="a", openai_model="b")}, None, id="two-model-names"),
        pytest.param({**_agent_calling(), **_model_server()}, None, id="model-server-sets-no-model"),
        pytest.param({**_agent_calling(), **_model_server(model="")}, None, id="empty"),
        pytest.param({**_agent_calling(), **_model_server(model="???")}, None, id="missing-mandatory-value"),
        pytest.param({**_agent_calling(), **_model_server(model="${nowhere}")}, None, id="unresolvable-interpolation"),
        pytest.param({**_agent_calling(), **_model_server(model=7)}, None, id="not-a-string"),
        pytest.param({**_agent_calling("elsewhere"), **_model_server(model="m")}, None, id="unknown-model-server"),
        pytest.param(
            {**_agent_calling(type="resources_servers"), **_model_server(model="m")}, None, id="not-a-model-server"
        ),
        pytest.param(
            {
                **_agent_calling(),
                "policy_model": {"responses_api_models": {"a": {"model": "m"}, "b": {"model": "m"}}},
            },
            None,
            id="two-model-implementations",
        ),
        pytest.param(
            {"aime_agent": _agent(), **_model_server(model="m"), "policy_model_name": "m"},
            None,
            id="agent-without-model-server",
        ),
    ],
)
def test_model_name_is_the_one_the_agents_model_server_sends(config, expected_model):
    row = {"agent_ref": {"name": "aime_agent"}}

    assert run_context_from_config(OmegaConf.create(config)).model_name(row) == expected_model


def test_a_shipped_policy_config_that_sets_its_model_directly_wins_over_policy_model_name():
    """HealthBench's Claude Opus 5 policy names its model on the server, so `policy_model_name` is not what it sends."""
    policy = OmegaConf.load(_REPO / "benchmarks/healthbench/configs/claude-opus-5.yaml")
    config = OmegaConf.merge(policy, _agent_calling(), {"policy_model_name": "an-env-yaml-default"})

    context = run_context_from_config(config)

    assert context.model_name({"agent_ref": {"name": "aime_agent"}}) == "aws/anthropic/bedrock-claude-opus-5"


def test_a_shipped_model_server_config_sends_policy_model_name():
    model = OmegaConf.load(_REPO / "responses_api_models/vllm_model/configs/vllm_model.yaml")
    config = OmegaConf.merge(model, _agent_calling(), {"policy_model_name": "nemotron"})

    assert run_context_from_config(config).model_name({"agent_ref": {"name": "aime_agent"}}) == "nemotron"


def test_a_row_routed_by_its_environment_server_runs_on_the_agent_that_server_names():
    config = {
        **_agent_calling(),
        **_model_server(model="m"),
        "harbor_env": {"environment_servers": {"harbor": {"agent_server": {"name": "aime_agent"}}}},
    }
    row = {"_ng_environment_server": "harbor", "task_id": {"taskset": "t", "task_id": "0"}, "task_input": {}}

    context = run_context_from_config(OmegaConf.create(config))

    assert context.model_name(row) is None
    assert context.model_name(row | {"_ng_environment_server": "harbor_env"}) == "m"


_KNOWN = RunContext(model_by_agent={"aime_agent": "m"}, benchmark_by_task_source={"aime_agent": "aime24"})


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


def _row_on(agent: str, task_source: str = "a") -> dict:
    return {"agent_ref": {"name": agent}, "task_source": task_source}


_TWO_AGENTS = RunContext(
    model_by_agent={"x": "m", "y": "m", "z": "other"}, benchmark_by_task_source={"a": "aime24", "b": "aime25"}
)


@pytest.mark.parametrize(
    ("context", "batches", "expected"),
    [
        pytest.param(
            _TWO_AGENTS,
            [[_row_on("x", "b"), _row_on("y", "a")], [_row_on("x", "b"), _row_on("y", "unknown")]],
            {"nemo.gym.model.name": "m", "nemo.gym.benchmark.names": ["aime24", "aime25"]},
            id="one-model-and-sorted-distinct-benchmarks",
        ),
        pytest.param(
            _TWO_AGENTS, [[_row_on("x")], [_row_on("z")]], {"nemo.gym.benchmark.names": ["aime24"]}, id="two-models"
        ),
        pytest.param(
            _TWO_AGENTS,
            [[_row_on("x"), _row_on("unknown_agent")]],
            {"nemo.gym.benchmark.names": ["aime24"]},
            id="a-row-whose-model-is-unknown",
        ),
        pytest.param(RunContext(), [[{"task_source": "a"}]], {}, id="nothing-known"),
        pytest.param(_KNOWN, [], {}, id="nothing-dispatched"),
    ],
)
def test_job_attributes_name_the_dispatched_rows(context, batches, expected):
    job = JobRunContext(context)
    for batch in batches:
        job.add(batch)

    assert job.attributes() == expected

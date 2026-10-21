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

import asyncio

import pytest
from pydantic import ValidationError

import nemo_gym.harness_optimization as harness_optimization
from nemo_gym.agent_harness_optimization import (
    AGENT_MODULE_REFS_KEY_NAME,
    HarnessEvaluator,
    SystemPromptCandidate,
    SystemPromptConfigAdapter,
)
from nemo_gym.harness_optimization import HarnessOptimizationRecipe
from nemo_gym.prompt import PromptConfig
from nemo_gym.reference_harness_optimizer import (
    CandidateSweepHarnessOptimizer,
    CandidateSweepOptimizerConfig,
)
from nemo_gym.rollout_observability import TrajectoryRecord


class FakeRolloutCollectionHelper:
    def run_examples(self, rows, head_server_config=None, semaphore=None):
        async def completed(row):
            task_index = row["_ng_task_index"]
            rollout_index = row["_ng_rollout_index"]
            trajectory = TrajectoryRecord(
                task_id=str(task_index),
                rollout_id=f"{task_index}-{rollout_index}",
                turns=[],
                tool_calls=[],
            )
            return row, {
                "reward": float(rollout_index),
                "response": {"ok": True},
                "ng_trajectory": trajectory.model_dump(mode="json"),
            }

        return [completed(row) for row in reversed(rows)]


class FakeHarnessEvaluator:
    def __init__(self, num_repeats=2):
        self.candidates = []
        self.num_repeats = num_repeats

    async def evaluate(self, candidate, *, cohort="train", example_indices=None):
        self.candidates.append(candidate)
        reward = float("test every constraint" in candidate.system.lower())
        return [
            {
                "reward": reward,
                "_ng_task_index": task_index,
                "_ng_rollout_index": rollout_index,
                "ng_trajectory": {"turns": []},
            }
            for task_index in (example_indices or [0])
            for rollout_index in range(self.num_repeats)
        ]


def optimization_recipe(**optimizer_config):
    return HarnessOptimizationRecipe.model_validate(
        {
            "train_jsonl_fpath": "rows.jsonl",
            "rollout": {
                "agent_name": "simple_agent",
                "head_server": {"host": "127.0.0.1", "port": 11000},
                "num_repeats": 2,
            },
            "target": {
                "type": "prompt_config.system",
                "module_name": "reasoning_prompt",
                "prompt_config": {"user": "{question}", "system": "Solve carefully."},
            },
            "optimizer": {"type": "candidate_sweep", **optimizer_config},
        }
    )


def test_candidate_is_canonical_and_content_addressed():
    candidate = SystemPromptCandidate.from_system(system="Solve carefully.\r\n")

    assert candidate.system == "Solve carefully."
    assert len(candidate.sha256) == 64

    with pytest.raises(ValidationError, match="sha256 does not match"):
        SystemPromptCandidate.model_validate(candidate.model_dump() | {"sha256": "0" * 64})


def test_system_prompt_adapter_stamps_module_refs_without_mutating_source():
    source = {
        "question": "What is 17 * 23?",
        "responses_create_params": {"temperature": 0.1},
        AGENT_MODULE_REFS_KEY_NAME: [{"name": "fixed_skill"}],
    }
    candidate = SystemPromptCandidate.from_system(system="Use a calculator.")
    adapter = SystemPromptConfigAdapter(
        module_name="answer_format_prompt",
        base_prompt=PromptConfig(user="{question}"),
    )

    applied = adapter.apply(candidate, source)

    assert "input" not in source["responses_create_params"]
    assert applied["responses_create_params"]["temperature"] == 0.1
    assert applied[AGENT_MODULE_REFS_KEY_NAME][0]["name"] == "answer_format_prompt"
    assert applied[AGENT_MODULE_REFS_KEY_NAME][0]["sha256"] == candidate.sha256
    assert applied[AGENT_MODULE_REFS_KEY_NAME][1] == {"name": "fixed_skill"}
    assert source[AGENT_MODULE_REFS_KEY_NAME] == [{"name": "fixed_skill"}]


def test_evaluator_returns_native_rollout_records():
    candidate = SystemPromptCandidate.from_system(system="Use a calculator.")
    adapter = SystemPromptConfigAdapter(
        module_name="answer_format_prompt",
        base_prompt=PromptConfig(user="{question}"),
    )
    evaluator = HarnessEvaluator(
        train_rows=[{"question": "What is 17 * 23?", "responses_create_params": {}}],
        config_adapter=adapter,
        agent_name="simple_agent",
        helper=FakeRolloutCollectionHelper(),
    )

    rollouts = asyncio.run(evaluator.evaluate(candidate))

    assert rollouts[0]["reward"] == 0.0
    assert rollouts[0][AGENT_MODULE_REFS_KEY_NAME][0]["sha256"] == candidate.sha256
    assert rollouts[0]["ng_trajectory"]["schema_version"] == "1.0"
    assert evaluator.freeze(candidate) == PromptConfig(user="{question}", system="Use a calculator.")


def test_optimizer_update_creates_new_candidate():
    candidate = SystemPromptCandidate.from_system(system="Use a calculator.")

    updated = SystemPromptCandidate.from_system(
        system=f"{candidate.system}\nBefore calculating, verify the requested operator.",
        parent_sha256=candidate.sha256,
    )

    assert updated.parent_sha256 == candidate.sha256
    assert updated.sha256 != candidate.sha256


def test_evaluator_uses_rollout_helper_and_restores_requested_order():
    candidate = SystemPromptCandidate.from_system(system="Use a calculator.")
    adapter = SystemPromptConfigAdapter(
        module_name="answer_format_prompt",
        base_prompt=PromptConfig(user="{question}"),
    )
    evaluator = HarnessEvaluator(
        train_rows=[{"question": "What is 17 * 23?", "responses_create_params": {}}],
        config_adapter=adapter,
        num_repeats=2,
        agent_name="simple_agent",
        helper=FakeRolloutCollectionHelper(),
    )

    results = asyncio.run(evaluator.evaluate(candidate))

    assert [result["reward"] for result in results] == [0.0, 1.0]
    assert results[0]["agent_ref"] == {"name": "simple_agent"}
    assert results[0][AGENT_MODULE_REFS_KEY_NAME][0]["sha256"] == candidate.sha256


def test_evaluator_binds_cohorts_and_preserves_selected_example_identity():
    candidate = SystemPromptCandidate.from_system(system="Use a calculator.")
    evaluator = HarnessEvaluator(
        train_rows=[
            {"question": "train 0", "responses_create_params": {}},
            {"question": "train 1", "responses_create_params": {}},
        ],
        validation_rows=[{"question": "validation 0", "responses_create_params": {}}],
        config_adapter=SystemPromptConfigAdapter(
            module_name="answer_format_prompt",
            base_prompt=PromptConfig(user="{question}"),
        ),
        agent_name="simple_agent",
        helper=FakeRolloutCollectionHelper(),
    )

    train_result = asyncio.run(evaluator.evaluate(candidate, example_indices=[1]))
    validation_result = asyncio.run(evaluator.evaluate(candidate, cohort="validation"))
    exposed_rows = evaluator.examples()
    exposed_rows[0]["question"] = "mutated"

    assert train_result[0]["_ng_task_index"] == 1
    assert validation_result[0]["_ng_task_index"] == 0
    assert evaluator.examples()[0]["question"] == "train 0"

    with pytest.raises(ValueError, match="must not contain duplicates"):
        asyncio.run(evaluator.evaluate(candidate, example_indices=[0, 0]))


def test_reference_optimizer_conforms_to_shared_interface(tmp_path):
    evaluator = FakeHarnessEvaluator()
    optimizer = CandidateSweepHarnessOptimizer(
        CandidateSweepOptimizerConfig(
            candidate_systems=["Test every constraint before answering."],
            output_dir=str(tmp_path),
        )
    )
    initial = SystemPromptCandidate.from_system(system="Solve carefully.")

    selected = asyncio.run(optimizer.optimize(initial, evaluator))

    assert len(evaluator.candidates) == 2
    assert selected.system == "Test every constraint before answering."
    assert optimizer.iterations[1].score == 1.0
    assert optimizer.iterations[0].feedback["num_with_trajectory"] == 2
    assert (tmp_path / "iteration_000.json").is_file()
    assert (tmp_path / "iteration_001.json").is_file()


def test_workflow_runs_optimizer_and_freezes_selected_candidate(monkeypatch):
    recipe = optimization_recipe()
    captured = {}

    class FakeEvaluator:
        def freeze(self, candidate):
            captured["frozen"] = candidate
            return PromptConfig(user="{question}", system=candidate.system)

    class FakeOptimizer:
        iterations = ()

        async def optimize(self, initial_candidate, evaluator):
            captured["initial"] = initial_candidate
            captured["evaluator"] = evaluator
            return SystemPromptCandidate.from_system(system="Selected.")

    evaluator = FakeEvaluator()
    monkeypatch.setattr(harness_optimization, "load_source_rows", lambda path, limit: [{"question": "test"}])
    monkeypatch.setattr(
        harness_optimization,
        "build_system_prompt_evaluator",
        lambda recipe, train_rows, validation_rows: evaluator,
    )
    monkeypatch.setattr(harness_optimization, "build_optimizer", lambda recipe: FakeOptimizer())

    result = asyncio.run(harness_optimization.run_harness_optimization(recipe))

    assert result["selected_candidate"]["system"] == "Selected."
    assert result["frozen_artifact"] == {"user": "{question}", "system": "Selected."}
    assert captured["initial"].system == "Solve carefully."
    assert captured["evaluator"] is evaluator
    assert captured["frozen"].system == "Selected."

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from asyncio import Future
from pathlib import Path

import orjson
import pytest

from nemo_gym.prompt import FewShotExamplesConfig, Prompt, PromptConfig, load_prompt
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


class TestRolloutCollection:
    def test_preprocess_rows_from_config(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(10)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath="abcd",
            limit=3,
            num_repeats=2,
            num_repeats_add_seed=True,
            num_samples_in_parallel=None,
            responses_create_params=dict(temperature=0.1),
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert rows == [
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1},
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1},
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "responses_create_params": {"input": [], "seed": 0, "temperature": 0.1},
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "responses_create_params": {"input": [], "seed": 1, "temperature": 0.1},
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
        ]

    def test_prompt_config_field(self) -> None:
        config = RolloutCollectionConfig(
            agent_name="agent",
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
            prompt_config="prompt_configs/math.yaml",
        )
        assert config.prompt_config == "prompt_configs/math.yaml"

    def test_prompt_config_default_none(self) -> None:
        config = RolloutCollectionConfig(
            agent_name="agent",
            input_jsonl_fpath="input.jsonl",
            output_jsonl_fpath="output.jsonl",
        )
        assert config.prompt_config is None

    async def test_run_from_config_sanity(self, tmp_path: Path) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(10)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=2,
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def run_examples(
                self,
                examples: list[dict],
                *args,
                **kwargs,
            ):
                futures = []
                for example in examples:
                    future = Future()
                    # (row, result)
                    future.set_result((example, {"response": {"usage": {"abc usage": 1}}}))
                    futures.append(future)

                return futures

        actual_returned_results = await TestRolloutCollectionHelper().run_from_config(config)

        expected_results = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 2, "_ng_rollout_index": 0, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 2, "_ng_rollout_index": 1, "response": {"usage": {"abc usage": 1}}},
        ]

        assert expected_results == actual_returned_results

        expected_materialized_inputs_len = 6
        with (tmp_path / "output_materialized_inputs.jsonl").open() as f:
            actual_materialized_inputs_len = len(list(f))
        assert expected_materialized_inputs_len == actual_materialized_inputs_len

        with output_jsonl_fpath.open() as f:
            actual_written_results = [json.loads(line) for line in f]
        assert expected_results == actual_written_results

        metrics_fpath = tmp_path / "output_metrics.json"
        assert metrics_fpath.exists()
        metrics_data = json.loads(metrics_fpath.read_text())
        assert "aggregate" in metrics_data
        assert "per_sample_aggregate" in metrics_data
        assert "per_task" in metrics_data
        assert "usage" in metrics_data
        assert metrics_data["usage"]["abc usage"] == 1.0

    async def test_run_from_config_sorted(self, tmp_path: Path) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "my agent name"}, "x": i})
            for i in range(10)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=2,
        )

        class TestRolloutCollectionHelper(RolloutCollectionHelper):
            def run_examples(
                self,
                examples: list[dict],
                *args,
                **kwargs,
            ):
                futures = []
                for example in examples:
                    future = Future()
                    # (row, result)
                    future.set_result((example, {"response": {"usage": {"abc usage": 1}}}))
                    futures.append(future)

                # Reverse!
                futures = reversed(futures)

                return futures

        actual_returned_results = await TestRolloutCollectionHelper().run_from_config(config)

        expected_results = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 2, "_ng_rollout_index": 0, "response": {"usage": {"abc usage": 1}}},
            {"_ng_task_index": 2, "_ng_rollout_index": 1, "response": {"usage": {"abc usage": 1}}},
        ]

        assert expected_results == actual_returned_results

    def test_load_from_cache(self, tmp_path: Path) -> None:
        input_jsonl_fpath = tmp_path / "input.jsonl"
        materialized_inputs_jsonl_fpath = tmp_path / "output_materialized_inputs.jsonl"

        materialized_inputs = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "input": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "input": True},
            {"_ng_task_index": 2, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 2, "_ng_rollout_index": 1, "input": True},
        ]
        materialized_inputs_jsonl_fpath.write_bytes(b"\n".join(map(orjson.dumps, materialized_inputs)) + b"\n")

        outputs = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "output": True},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "output": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "output": True},
        ]
        output_jsonl_fpath = tmp_path / "output.jsonl"
        output_jsonl_fpath.write_bytes(b"\n".join(map(orjson.dumps, outputs)) + b"\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            limit=3,
            num_repeats=2,
        )

        actual_returned_results = RolloutCollectionHelper()._load_from_cache(config)

        expected_results = (
            [
                {"_ng_task_index": 1, "_ng_rollout_index": 0, "input": True},
                {"_ng_task_index": 2, "_ng_rollout_index": 0, "input": True},
                {"_ng_task_index": 2, "_ng_rollout_index": 1, "input": True},
            ],
            [
                {"_ng_task_index": 0, "_ng_rollout_index": 0, "input": True},
                {"_ng_task_index": 0, "_ng_rollout_index": 1, "input": True},
                {"_ng_task_index": 1, "_ng_rollout_index": 1, "input": True},
            ],
            [
                {"_ng_task_index": 0, "_ng_rollout_index": 0, "output": True},
                {"_ng_task_index": 0, "_ng_rollout_index": 1, "output": True},
                {"_ng_task_index": 1, "_ng_rollout_index": 1, "output": True},
            ],
            [
                [orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": 0, "output": True})],
                [orjson.dumps({"_ng_task_index": 0, "_ng_rollout_index": 1, "output": True})],
                [orjson.dumps({"_ng_task_index": 1, "_ng_rollout_index": 1, "output": True})],
            ],
        )

        assert expected_results == actual_returned_results


class TestPrompt:
    def test_fill_system_and_user(self) -> None:
        config = PromptConfig(system="You are a math tutor.", user="Solve: {problem}")
        prompt = Prompt(config)
        messages = prompt.fill({"problem": "2+2"})
        assert messages == [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve: 2+2"},
        ]

    def test_fill_user_only(self) -> None:
        config = PromptConfig(user="{problem}")
        prompt = Prompt(config)
        messages = prompt.fill({"problem": "What is 5*3?"})
        assert messages == [{"role": "user", "content": "What is 5*3?"}]

    def test_fill_with_few_shot_examples(self) -> None:
        config = PromptConfig(
            user="Solve: {problem}",
            few_shot_examples=FewShotExamplesConfig(
                prefix="Here are some examples:\n",
                template="Q: {question}\nA: {answer}\n",
                suffix="\nNow solve the following:\n",
                examples=[
                    {"question": "1+1", "answer": "2"},
                    {"question": "2+3", "answer": "5"},
                ],
            ),
        )
        prompt = Prompt(config)
        messages = prompt.fill({"problem": "3+4"})
        assert len(messages) == 1
        content = messages[0]["content"]
        assert content.startswith("Here are some examples:\n")
        assert "Q: 1+1\nA: 2\n" in content
        assert "Q: 2+3\nA: 5\n" in content
        assert content.endswith("Solve: 3+4")

    def test_fill_missing_field_raises_key_error(self) -> None:
        config = PromptConfig(user="{problem}")
        prompt = Prompt(config)
        with pytest.raises(KeyError):
            prompt.fill({"other_field": "value"})

    def test_load_prompt_from_yaml(self, tmp_path) -> None:
        yaml_content = 'system: "Be helpful."\nuser: "{question}"'
        yaml_file = tmp_path / "test_prompt.yaml"
        yaml_file.write_text(yaml_content)

        prompt = load_prompt(str(yaml_file))
        messages = prompt.fill({"question": "Hello?"})
        assert messages == [
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello?"},
        ]

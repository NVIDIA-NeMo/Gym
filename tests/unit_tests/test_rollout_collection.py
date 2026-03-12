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
import yaml

from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper


class TestRolloutCollection:
    def test_preprocess_rows_with_prompt_config(self, tmp_path: Path) -> None:
        """prompt_config builds responses_create_params.input from template."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"system": "You are a math tutor.", "user": "Solve: {question}"}))

        fpath = tmp_path / "input.jsonl"
        rows = [
            {"question": "What is 2+2?", "expected_answer": "4"},
            {"question": "What is 3*5?", "expected_answer": "15"},
        ]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            prompt_config=str(prompt_path),
            num_repeats=1,
        )

        result = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        assert len(result) == 2
        assert result[0]["responses_create_params"]["input"] == [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve: What is 2+2?"},
        ]
        assert result[0]["expected_answer"] == "4"
        assert result[1]["responses_create_params"]["input"][1]["content"] == "Solve: What is 3*5?"

    def test_preprocess_rows_prompt_config_rejects_prebaked(self, tmp_path: Path) -> None:
        """prompt_config raises when rows already have responses_create_params.input."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{question}"}))

        fpath = tmp_path / "input.jsonl"
        rows = [{"question": "test", "responses_create_params": {"input": [{"role": "user", "content": "baked"}]}}]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            prompt_config=str(prompt_path),
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)

    def test_preprocess_rows_prompt_config_preserves_rcp_fields(self, tmp_path: Path) -> None:
        """prompt_config preserves other responses_create_params fields like tools."""
        prompt_path = tmp_path / "prompt.yaml"
        prompt_path.write_text(yaml.dump({"user": "{question}"}))

        fpath = tmp_path / "input.jsonl"
        rows = [{"question": "test", "responses_create_params": {"tools": [{"type": "function", "name": "calc"}]}}]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            prompt_config=str(prompt_path),
            num_repeats=1,
        )

        result = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert result[0]["responses_create_params"]["tools"] == [{"type": "function", "name": "calc"}]
        assert result[0]["responses_create_params"]["input"] == [{"role": "user", "content": "test"}]

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

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                """Compute aggregate metrics locally (no server needed)."""
                stripped = [{k: v for k, v in r.items() if k not in ("responses_create_params",)} for r in results]
                agg = compute_aggregate_metrics(stripped)
                metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
                metrics_fpath.write_bytes(
                    orjson.dumps(
                        [{"agent_ref": {"name": "my agent name"}, **agg.model_dump()}], option=orjson.OPT_INDENT_2
                    )
                )
                return metrics_fpath

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

        aggregate_metrics_fpath = tmp_path / "output_aggregate_metrics.json"
        actual_aggregate_metrics = json.loads(aggregate_metrics_fpath.read_text())
        expected_aggregate_metrics = [
            {
                "agent_ref": {"name": "my agent name"},
                "agent_metrics": {
                    "mean/abc usage": 1.0,
                    "max/abc usage": 1,
                    "min/abc usage": 1,
                    "median/abc usage": 1.0,
                    "std/abc usage": 0.0,
                },
                "key_metrics": {"mean/abc usage": 1.0},
                "group_level_metrics": actual_aggregate_metrics[0]["group_level_metrics"],
            }
        ]
        assert expected_aggregate_metrics == actual_aggregate_metrics

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

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

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

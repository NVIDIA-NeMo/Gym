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
import urllib.error
from asyncio import Future
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
import yaml

from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, _fetch_gym_tasks


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
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
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
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "response": {"usage": {"abc usage": 1}},
                "agent_ref": {"name": "my agent name"},
            },
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

        actual_input_rows, actual_rows = RolloutCollectionHelper()._load_from_cache(config)

        expected_input_rows = [
            {"_ng_task_index": 1, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 2, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 2, "_ng_rollout_index": 1, "input": True},
        ]
        expected_rows = [
            {"_ng_task_index": 0, "_ng_rollout_index": 0, "input": True},
            {"_ng_task_index": 0, "_ng_rollout_index": 1, "input": True},
            {"_ng_task_index": 1, "_ng_rollout_index": 1, "input": True},
        ]

        assert expected_input_rows == actual_input_rows
        assert expected_rows == actual_rows

    async def test_call_aggregate_metrics(self, tmp_path: Path) -> None:
        """Test _call_aggregate_metrics with a mocked server client."""

        agg = AggregateMetrics(
            agent_metrics={"mean/reward": 0.5},
            key_metrics={"mean/reward": 0.5},
            group_level_metrics=[{"mean/reward": 1.0}, {"mean/reward": 0.0}],
        )

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
        mock_response.status = 200

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=mock_response)

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self):
                return mock_server_client

        helper = MockHelper()

        rows = [
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 1},
        ]
        results = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {"usage": {"tokens": 10}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {"usage": {"tokens": 12}}},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {"usage": {"tokens": 8}}},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {"usage": {"tokens": 15}}},
        ]

        output_fpath = tmp_path / "output.jsonl"
        metrics_fpath = await helper._call_aggregate_metrics(results, rows, output_fpath)

        # Verify file was written
        assert metrics_fpath is not None
        assert metrics_fpath.exists()
        written = json.loads(metrics_fpath.read_text())
        assert len(written) == 1
        assert written[0][AGENT_REF_KEY_NAME] == {"name": "my_agent"}
        assert written[0]["agent_metrics"]["mean/reward"] == 0.5
        assert written[0]["key_metrics"]["mean/reward"] == 0.5
        assert len(written[0]["group_level_metrics"]) == 2

        # Verify server_client.post was called with stripped data (usage preserved)
        call_kwargs = mock_server_client.post.call_args
        sent_request = call_kwargs.kwargs["json"]
        sent_data = (
            sent_request.verify_responses
            if isinstance(sent_request, AggregateMetricsRequest)
            else sent_request["verify_responses"]
        )
        for item in sent_data:
            assert "responses_create_params" not in item
            assert "usage" in item["response"]

    async def test_call_aggregate_metrics_multiple_agents(self, tmp_path: Path) -> None:
        """Test _call_aggregate_metrics with multiple agents runs concurrently via as_completed."""

        agg_a = AggregateMetrics(
            agent_metrics={"mean/reward": 1.0},
            key_metrics={"mean/reward": 1.0},
            group_level_metrics=[{"mean/reward": 1.0}],
        )
        agg_b = AggregateMetrics(
            agent_metrics={"mean/reward": 0.0},
            key_metrics={"mean/reward": 0.0},
            group_level_metrics=[{"mean/reward": 0.0}],
        )

        # Return different responses per agent based on server_name
        async def mock_post(server_name, **kwargs):
            agg = agg_a if server_name == "agent_a" else agg_b
            resp = AsyncMock()
            resp.raise_for_status = MagicMock()
            resp.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
            resp.status = 200
            return resp

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(side_effect=mock_post)

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self):
                return mock_server_client

        helper = MockHelper()

        rows = [
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1},
        ]
        results = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {"usage": {"tokens": 10}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 1.0, "response": {"usage": {"tokens": 12}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0, "response": {"usage": {"tokens": 8}}},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0, "response": {"usage": {"tokens": 15}}},
        ]

        output_fpath = tmp_path / "output.jsonl"
        metrics_fpath = await helper._call_aggregate_metrics(results, rows, output_fpath)

        written = json.loads(metrics_fpath.read_text())
        assert len(written) == 2

        # Both agents should be present (order may vary due to as_completed)
        agent_names = {entry[AGENT_REF_KEY_NAME]["name"] for entry in written}
        assert agent_names == {"agent_a", "agent_b"}

        for entry in written:
            if entry[AGENT_REF_KEY_NAME]["name"] == "agent_a":
                assert entry["agent_metrics"]["mean/reward"] == 1.0
            else:
                assert entry["agent_metrics"]["mean/reward"] == 0.0

        # Verify both agents were called
        assert mock_server_client.post.call_count == 2

    async def test_call_aggregate_metrics_empty(self, tmp_path: Path) -> None:
        """_call_aggregate_metrics returns None for empty results."""
        helper = RolloutCollectionHelper()
        output_fpath = tmp_path / "output.jsonl"
        result = await helper._call_aggregate_metrics([], [], output_fpath)
        assert result is None

    async def test_call_aggregate_metrics_skips_rows_without_agent(self, tmp_path: Path) -> None:
        """Rows missing agent_ref are silently skipped during aggregation."""
        agg = AggregateMetrics(
            agent_metrics={"mean/reward": 1.0},
            key_metrics={"mean/reward": 1.0},
            group_level_metrics=[{"mean/reward": 1.0}],
        )

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
        mock_response.status = 200

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=mock_response)

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self):
                return mock_server_client

        helper = MockHelper()

        rows = [
            {},
            {AGENT_REF_KEY_NAME: {"name": "my_agent"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
        ]
        results = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0, "response": {"usage": {"tokens": 5}}},
        ]

        output_fpath = tmp_path / "output.jsonl"
        metrics_fpath = await helper._call_aggregate_metrics(results, rows, output_fpath)

        assert metrics_fpath is not None
        written = json.loads(metrics_fpath.read_text())
        assert len(written) == 1
        assert written[0][AGENT_REF_KEY_NAME] == {"name": "my_agent"}


class TestRolloutCollectionConfigValidation:
    def test_config_requires_input_source(self) -> None:
        """Neither input_jsonl_fpath nor input_gym_url raises ValueError."""
        with pytest.raises(ValueError, match="Either input_jsonl_fpath or input_gym_url must be provided"):
            RolloutCollectionConfig(output_jsonl_fpath="out.jsonl")

    def test_config_rejects_both_input_sources(self) -> None:
        """Both input_jsonl_fpath and input_gym_url raises ValueError."""
        with pytest.raises(ValueError, match="Cannot provide both"):
            RolloutCollectionConfig(
                input_jsonl_fpath="input.jsonl",
                input_gym_url="http://localhost:3000",
                output_jsonl_fpath="out.jsonl",
            )

    def test_config_gym_task_id_requires_gym_url(self) -> None:
        """input_gym_task_id without input_gym_url raises ValueError."""
        with pytest.raises(ValueError, match="input_gym_task_id requires input_gym_url"):
            RolloutCollectionConfig(
                input_jsonl_fpath="input.jsonl",
                input_gym_task_id=["TASK-001"],
                output_jsonl_fpath="out.jsonl",
            )


class TestFetchGymTasks:
    def _mock_urlopen(self, payload: dict):
        """Return a context-manager mock for urllib.request.urlopen."""
        response = MagicMock()
        response.read.return_value = orjson.dumps(payload)
        response.__enter__ = lambda s: s
        response.__exit__ = MagicMock(return_value=False)
        return response

    def test_happy_path_with_task_statement_and_viewport_list(self) -> None:
        """Tasks with task_statement and list viewport are parsed correctly."""
        payload = {
            "verifiers": {
                "TASK-001": {
                    "task_statement": "Click the button",
                    "start_url": "http://app.test/page1",
                    "viewport_size": [1920, 1080],
                }
            }
        }
        with patch("nemo_gym.rollout_collection.urllib.request.urlopen", return_value=self._mock_urlopen(payload)):
            rows = _fetch_gym_tasks("http://gym.test")

        assert len(rows) == 1
        row = orjson.loads(rows[0])
        assert row["responses_create_params"]["input"][0]["content"] == "Click the button"
        assert row["verifier_metadata"]["start_url"] == "http://app.test/page1"
        assert row["verifier_metadata"]["viewport"] == {"width": 1920, "height": 1080}

    def test_prompt_fallback_when_no_task_statement(self) -> None:
        """Falls back to 'prompt' field when 'task_statement' is absent."""
        payload = {
            "verifiers": {
                "TASK-002": {
                    "prompt": "Fill the form",
                    "start_url": "http://app.test/form",
                }
            }
        }
        with patch("nemo_gym.rollout_collection.urllib.request.urlopen", return_value=self._mock_urlopen(payload)):
            rows = _fetch_gym_tasks("http://gym.test/")

        row = orjson.loads(rows[0])
        assert row["responses_create_params"]["input"][0]["content"] == "Fill the form"

    def test_viewport_dict_passthrough(self) -> None:
        """Viewport already in dict format passes through unchanged."""
        payload = {
            "verifiers": {
                "TASK-003": {
                    "task_statement": "test",
                    "viewport_size": {"width": 800, "height": 600},
                }
            }
        }
        with patch("nemo_gym.rollout_collection.urllib.request.urlopen", return_value=self._mock_urlopen(payload)):
            rows = _fetch_gym_tasks("http://gym.test")

        row = orjson.loads(rows[0])
        assert row["verifier_metadata"]["viewport"] == {"width": 800, "height": 600}

    def test_viewport_default_when_missing(self) -> None:
        """Default 1280x720 viewport when not specified."""
        payload = {"verifiers": {"TASK-004": {"task_statement": "test"}}}
        with patch("nemo_gym.rollout_collection.urllib.request.urlopen", return_value=self._mock_urlopen(payload)):
            rows = _fetch_gym_tasks("http://gym.test")

        row = orjson.loads(rows[0])
        assert row["verifier_metadata"]["viewport"] == {"width": 1280, "height": 720}

    def test_non_dict_details_uses_gym_url_fallback(self) -> None:
        """Non-dict verifier details falls back to gym_url for start_url and defaults for viewport."""
        payload = {"verifiers": {"TASK-005": "just a string"}}
        with patch("nemo_gym.rollout_collection.urllib.request.urlopen", return_value=self._mock_urlopen(payload)):
            rows = _fetch_gym_tasks("http://gym.test")

        row = orjson.loads(rows[0])
        assert row["responses_create_params"]["input"][0]["content"] == ""
        assert row["verifier_metadata"]["start_url"] == "http://gym.test"
        assert row["verifier_metadata"]["viewport"] == {"width": 1280, "height": 720}

    def test_start_url_defaults_to_gym_url(self) -> None:
        """When start_url is absent from details, gym_url is used."""
        payload = {"verifiers": {"TASK-006": {"task_statement": "test"}}}
        with patch("nemo_gym.rollout_collection.urllib.request.urlopen", return_value=self._mock_urlopen(payload)):
            rows = _fetch_gym_tasks("http://gym.test")

        row = orjson.loads(rows[0])
        assert row["verifier_metadata"]["start_url"] == "http://gym.test"

    def test_filter_by_task_ids(self) -> None:
        """Only requested task IDs are returned."""
        payload = {
            "verifiers": {
                "A": {"task_statement": "a"},
                "B": {"task_statement": "b"},
                "C": {"task_statement": "c"},
            }
        }
        with patch("nemo_gym.rollout_collection.urllib.request.urlopen", return_value=self._mock_urlopen(payload)):
            rows = _fetch_gym_tasks("http://gym.test", task_ids=["A", "C"])

        assert len(rows) == 2
        task_ids = [orjson.loads(r)["verifier_metadata"]["task_id"] for r in rows]
        assert task_ids == ["A", "C"]

    def test_missing_task_ids_raises(self) -> None:
        """Requesting non-existent task IDs raises ValueError."""
        payload = {"verifiers": {"A": {"task_statement": "a"}}}
        with patch("nemo_gym.rollout_collection.urllib.request.urlopen", return_value=self._mock_urlopen(payload)):
            with pytest.raises(ValueError, match="Task ID.*not found"):
                _fetch_gym_tasks("http://gym.test", task_ids=["MISSING"])

    def test_empty_verifiers_raises(self) -> None:
        """Empty verifiers dict raises ValueError."""
        payload = {"verifiers": {}}
        with patch("nemo_gym.rollout_collection.urllib.request.urlopen", return_value=self._mock_urlopen(payload)):
            with pytest.raises(ValueError, match="No verifiers found"):
                _fetch_gym_tasks("http://gym.test")

    def test_http_error_raises(self) -> None:
        """HTTPError is wrapped in a ValueError."""
        with patch(
            "nemo_gym.rollout_collection.urllib.request.urlopen",
            side_effect=urllib.error.HTTPError("http://gym.test", 500, "Internal Error", {}, BytesIO(b"")),
        ):
            with pytest.raises(ValueError, match="HTTP 500"):
                _fetch_gym_tasks("http://gym.test")

    def test_url_error_raises(self) -> None:
        """URLError (connection failure) is wrapped in a ValueError."""
        with patch(
            "nemo_gym.rollout_collection.urllib.request.urlopen",
            side_effect=urllib.error.URLError("Connection refused"),
        ):
            with pytest.raises(ValueError, match="Could not connect"):
                _fetch_gym_tasks("http://gym.test")


class TestPreprocessRowsGymUrl:
    def test_preprocess_rows_from_gym_url(self) -> None:
        """input_gym_url branch calls _fetch_gym_tasks and processes rows."""
        fake_row = orjson.dumps(
            {
                "responses_create_params": {"input": [{"role": "user", "content": "Do task"}]},
                "verifier_metadata": {"task_id": "T1", "gym_url": "http://gym.test"},
            }
        ).decode()

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_gym_url="http://gym.test",
            output_jsonl_fpath="out.jsonl",
            num_repeats=1,
        )

        with patch("nemo_gym.rollout_collection._fetch_gym_tasks", return_value=[fake_row]):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        assert len(rows) == 1
        assert rows[0]["responses_create_params"]["input"][0]["content"] == "Do task"
        assert rows[0]["agent_ref"] == {"name": "my_agent"}
        assert rows[0][TASK_INDEX_KEY_NAME] == 0
        assert rows[0][ROLLOUT_INDEX_KEY_NAME] == 0

    def test_preprocess_rows_missing_agent_ref_raises(self, tmp_path: Path) -> None:
        """Rows without agent_ref and no agent_name config raises ValueError."""
        fpath = tmp_path / "input.jsonl"
        fpath.write_text(json.dumps({"responses_create_params": {"input": []}}) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats=1,
        )

        with pytest.raises(ValueError, match="No agent specified"):
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)


class TestRunFromConfigEdgeCases:
    def _make_helper(self):
        class StubHelper(RolloutCollectionHelper):
            def run_examples(self, examples, *args, **kwargs):
                futures = []
                for example in examples:
                    future = Future()
                    future.set_result((example, {"response": {"usage": {"tokens": 1}}}))
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                return None

        return StubHelper()

    async def test_resume_from_cache_skips_when_output_missing(self, tmp_path: Path) -> None:
        """resume_from_cache=True with missing output file falls through to normal path."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=True,
            num_repeats=1,
        )

        results = await self._make_helper().run_from_config(config)
        assert len(results) == 1

    async def test_resume_from_cache_skips_when_materialized_missing(self, tmp_path: Path) -> None:
        """resume_from_cache=True with missing materialized file falls through to normal path."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"
        output_jsonl_fpath.write_text("")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            resume_from_cache=True,
            num_repeats=1,
        )

        results = await self._make_helper().run_from_config(config)
        assert len(results) == 1

    async def test_run_from_config_with_num_samples_in_parallel(self, tmp_path: Path) -> None:
        """num_samples_in_parallel creates a Semaphore for concurrency control."""
        input_jsonl_fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "agent"}, "x": i})
            for i in range(3)
        ]
        input_jsonl_fpath.write_text("\n".join(samples) + "\n")
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            num_repeats=1,
            num_samples_in_parallel=2,
        )

        results = await self._make_helper().run_from_config(config)
        assert len(results) == 3

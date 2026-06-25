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
from collections import Counter
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
import yaml

import nemo_gym.rollout_collection
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import (
    _DEFAULT_MAX_ROLLOUT_ATTEMPTS,
    RolloutAggregationConfig,
    RolloutAggregationHelper,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _expand_input_glob,
    _get_max_rollout_attempts,
    _rollout_request_debug_summary,
)


class TestGetMaxRolloutAttempts:
    def test_default_when_unset(self, monkeypatch) -> None:
        monkeypatch.delenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", raising=False)
        assert _get_max_rollout_attempts() == _DEFAULT_MAX_ROLLOUT_ATTEMPTS

    def test_default_when_empty(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "")
        assert _get_max_rollout_attempts() == _DEFAULT_MAX_ROLLOUT_ATTEMPTS

    def test_valid_value(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "5")
        assert _get_max_rollout_attempts() == 5

    def test_non_integer_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "not-an-int")
        assert _get_max_rollout_attempts() == _DEFAULT_MAX_ROLLOUT_ATTEMPTS

    def test_non_positive_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("NEMO_GYM_MAX_ROLLOUT_ATTEMPTS", "0")
        assert _get_max_rollout_attempts() == _DEFAULT_MAX_ROLLOUT_ATTEMPTS


class TestRolloutCollection:
    def test_rollout_request_debug_summary_compact(self) -> None:
        row = {
            AGENT_REF_KEY_NAME: {"name": "my_agent"},
            TASK_INDEX_KEY_NAME: 12,
            ROLLOUT_INDEX_KEY_NAME: 3,
            "env_specific_metadata": "do not include",
            "responses_create_params": {"input": "large prompt", "tools": ["large schema"]},
        }

        assert _rollout_request_debug_summary(row) == {
            "agent_name": "my_agent",
            TASK_INDEX_KEY_NAME: 12,
            ROLLOUT_INDEX_KEY_NAME: 3,
        }

    @pytest.mark.parametrize("request_debug_enabled", [True, False])
    async def test_run_examples_logs_failed_run_when_request_debug_enabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        request_debug_enabled: bool,
    ) -> None:
        row = {
            AGENT_REF_KEY_NAME: {"name": "my_agent"},
            TASK_INDEX_KEY_NAME: 7,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "env_specific_metadata": "do not log this either",
            "responses_create_params": {"input": "do not log this"},
        }
        response = MagicMock()
        response.status = 500

        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=response)

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                return mock_server_client

        async def fail_raise_for_status(_response):
            raise RuntimeError("boom")

        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", fail_raise_for_status)
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "is_global_aiohttp_client_request_debug_enabled",
            lambda: request_debug_enabled,
        )

        with pytest.raises(RuntimeError, match="boom"):
            await next(MockHelper().run_examples([row]))

        captured = capsys.readouterr()
        if request_debug_enabled:
            assert "[rollout_collection] /run failed status=500" in captured.out
            assert '"_ng_task_index": 7' in captured.out
            assert '"_ng_rollout_index": 0' in captured.out
            assert '"agent_name": "my_agent"' in captured.out
            assert "env_specific_metadata" not in captured.out
            assert "do not log this either" not in captured.out
            assert "responses_create_params" not in captured.out
            assert "do not log this" not in captured.out
        else:
            assert "[rollout_collection] /run failed" not in captured.out

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
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 0,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 1}'},
                    "temperature": 0.1,
                },
                "x": 0,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 1,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 1}'},
                    "temperature": 0.1,
                },
                "x": 1,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 0,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 0}'},
                    "temperature": 0.1,
                },
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
            {
                "_ng_task_index": 2,
                "_ng_rollout_index": 1,
                "responses_create_params": {
                    "input": [],
                    "metadata": {"extra_body": '{"seed": 1}'},
                    "temperature": 0.1,
                },
                "x": 2,
                "agent_ref": {"name": "my_agent"},
            },
        ]

    def test_preprocess_rows_num_repeats_add_seed_passes_pydantic_validation(self, tmp_path: Path) -> None:
        """Rows emitted with num_repeats_add_seed=True must round-trip through the strict
        NeMoGymResponseCreateParamsNonStreaming schema (extra='forbid'). Seed is passed via
        metadata.extra_body so it doesn't violate the OpenAI Responses schema."""
        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(2)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats=3,
            num_repeats_add_seed=True,
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        assert len(rows) == 6
        seeds_seen = []
        for row in rows:
            rcp = row["responses_create_params"]
            # seed lives in metadata.extra_body, not at the top level
            assert "seed" not in rcp
            extra_body = json.loads(rcp["metadata"]["extra_body"])
            seeds_seen.append(extra_body["seed"])
            # Must still pass the strict schema validation
            NeMoGymResponseCreateParamsNonStreaming.model_validate(rcp)
        # Seeds should track rollout index within each task (0, 1, 2 per task).
        assert seeds_seen == [0, 1, 2, 0, 1, 2]

    def test_preprocess_rows_num_repeats_dict_form(self, tmp_path: Path) -> None:
        """Dict-form num_repeats applies the per-agent value to each row."""
        fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}, "x": 0}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "beta"}, "x": 1}),
        ]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats={"alpha": 2, "beta": 4},
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        per_agent_counts = Counter(row[AGENT_REF_KEY_NAME]["name"] for row in rows)
        assert per_agent_counts == Counter({"alpha": 2, "beta": 4})
        assert [r[ROLLOUT_INDEX_KEY_NAME] for r in rows if r[AGENT_REF_KEY_NAME]["name"] == "alpha"] == [0, 1]
        assert [r[ROLLOUT_INDEX_KEY_NAME] for r in rows if r[AGENT_REF_KEY_NAME]["name"] == "beta"] == [0, 1, 2, 3]

    def test_preprocess_rows_num_repeats_dict_with_default(self, tmp_path: Path) -> None:
        """`_default` key acts as the fallback for agents not explicitly listed."""
        fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}, "x": 0}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "beta"}, "x": 1}),
        ]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats={"alpha": 3, "_default": 1},
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        per_agent_counts = Counter(row[AGENT_REF_KEY_NAME]["name"] for row in rows)
        assert per_agent_counts == Counter({"alpha": 3, "beta": 1})

    def test_preprocess_rows_num_repeats_dict_raises_on_missing_agent_no_default(self, tmp_path: Path) -> None:
        """Dict form without `_default` raises if a row's agent is unlisted, and reports ALL
        missing agents in one error so the user can fix them in one pass."""
        fpath = tmp_path / "input.jsonl"
        samples = [
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}, "x": 0}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "beta"}, "x": 1}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "gamma"}, "x": 2}),
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "beta"}, "x": 3}),
        ]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats={"alpha": 2},
        )

        with pytest.raises(ValueError) as exc_info:
            RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        msg = str(exc_info.value)
        # All missing agents reported in one shot, deduped:
        assert "'beta'" in msg
        assert "'gamma'" in msg

    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_preprocess_rows_num_repeats_rejects_zero_or_negative(self, tmp_path: Path, bad_value: int) -> None:
        # int form
        with pytest.raises(ValueError, match="num_repeats"):
            RolloutCollectionConfig(
                agent_name="my_agent",
                input_jsonl_fpath=str(tmp_path / "in.jsonl"),
                output_jsonl_fpath=str(tmp_path / "out.jsonl"),
                num_repeats=bad_value,
            )
        # dict form
        with pytest.raises(ValueError, match="num_repeats dict"):
            RolloutCollectionConfig(
                agent_name="my_agent",
                input_jsonl_fpath=str(tmp_path / "in.jsonl"),
                output_jsonl_fpath=str(tmp_path / "out.jsonl"),
                num_repeats={"alpha": bad_value},
            )

    def test_preprocess_rows_num_repeats_dict_unknown_agent_warns(self, tmp_path: Path) -> None:
        """An agent listed in the dict that never appears in input rows warns (likely typo)."""
        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "alpha"}, "x": 0})]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats={"alpha": 2, "alpah_typo": 3},
        )

        with pytest.warns(UserWarning, match="alpah_typo"):
            rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert len(rows) == 2

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


class TestExpandInputGlob:
    """`_expand_input_glob` accepts a single glob, a comma-separated list of globs, or a mix.

    Mirrors the multi-pattern conventions used elsewhere in NeMo Skills
    (e.g. comma-separated `config_paths` on `ns nemo_gym_rollouts`).
    """

    def test_single_path(self, tmp_path: Path) -> None:
        a = tmp_path / "a.jsonl"
        a.write_text("{}\n")
        assert _expand_input_glob(str(a)) == [str(a)]

    def test_single_glob(self, tmp_path: Path) -> None:
        for i in range(3):
            (tmp_path / f"rollouts-chunk{i}.jsonl").write_text("{}\n")
        result = _expand_input_glob(str(tmp_path / "rollouts-chunk*.jsonl"))
        assert result == sorted(str(tmp_path / f"rollouts-chunk{i}.jsonl") for i in range(3))

    def test_comma_separated_paths(self, tmp_path: Path) -> None:
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text("{}\n")
        b.write_text("{}\n")
        result = _expand_input_glob(f"{a},{b}")
        assert set(result) == {str(a), str(b)}

    def test_comma_separated_globs(self, tmp_path: Path) -> None:
        for sub in ("run1", "run2"):
            (tmp_path / sub).mkdir()
            (tmp_path / sub / "rollouts.jsonl").write_text("{}\n")
            (tmp_path / sub / "extra.txt").write_text("ignore me")
        result = _expand_input_glob(f"{tmp_path / 'run1' / 'rollouts*.jsonl'},{tmp_path / 'run2' / 'rollouts*.jsonl'}")
        assert set(result) == {
            str(tmp_path / "run1" / "rollouts.jsonl"),
            str(tmp_path / "run2" / "rollouts.jsonl"),
        }

    def test_whitespace_around_commas_is_stripped(self, tmp_path: Path) -> None:
        a = tmp_path / "a.jsonl"
        b = tmp_path / "b.jsonl"
        a.write_text("{}\n")
        b.write_text("{}\n")
        result = _expand_input_glob(f"  {a}  ,  {b}  ")
        assert set(result) == {str(a), str(b)}

    def test_overlapping_patterns_dedup(self, tmp_path: Path) -> None:
        """A file matched by two patterns appears once in the output."""
        a = tmp_path / "a.jsonl"
        a.write_text("{}\n")
        result = _expand_input_glob(f"{tmp_path / '*.jsonl'},{a}")
        assert result == [str(a)]

    def test_no_matches_returns_empty(self, tmp_path: Path) -> None:
        assert _expand_input_glob(str(tmp_path / "nonexistent-*.jsonl")) == []

    def test_empty_strings_in_csv_are_dropped(self, tmp_path: Path) -> None:
        """Trailing/leading commas don't produce an empty-pattern glob that matches everything."""
        a = tmp_path / "a.jsonl"
        a.write_text("{}\n")
        result = _expand_input_glob(f",{a},,")
        assert result == [str(a)]


class TestDisableAggregationAndCallerTaskIndex:
    """Branches added for sharded rollouts: `disable_aggregation` flag and
    caller-provided `_ng_task_index`. Both must be backward-compatible with
    the existing default-on aggregation + auto-numbering behaviour.
    """

    async def test_run_from_config_disable_aggregation_skips_call(self, tmp_path: Path) -> None:
        """When disable_aggregation=True, _call_aggregate_metrics MUST NOT run.

        Shows up in chunked-rollouts flows where the aggregation pass is deferred
        to a single ng_aggregate_rollouts run over the union of shards.
        """
        input_jsonl_fpath = tmp_path / "input.jsonl"
        input_jsonl_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}, "x": 0}) + "\n"
        )
        output_jsonl_fpath = tmp_path / "output.jsonl"

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_jsonl_fpath),
            output_jsonl_fpath=str(output_jsonl_fpath),
            disable_aggregation=True,
            num_repeats=1,
        )

        class Helper(RolloutCollectionHelper):
            def run_examples(self, examples, *args, **kwargs):
                futures = []
                for ex in examples:
                    fut = Future()
                    fut.set_result((ex, {"response": {"usage": {}}}))
                    futures.append(fut)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                raise AssertionError("aggregator must not run when disable_aggregation=True")

        await Helper().run_from_config(config)

        # Rollouts file written (proves the rollout phase ran); aggregator file absent.
        assert output_jsonl_fpath.exists()
        assert not (tmp_path / "output_aggregate_metrics.json").exists()

    def test_preprocess_honors_caller_task_index(self, tmp_path: Path) -> None:
        """A row arriving with `_ng_task_index` pre-set is used verbatim — the
        original `row_to_task_idx` auto-numbering is bypassed. This is the seam
        an upstream slicer relies on to keep task identifiers globally-stable
        across shards.
        """
        fpath = tmp_path / "input.jsonl"
        rows = [
            # Same prompt twice with *different* caller-stamped indices — must
            # NOT be collapsed to one task by the row_str dedup path.
            {"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}, TASK_INDEX_KEY_NAME: 42},
            {"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}, TASK_INDEX_KEY_NAME: 99},
            # And a third row with no caller index — auto-numbering still applies.
            {"responses_create_params": {"input": []}, "agent_ref": {"name": "a"}, "diff": "row"},
        ]
        fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

        config = RolloutCollectionConfig(
            agent_name="a",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats=1,
        )

        result = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        indices = [r[TASK_INDEX_KEY_NAME] for r in result]

        # Caller-provided indices preserved; the no-index row gets an auto-generated
        # one starting at 0 (the row_to_task_idx counter is independent of caller stamps).
        assert indices[:2] == [42, 99]
        assert indices[2] == 0  # auto-assigned; not 100 or 43


class TestRolloutAggregationHelper:
    """End-to-end shape of `ng_aggregate_rollouts`: glob → load → sort → aggregate."""

    async def test_run_from_config_full_path(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Two shards. Records have globally-stamped task indices (out of order)
        # — the helper should sort by (task_index, rollout_index) before calling
        # _call_aggregate_metrics so downstream groupby is deterministic.
        shard0 = tmp_path / "rollouts-chunk0.jsonl"
        shard1 = tmp_path / "rollouts-chunk1.jsonl"
        records_shard0 = [
            {
                AGENT_REF_KEY_NAME: {"name": "a"},
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "response": {"usage": {"x": 2}},
                "reward": 1.0,
            },
            {
                AGENT_REF_KEY_NAME: {"name": "a"},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "response": {"usage": {"x": 1}},
                "reward": 0.0,
            },
        ]
        records_shard1 = [
            {
                AGENT_REF_KEY_NAME: {"name": "a"},
                TASK_INDEX_KEY_NAME: 2,
                ROLLOUT_INDEX_KEY_NAME: 0,
                "response": {"usage": {"x": 3}},
                "reward": 1.0,
            },
        ]
        shard0.write_text("\n".join(json.dumps(r) for r in records_shard0) + "\n")
        shard1.write_text("\n".join(json.dumps(r) for r in records_shard1) + "\n")

        output_fpath = tmp_path / "rollouts.jsonl"

        captured: dict[str, list] = {}

        async def fake_call(self, results, rows, output_fpath):
            captured["results"] = results
            captured["rows"] = rows
            captured["output_fpath"] = output_fpath
            # Touch a sentinel file so the helper's return value is meaningful.
            metrics_fpath = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
            metrics_fpath.write_text("[]")
            return metrics_fpath

        monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", fake_call)

        cfg = RolloutAggregationConfig(
            input_glob=f"{shard0},{shard1}",
            output_jsonl_fpath=str(output_fpath),
            merge_shards=True,
        )
        metrics_fpath = await RolloutAggregationHelper().run_from_config(cfg)

        # 3 records total, sorted by (task_index, rollout_index): tasks 0, 1, 2.
        assert [r[TASK_INDEX_KEY_NAME] for r in captured["results"]] == [0, 1, 2]
        # rows passed twice == results (helper uses results both ways since each
        # row already carries AGENT_REF_KEY_NAME).
        assert captured["rows"] is captured["results"]
        # Merged shard concatenation honoured (merge_shards=True).
        assert output_fpath.exists()
        assert sum(1 for _ in output_fpath.open()) == 3
        # Metrics file path returned and points next to the merged JSONL.
        assert metrics_fpath == tmp_path / "rollouts_aggregate_metrics.json"
        assert metrics_fpath.exists()

    async def test_run_from_config_no_matches_raises(self, tmp_path: Path) -> None:
        cfg = RolloutAggregationConfig(
            input_glob=str(tmp_path / "nothing-matches-*.jsonl"),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        )
        with pytest.raises(FileNotFoundError, match="No shards matched"):
            await RolloutAggregationHelper().run_from_config(cfg)

    async def test_run_from_config_merge_shards_false_skips_concat(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        shard = tmp_path / "shard.jsonl"
        record = {
            AGENT_REF_KEY_NAME: {"name": "a"},
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "response": {"usage": {}},
            "reward": 0.5,
        }
        shard.write_text(json.dumps(record) + "\n")
        output_fpath = tmp_path / "rollouts.jsonl"

        async def _noop(self, results, rows, output_fpath):
            m = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
            m.write_text("[]")
            return m

        monkeypatch.setattr(RolloutCollectionHelper, "_call_aggregate_metrics", _noop)
        cfg = RolloutAggregationConfig(
            input_glob=str(shard),
            output_jsonl_fpath=str(output_fpath),
            merge_shards=False,
        )
        await RolloutAggregationHelper().run_from_config(cfg)

        # merge_shards=False ⇒ no concatenated rollouts file is written, even
        # though output_jsonl_fpath is used to derive the metrics path.
        assert not output_fpath.exists()
        assert (tmp_path / "rollouts_aggregate_metrics.json").exists()

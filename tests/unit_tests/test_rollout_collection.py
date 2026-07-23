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
import asyncio
import json
from asyncio import Future
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from urllib.parse import urlparse

import orjson
import pytest
import yaml
from aiohttp import ClientConnectorError, ClientPayloadError, ServerDisconnectedError
from pydantic import ValidationError

import nemo_gym.rollout_collection
from nemo_gym.base_resources_server import AggregateMetrics, AggregateMetricsRequest
from nemo_gym.config_types import ConfigError, ConfigPathNotFoundError
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME, row_agent_key
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.reward_profile import compute_aggregate_metrics
from nemo_gym.rollout_collection import (
    _DEFAULT_MAX_ROLLOUT_ATTEMPTS,
    EXTERNAL_AGENT_FAILURE_CLASS,
    NG_FAILURE_CLASS_KEY,
    RolloutAggregationConfig,
    RolloutAggregationHelper,
    RolloutCollectionConfig,
    RolloutCollectionHelper,
    _agent_metric_label,
    _expand_input_glob,
    _get_max_rollout_attempts,
    _normalize_agent_url,
    _post_external_agent_run,
    _post_external_aggregate_metrics,
    _rollout_request_debug_summary,
    loads_jsonl_line,
)


class TestLoadsJsonlLine:
    def test_parses_valid_line(self) -> None:
        assert loads_jsonl_line('{"a": 1}', "f.jsonl", 1) == {"a": 1}

    def test_malformed_line_raises_config_error_with_location(self) -> None:
        with pytest.raises(ConfigError, match=r"Malformed JSON in 'f.jsonl' at line 3"):
            loads_jsonl_line("{not json", "f.jsonl", 3)


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

    def test_preprocess_rows_missing_input_raises_config_error(self, tmp_path: Path) -> None:
        """A non-existent input file fails with a clean ConfigPathNotFoundError, not a raw FileNotFoundError."""
        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(tmp_path / "does_not_exist.jsonl"),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        )

        with pytest.raises(ConfigPathNotFoundError, match="does_not_exist.jsonl.*--input"):
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

    def test_preprocess_rows_stamps_skills_ref(self, tmp_path: Path) -> None:
        """skills.path is a run-level knob: each row is stamped with skills_ref (path + hash +
        metadata) without the source dataset carrying any skills field."""
        skills_dir = tmp_path / "variant_a"
        skill = skills_dir / "cot_enhanced"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: cot_enhanced\ndescription: Think step by step.\n---\n# Body\n")

        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(2)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            skills={"path": str(skills_dir)},
        )

        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)

        assert len(rows) == 2
        for row in rows:
            skills_ref = row["skills_ref"]
            assert skills_ref["path"] == str(skills_dir)
            assert len(skills_ref["hash"]) == 12
            assert [s["name"] for s in skills_ref["skills"]] == ["cot_enhanced"]
            assert skills_ref["skills"][0]["description"] == "Think step by step."

    def test_preprocess_rows_no_skills_leaves_rows_clean(self, tmp_path: Path) -> None:
        fpath = tmp_path / "input.jsonl"
        fpath.write_text(json.dumps({"responses_create_params": {"input": []}}) + "\n")
        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
        )
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        assert "skills_ref" not in rows[0]

    def test_skills_ref_survives_resume_from_cache(self, tmp_path: Path) -> None:
        """skills_ref is stamped once at preprocess, persisted to materialized inputs, and
        re-read onto already-done rows on resume -- even after the source skill dir is gone.
        Identity is byte-for-byte from the materialized cache, not recomputed at resume."""
        import shutil

        skills_dir = tmp_path / "variant_a"
        skill = skills_dir / "cot_enhanced"
        skill.mkdir(parents=True)
        (skill / "SKILL.md").write_text("---\nname: cot_enhanced\ndescription: Think step by step.\n---\n# Body\n")

        fpath = tmp_path / "input.jsonl"
        samples = [json.dumps({"responses_create_params": {"input": []}, "x": i}) for i in range(2)]
        fpath.write_text("\n".join(samples) + "\n")

        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            skills={"path": str(skills_dir)},
            resume_from_cache=True,
        )

        # Preprocess stamps skills_ref, then we persist exactly what a prior run would have written.
        rows = RolloutCollectionHelper._preprocess_rows_from_config(None, config)
        stamped_skills_ref = rows[0]["skills_ref"]
        config.materialized_jsonl_fpath.write_bytes(b"\n".join(orjson.dumps(r) for r in rows) + b"\n")

        # Only the first task's rollout is "done" in the main output jsonl.
        done = {k: rows[0][k] for k in (TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME)} | {"reward": 1.0}
        Path(config.output_jsonl_fpath).write_bytes(orjson.dumps(done) + b"\n")

        # The source skill dir disappears before resume (e.g. an optimizer overwrote /tmp).
        shutil.rmtree(skills_dir)

        input_rows, resumed_rows, _results, _result_strs = RolloutCollectionHelper()._load_from_cache(config)

        # The already-done row carries the original skills_ref read back from the cache.
        assert resumed_rows[0]["skills_ref"] == stamped_skills_ref
        # And the still-to-run rows do too, so the second pass stamps results identically.
        assert all(r["skills_ref"] == stamped_skills_ref for r in input_rows)

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

    def test_num_repeats_null_coerces_to_one(self, tmp_path: Path) -> None:
        # `--num-repeats null` (None) restores the pre-#1356 default of 1.
        config = RolloutCollectionConfig(
            agent_name="my_agent",
            input_jsonl_fpath=str(tmp_path / "in.jsonl"),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            num_repeats=None,
        )
        assert config.num_repeats == 1

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


class TestAgentUrlConfig:
    _REQUIRED = {"input_jsonl_fpath": "in.jsonl", "output_jsonl_fpath": "out.jsonl"}

    def test_agent_url_and_agent_name_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            RolloutCollectionConfig(agent_name="my_agent", agent_url="http://localhost:9000", **self._REQUIRED)

    def test_agent_url_normalized(self) -> None:
        config = RolloutCollectionConfig(agent_url="http://localhost:9000/", **self._REQUIRED)
        assert config.agent_url == "http://localhost:9000"

    @pytest.mark.parametrize("bad_url", ["ftp://localhost:9000", "localhost:9000", "http://"])
    def test_agent_url_invalid_rejected(self, bad_url: str) -> None:
        with pytest.raises(ValidationError, match="absolute http:// or https:// URL"):
            RolloutCollectionConfig(agent_url=bad_url, **self._REQUIRED)

    def test_agent_run_timeout_must_be_positive(self) -> None:
        with pytest.raises(ValidationError, match="agent_run_timeout_secs must be > 0"):
            RolloutCollectionConfig(agent_url="http://localhost:9000", agent_run_timeout_secs=0, **self._REQUIRED)

    def test_neither_agent_name_nor_url_is_valid(self) -> None:
        # Rows may carry their own agent_ref; the config-level default is optional.
        config = RolloutCollectionConfig(**self._REQUIRED)
        assert config.agent_name is None and config.agent_url is None


class TestAgentUrlPreprocess:
    def _write_input(self, tmp_path: Path, rows: list[dict]) -> str:
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(input_fpath)

    def _config(self, tmp_path: Path, rows: list[dict], **kwargs) -> RolloutCollectionConfig:
        return RolloutCollectionConfig(
            input_jsonl_fpath=self._write_input(tmp_path, rows),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            **kwargs,
        )

    def test_agent_url_stamped_on_rows_missing_ref(self, tmp_path: Path) -> None:
        rows = [
            {"responses_create_params": {"input": []}},
            {"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "named_agent"}},
        ]
        config = self._config(tmp_path, rows, agent_url="http://localhost:9000")
        processed = RolloutCollectionHelper()._preprocess_rows_from_config(config)

        assert processed[0][AGENT_REF_KEY_NAME] == {"url": "http://localhost:9000"}
        # Row-level named refs win over the config default (same semantics as agent_name)
        assert processed[1][AGENT_REF_KEY_NAME] == {"name": "named_agent"}

    def test_row_level_url_must_match_config(self, tmp_path: Path) -> None:
        rows = [{"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"url": "http://evil:1"}}]
        config = self._config(tmp_path, rows, agent_url="http://localhost:9000")
        with pytest.raises(ValueError, match="does not match the configured \\+agent_url"):
            RolloutCollectionHelper()._preprocess_rows_from_config(config)

    def test_row_level_url_without_config_rejected(self, tmp_path: Path) -> None:
        rows = [{"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"url": "http://localhost:9000"}}]
        config = self._config(tmp_path, rows)
        with pytest.raises(ValueError, match="does not match the configured \\+agent_url"):
            RolloutCollectionHelper()._preprocess_rows_from_config(config)

    def test_row_level_url_matching_config_normalized(self, tmp_path: Path) -> None:
        rows = [{"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"url": "http://localhost:9000/"}}]
        config = self._config(tmp_path, rows, agent_url="http://localhost:9000")
        processed = RolloutCollectionHelper()._preprocess_rows_from_config(config)
        assert processed[0][AGENT_REF_KEY_NAME] == {"url": "http://localhost:9000"}

    def test_ref_with_both_name_and_url_rejected(self, tmp_path: Path) -> None:
        rows = [
            {
                "responses_create_params": {"input": []},
                AGENT_REF_KEY_NAME: {"name": "x", "url": "http://localhost:9000"},
            }
        ]
        config = self._config(tmp_path, rows, agent_url="http://localhost:9000")
        with pytest.raises(ValueError, match="both 'name' and 'url'"):
            RolloutCollectionHelper()._preprocess_rows_from_config(config)

    def test_missing_agent_error_mentions_agent_url(self, tmp_path: Path) -> None:
        rows = [{"responses_create_params": {"input": []}}]
        config = self._config(tmp_path, rows)
        with pytest.raises(ValueError, match=r"\+agent_url \(external agent endpoint\)"):
            RolloutCollectionHelper()._preprocess_rows_from_config(config)

    def test_agent_url_noop_warning_when_all_rows_have_refs(self, tmp_path: Path) -> None:
        rows = [{"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"name": "named_agent"}}]
        config = self._config(tmp_path, rows, agent_url="http://localhost:9000")
        with pytest.warns(UserWarning, match="nothing will be dispatched to the external agent"):
            RolloutCollectionHelper()._preprocess_rows_from_config(config)

    def test_num_repeats_dict_keyed_by_url(self, tmp_path: Path) -> None:
        rows = [{"responses_create_params": {"input": []}}]
        config = self._config(
            tmp_path, rows, agent_url="http://localhost:9000", num_repeats={"http://localhost:9000": 3}
        )
        processed = RolloutCollectionHelper()._preprocess_rows_from_config(config)
        assert len(processed) == 3


class _FakeAiohttpResponse:
    def __init__(self, status: int, content: bytes, headers: dict | None = None):
        self.status = status
        self._content = content
        self.headers = headers or {}

    @property
    def ok(self) -> bool:
        return self.status < 400

    async def read(self) -> bytes:
        return self._content


def _mock_global_client(monkeypatch: pytest.MonkeyPatch, request_mock: AsyncMock) -> MagicMock:
    client = MagicMock()
    client.request = request_mock
    monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_aiohttp_client", lambda: client)
    monkeypatch.setattr(nemo_gym.rollout_collection, "_EXTERNAL_AGENT_RETRY_SLEEP_SECS", 0)
    return client


_URL_ROW = {
    AGENT_REF_KEY_NAME: {"url": "http://localhost:9000"},
    "responses_create_params": {"input": []},
    # Intentional (issue #1305): the full row — including the answer key in verifier_metadata —
    # is sent to the external agent; this test locks that behavior in
    "verifier_metadata": {"expected_answer": "the-answer-key"},
}


class TestExternalAgentDispatch:
    async def test_success_posts_row_with_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, orjson.dumps({"reward": 1.0, "response": {}})))
        client = _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=123.0)

        assert result == {"reward": 1.0, "response": {}}
        assert client.request.call_count == 1
        args, kwargs = client.request.call_args
        assert args == ("POST", "http://localhost:9000/run")
        assert orjson.loads(kwargs["data"]) == _URL_ROW
        assert kwargs["headers"] == {"Content-Type": "application/json"}
        assert kwargs["timeout"].total == 123.0
        assert kwargs["allow_redirects"] is False

    async def test_connect_error_bounded_retries_then_failure_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(side_effect=ClientConnectorError(MagicMock(), OSError("connection refused")))
        client = _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert client.request.call_count == 3
        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "Is your agent running at this URL?" in result["error"]

    async def test_server_disconnected_retry_then_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(
            side_effect=[
                ServerDisconnectedError(),
                _FakeAiohttpResponse(200, orjson.dumps({"reward": 0.5, "response": {}})),
            ]
        )
        client = _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert client.request.call_count == 2
        assert result == {"reward": 0.5, "response": {}}

    async def test_timeout_fails_once_without_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(side_effect=asyncio.TimeoutError())
        client = _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert client.request.call_count == 1
        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "timed out after 5.0s" in result["error"]

    async def test_non_2xx_becomes_failure_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(500, b"kaboom"))
        _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "HTTP 500" in result["error"] and "kaboom" in result["error"]

    async def test_non_dict_body_becomes_failure_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, b"[1, 2]"))
        _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "expected a JSON object" in result["error"]

    async def test_invalid_json_body_becomes_failure_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, b"not json"))
        _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "not valid JSON" in result["error"]

    async def test_unexpected_exception_becomes_failure_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(side_effect=RuntimeError("surprise"))
        client = _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert client.request.call_count == 1
        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "RuntimeError: surprise" in result["error"]

    @pytest.mark.parametrize(
        "read_exc",
        [ClientPayloadError("Response payload is not completed"), asyncio.TimeoutError()],
        ids=["mid-body disconnect", "deadline during body read"],
    )
    async def test_body_read_failure_becomes_failure_row(
        self, monkeypatch: pytest.MonkeyPatch, read_exc: Exception
    ) -> None:
        # client.request() returns once HEADERS arrive; a failure while reading the BODY must
        # honor the same never-raise contract instead of aborting the whole collection run.
        class FailingReadResponse:
            status = 200
            ok = True

            async def read(self):
                raise read_exc

        request_mock = AsyncMock(return_value=FailingReadResponse())
        _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "reading the response body failed" in result["error"]

    async def test_run_examples_mixed_named_and_url_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        url_request_mock = AsyncMock(
            return_value=_FakeAiohttpResponse(200, orjson.dumps({"reward": 1.0, "response": {}}))
        )
        url_client = _mock_global_client(monkeypatch, url_request_mock)

        named_response = MagicMock()
        named_response.status = 200
        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=named_response)

        async def ok_raise_for_status(_response):
            return None

        async def named_response_json(_response):
            return {"reward": 0.25}

        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", ok_raise_for_status)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", named_response_json)

        setup_calls: list[int] = []

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                setup_calls.append(1)
                return mock_server_client

        rows = [
            {AGENT_REF_KEY_NAME: {"name": "named_agent"}, "responses_create_params": {"input": []}},
            deepcopy(_URL_ROW),
        ]
        results_by_agent: dict = {}
        for future in MockHelper().run_examples(rows):
            row, result = await future
            results_by_agent[row_agent_key(row)] = result

        # Named rows present → ServerClient constructed exactly once; each row took its own path
        assert setup_calls == [1]
        assert results_by_agent["named_agent"] == {"reward": 0.25}
        assert results_by_agent["http://localhost:9000"] == {"reward": 1.0, "response": {}}
        assert mock_server_client.post.call_args.kwargs["server_name"] == "named_agent"
        assert url_client.request.call_count == 1


_AGG_BODY = {"agent_metrics": {"mean/reward": 1.0}, "key_metrics": {"mean/reward": 1.0}, "group_level_metrics": []}


class TestExternalAggregateMetrics:
    async def test_success_sends_model_dumped_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, orjson.dumps(_AGG_BODY)))
        client = _mock_global_client(monkeypatch, request_mock)
        agg_request = AggregateMetricsRequest(verify_responses=[{"reward": 1.0}])

        result = await _post_external_aggregate_metrics("http://localhost:9000", agg_request)

        assert isinstance(result, AggregateMetrics)
        assert result.key_metrics == {"mean/reward": 1.0}
        args, kwargs = client.request.call_args
        assert args == ("POST", "http://localhost:9000/aggregate_metrics")
        assert orjson.loads(kwargs["data"])["verify_responses"] == [{"reward": 1.0}]

    @pytest.mark.parametrize("status", [404, 405, 501])
    async def test_not_implemented_skips(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status: int
    ) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(status, b""))
        _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_aggregate_metrics(
            "http://localhost:9000", AggregateMetricsRequest(verify_responses=[])
        )

        assert result is None
        assert "does not implement /aggregate_metrics" in capsys.readouterr().out

    async def test_connect_error_skips(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        request_mock = AsyncMock(side_effect=ClientConnectorError(MagicMock(), OSError("refused")))
        _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_aggregate_metrics(
            "http://localhost:9000", AggregateMetricsRequest(verify_responses=[])
        )

        assert result is None
        assert "Skipping aggregate metrics" in capsys.readouterr().out

    def test_agent_metric_label(self) -> None:
        assert _agent_metric_label({"url": "http://localhost:9000"}) == "localhost_9000"
        assert _agent_metric_label({"name": "my_agent"}) == "my_agent"
        # Two agents behind one gateway must keep distinct labels
        assert _agent_metric_label({"url": "http://gw:9000/agentA"}) == "gw_9000_agentA"

    async def test_call_aggregate_metrics_pure_url_never_builds_server_client(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, orjson.dumps(_AGG_BODY)))
        _mock_global_client(monkeypatch, request_mock)

        class NoServerClientHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                raise AssertionError("pure agent_url runs must not construct a ServerClient")

        rows = [deepcopy(_URL_ROW)]
        results = [{"reward": 1.0, "response": {"usage": {"total_tokens": 3}}}]
        metrics_fpath = await NoServerClientHelper()._call_aggregate_metrics(
            results, rows, tmp_path / "rollouts.jsonl"
        )

        entries = json.loads(metrics_fpath.read_text())
        assert entries == [
            {
                AGENT_REF_KEY_NAME: {"url": "http://localhost:9000"},
                "agent_metrics": {"mean/reward": 1.0},
                "key_metrics": {"mean/reward": 1.0},
                "group_level_metrics": [],
            }
        ]

    async def test_call_aggregate_metrics_url_agent_skip_leaves_named_agents_intact(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # External agent 404s /aggregate_metrics; the named agent still gets its entry.
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(404, b""))
        _mock_global_client(monkeypatch, request_mock)

        agg = AggregateMetrics(agent_metrics={"mean/reward": 0.0}, key_metrics={}, group_level_metrics=[])
        named_response = AsyncMock()
        named_response.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
        named_response.status = 200
        named_response.raise_for_status = MagicMock()
        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=named_response)

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                return mock_server_client

        rows = [deepcopy(_URL_ROW), {AGENT_REF_KEY_NAME: {"name": "named_agent"}}]
        results = [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}]
        metrics_fpath = await MockHelper()._call_aggregate_metrics(results, rows, tmp_path / "rollouts.jsonl")

        entries = json.loads(metrics_fpath.read_text())
        assert [e[AGENT_REF_KEY_NAME] for e in entries] == [{"name": "named_agent"}]
        assert mock_server_client.post.call_args.kwargs["server_name"] == "named_agent"


class _ExternalAgentASGIAdapter:
    """Bridges the aiohttp client call shape onto an in-process FastAPI app (no sockets)."""

    def __init__(self, app):
        from fastapi.testclient import TestClient

        self._client = TestClient(app, raise_server_exceptions=False)
        self.timeouts: list = []

    async def request(self, method: str, url: str, data=None, headers=None, timeout=None, allow_redirects=True):
        self.timeouts.append(timeout)
        response = self._client.request(method, urlparse(url).path, content=data, headers=headers or {})
        return _FakeAiohttpResponse(response.status_code, response.content)


class TestAgentUrlRunFromConfig:
    def _install_external_agent(self, monkeypatch: pytest.MonkeyPatch) -> tuple[_ExternalAgentASGIAdapter, list]:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI()
        received_rows: list = []

        @app.post("/run")
        async def run(request: Request):
            row = await request.json()
            received_rows.append(row)
            if row.get("x") == "fail":
                return JSONResponse(status_code=500, content={"detail": "agent exploded"})
            return {"reward": 1.0, "response": {"usage": {"total_tokens": 3}}}

        adapter = _ExternalAgentASGIAdapter(app)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_aiohttp_client", lambda: adapter)
        monkeypatch.setattr(nemo_gym.rollout_collection, "_EXTERNAL_AGENT_RETRY_SLEEP_SECS", 0)
        return adapter, received_rows

    def _config(self, tmp_path: Path, rows: list[dict], **kwargs) -> RolloutCollectionConfig:
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            agent_url="http://localhost:9000",
            upload_rollouts_to_wandb=False,
            **kwargs,
        )

    class _NoServerClientHelper(RolloutCollectionHelper):
        def setup_server_client(self, *args, **kwargs):
            raise AssertionError("pure agent_url runs must not construct a ServerClient")

    async def test_end_to_end_success_with_aggregate_skip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        adapter, received_rows = self._install_external_agent(monkeypatch)
        rows = [
            {"responses_create_params": {"input": []}, "x": str(i), "verifier_metadata": {"answer": "42"}}
            for i in range(3)
        ]
        config = self._config(tmp_path, rows, agent_run_timeout_secs=77.0)

        results = await self._NoServerClientHelper().run_from_config(config)

        assert len(results) == 3
        assert all(r["reward"] == 1.0 for r in results)
        assert all(r[AGENT_REF_KEY_NAME] == {"url": "http://localhost:9000"} for r in results)

        # Intentional (issue #1305): the external agent received the full row, answer key included
        assert all(r["verifier_metadata"] == {"answer": "42"} for r in received_rows)
        # Config → request timeout plumbing: every /run call carried agent_run_timeout_secs
        run_timeouts = adapter.timeouts[:3]
        assert all(t.total == 77.0 for t in run_timeouts)

        written = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").open()]
        assert len(written) == 3
        out = capsys.readouterr().out
        # The external agent app defines no /aggregate_metrics route → graceful skip
        assert "does not implement /aggregate_metrics" in out
        # num_samples_in_parallel unset → concurrency is capped at the per-host connection limit
        assert "Capping concurrency at 1024" in out
        assert json.loads((tmp_path / "rollouts_aggregate_metrics.json").read_text()) == []

    async def test_failures_route_to_sidecar_not_main_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        self._install_external_agent(monkeypatch)
        rows = [
            {"responses_create_params": {"input": []}, "x": "ok"},
            {"responses_create_params": {"input": []}, "x": "fail"},
        ]
        config = self._config(tmp_path, rows, num_samples_in_parallel=2)

        results = await self._NoServerClientHelper().run_from_config(config)

        assert len(results) == 2
        written = [json.loads(line) for line in (tmp_path / "rollouts.jsonl").open()]
        assert len(written) == 1 and written[0]["reward"] == 1.0

        sidecar = [json.loads(line) for line in (tmp_path / "rollouts_failures.jsonl").open()]
        assert len(sidecar) == 1
        assert sidecar[0][NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "HTTP 500" in sidecar[0]["error"]
        assert sidecar[0][AGENT_REF_KEY_NAME] == {"url": "http://localhost:9000"}
        # Bounded num_samples_in_parallel (<= per-host cap) → no capping needed
        assert "Capping concurrency at" not in capsys.readouterr().out


class TestAgentUrlResume:
    async def test_resume_restamps_changed_url(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        output_fpath = tmp_path / "rollouts.jsonl"
        materialized = [
            {
                "responses_create_params": {"input": []},
                AGENT_REF_KEY_NAME: {"url": "http://old-host:1111"},
                TASK_INDEX_KEY_NAME: i,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
            for i in range(2)
        ]
        # A named row must pass through the re-stamp untouched (the guard's None arm)
        materialized.append(
            {
                "responses_create_params": {"input": []},
                AGENT_REF_KEY_NAME: {"name": "named_agent"},
                TASK_INDEX_KEY_NAME: 2,
                ROLLOUT_INDEX_KEY_NAME: 0,
            }
        )
        done_result = {
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: 0,
            "response": {"usage": {}},
            AGENT_REF_KEY_NAME: {"url": "http://old-host:1111"},
        }
        output_fpath.write_text(json.dumps(done_result) + "\n")

        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "unused_input.jsonl"),
            output_jsonl_fpath=str(output_fpath),
            agent_url="http://localhost:9000",
            resume_from_cache=True,
            upload_rollouts_to_wandb=False,
        )
        with config.materialized_jsonl_fpath.open("wb") as f:
            for row in materialized:
                f.write(orjson.dumps(row) + b"\n")

        dispatched_agents: list[str] = []
        aggregation_row_keys: list = []

        class MockHelper(RolloutCollectionHelper):
            def run_examples(self, examples, *args, **kwargs):
                futures = []
                for example in examples:
                    dispatched_agents.append(row_agent_key(example))
                    future = Future()
                    future.set_result((example, {"response": {"usage": {}}}))
                    futures.append(future)
                return futures

            async def _call_aggregate_metrics(self, results, rows, output_fpath):
                aggregation_row_keys.extend(sorted(map(row_agent_key, rows)))
                return None

        results = await MockHelper().run_from_config(config)

        # Only pending rows are re-dispatched; the url row to the NEW url, the named row untouched
        assert dispatched_agents == ["http://localhost:9000", "named_agent"]
        out = capsys.readouterr().out
        assert "Re-pointed 1 pending and 1 completed rows to agent_url=http://localhost:9000" in out
        # The already-done row keeps its historical ref in the returned RESULTS (on-disk artifact)...
        assert len(results) == 3
        assert results[0][AGENT_REF_KEY_NAME] == {"url": "http://old-host:1111"}
        assert results[1][AGENT_REF_KEY_NAME] == {"url": "http://localhost:9000"}
        assert results[2][AGENT_REF_KEY_NAME] == {"name": "named_agent"}
        # ...but aggregation groups ALL url rollouts under the live url (no dead-host group)
        assert aggregation_row_keys == ["http://localhost:9000", "http://localhost:9000", "named_agent"]


class TestNormalizeAgentUrlHelper:
    def test_strips_trailing_slash_and_whitespace(self) -> None:
        assert _normalize_agent_url(" http://localhost:9000/ ") == "http://localhost:9000"

    def test_https_accepted(self) -> None:
        assert _normalize_agent_url("https://agents.example.com") == "https://agents.example.com"

    def test_row_agent_key_shapes(self) -> None:
        assert row_agent_key({AGENT_REF_KEY_NAME: {"name": "a"}}) == "a"
        assert row_agent_key({AGENT_REF_KEY_NAME: {"url": "http://h:1"}}) == "http://h:1"
        assert row_agent_key({AGENT_REF_KEY_NAME: None}) is None
        assert row_agent_key({}) is None
        assert row_agent_key({AGENT_REF_KEY_NAME: "not-a-dict"}) is None


class TestExternalAggregateCoverageGaps:
    async def test_external_aggregate_http_500_skips(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(500, b"agent exploded"))
        _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_aggregate_metrics(
            "http://localhost:9000", AggregateMetricsRequest(verify_responses=[])
        )

        assert result is None
        out = capsys.readouterr().out
        assert "Skipping aggregate metrics" in out and "HTTP 500" in out and "agent exploded" in out

    async def test_call_aggregate_metrics_skips_rows_without_any_ref(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, orjson.dumps(_AGG_BODY)))
        _mock_global_client(monkeypatch, request_mock)

        rows = [deepcopy(_URL_ROW), {"responses_create_params": {"input": []}}]  # second row: no agent_ref at all
        results = [{"reward": 1.0, "response": {}}, {"reward": 0.0, "response": {}}]
        metrics_fpath = await RolloutCollectionHelper()._call_aggregate_metrics(
            results, rows, tmp_path / "rollouts.jsonl"
        )

        entries = json.loads(metrics_fpath.read_text())
        assert [e[AGENT_REF_KEY_NAME] for e in entries] == [{"url": "http://localhost:9000"}]

    async def test_call_aggregate_metrics_pure_url_never_touches_client_setup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Pure agent_url runs rely on get_global_aiohttp_client()'s lazy self-initialization
        # (which honors user connector overrides from the global config) — the aggregate path
        # must not install its own client config.
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(404, b""))
        _mock_global_client(monkeypatch, request_mock)
        set_global_mock = MagicMock()
        monkeypatch.setattr(nemo_gym.rollout_collection, "set_global_aiohttp_client", set_global_mock)

        await RolloutCollectionHelper()._call_aggregate_metrics(
            [{"reward": 1.0, "response": {}}], [deepcopy(_URL_ROW)], tmp_path / "rollouts.jsonl"
        )

        set_global_mock.assert_not_called()


class TestExternalAgentEdgeCases:
    """Failure-containment and identity edge cases of external-agent dispatch."""

    _REQUIRED = {"input_jsonl_fpath": "in.jsonl", "output_jsonl_fpath": "out.jsonl"}

    @pytest.mark.parametrize("bad_url", ["http://h:9000?token=abc", "http://h:9000#frag"])
    def test_agent_url_with_query_or_fragment_rejected(self, bad_url: str) -> None:
        with pytest.raises(ValidationError, match="query string or fragment"):
            RolloutCollectionConfig(agent_url=bad_url, **self._REQUIRED)

    async def test_response_missing_required_keys_becomes_failure_row(self, monkeypatch: pytest.MonkeyPatch) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, orjson.dumps({"reward": 1.0})))
        _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "missing required key(s) ['response']" in result["error"]

    async def test_sentinel_failure_response_is_exempt_from_shape_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An external agent reporting its own failure via the sentinel contract carries no reward
        body = {NG_FAILURE_CLASS_KEY: "agent_side_oom", "error": "sandbox died"}
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, orjson.dumps(body)))
        _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert result == body

    async def test_aggregate_payload_excludes_failure_rows(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, orjson.dumps(_AGG_BODY)))
        client = _mock_global_client(monkeypatch, request_mock)

        rows = [deepcopy(_URL_ROW), deepcopy(_URL_ROW)]
        results = [
            {"reward": 1.0, "response": {"usage": {"total_tokens": 3}}},
            {NG_FAILURE_CLASS_KEY: EXTERNAL_AGENT_FAILURE_CLASS, "error": "boom"},
        ]
        await RolloutCollectionHelper()._call_aggregate_metrics(results, rows, tmp_path / "rollouts.jsonl")

        payload = orjson.loads(client.request.call_args.kwargs["data"])
        assert len(payload["verify_responses"]) == 1
        assert NG_FAILURE_CLASS_KEY not in payload["verify_responses"][0]

    async def test_aggregate_skipped_when_only_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        request_mock = AsyncMock(return_value=_FakeAiohttpResponse(200, orjson.dumps(_AGG_BODY)))
        client = _mock_global_client(monkeypatch, request_mock)

        rows = [deepcopy(_URL_ROW)]
        results = [{NG_FAILURE_CLASS_KEY: EXTERNAL_AGENT_FAILURE_CLASS, "error": "boom"}]
        metrics_fpath = await RolloutCollectionHelper()._call_aggregate_metrics(
            results, rows, tmp_path / "rollouts.jsonl"
        )

        assert metrics_fpath is None
        client.request.assert_not_called()
        assert "No successful rollouts to aggregate" in capsys.readouterr().out

    async def test_resume_without_agent_url_rejects_frozen_url_rows(self, tmp_path: Path) -> None:
        output_fpath = tmp_path / "rollouts.jsonl"
        output_fpath.write_text("")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "unused.jsonl"),
            output_jsonl_fpath=str(output_fpath),
            resume_from_cache=True,
            upload_rollouts_to_wandb=False,
        )
        materialized = [
            {
                "responses_create_params": {"input": []},
                AGENT_REF_KEY_NAME: {"name": "named_agent"},
                TASK_INDEX_KEY_NAME: 0,
                ROLLOUT_INDEX_KEY_NAME: 0,
            },
            {
                "responses_create_params": {"input": []},
                AGENT_REF_KEY_NAME: {"url": "http://old-host:1111"},
                TASK_INDEX_KEY_NAME: 1,
                ROLLOUT_INDEX_KEY_NAME: 0,
            },
        ]
        with config.materialized_jsonl_fpath.open("wb") as f:
            for row in materialized:
                f.write(orjson.dumps(row) + b"\n")

        with pytest.raises(ValueError, match=r"\+agent_url was not provided"):
            await RolloutCollectionHelper().run_from_config(config)

    async def test_aggregate_both_key_ref_groups_and_labels_by_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Both-key refs are rejected at preprocess, but shard aggregation and hand-built
        # artifacts bypass preprocessing — grouping, dispatch, and the emitted ref must all
        # follow row_agent_key's name-first precedence.
        url_client = _mock_global_client(monkeypatch, AsyncMock())

        agg = AggregateMetrics(agent_metrics={"mean/reward": 1.0}, key_metrics={}, group_level_metrics=[])
        named_response = AsyncMock()
        named_response.read = AsyncMock(return_value=orjson.dumps(agg.model_dump()))
        named_response.status = 200
        named_response.raise_for_status = MagicMock()
        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=named_response)

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                return mock_server_client

        rows = [{AGENT_REF_KEY_NAME: {"name": "my_agent", "url": "http://sneaky:1"}}]
        results = [{"reward": 1.0, "response": {}}]
        metrics_fpath = await MockHelper()._call_aggregate_metrics(results, rows, tmp_path / "rollouts.jsonl")

        entries = json.loads(metrics_fpath.read_text())
        assert entries[0][AGENT_REF_KEY_NAME] == {"name": "my_agent"}
        assert mock_server_client.post.call_args.kwargs["server_name"] == "my_agent"
        url_client.request.assert_not_called()


class TestOwedFixes:
    """Pins for the review-follow-up fixes: malformed refs, bare URL delimiters, failure summary."""

    @pytest.mark.parametrize(
        "bad_ref",
        [None, {}, {"name": None}, "not-a-dict", ["not", "a", "dict"], 5],
        ids=["null", "empty-dict", "name-null", "str-ref", "list-ref", "int-ref"],
    )
    def test_preprocess_malformed_ref_raises_bulk_missing_agent(self, tmp_path: Path, bad_ref) -> None:
        # Every malformed shape must hit the consolidated missing-agent error — never an
        # AttributeError mid-loop (the pre-agent_url code crashed on truthy non-dict refs).
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(
            json.dumps({"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: bad_ref}) + "\n"
        )
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath), output_jsonl_fpath=str(tmp_path / "out.jsonl")
        )
        with pytest.raises(ValueError, match="No agent specified for rows"):
            RolloutCollectionHelper()._preprocess_rows_from_config(config)

    @pytest.mark.parametrize(
        "bad_url", ["http://h:9000?", "http://h:9000#", "http://h:9000?x=1", "http://h:9000#frag"]
    )
    def test_agent_url_bare_or_full_delimiters_rejected(self, bad_url: str) -> None:
        with pytest.raises(ValidationError, match="query string or fragment"):
            RolloutCollectionConfig(agent_url=bad_url, input_jsonl_fpath="in.jsonl", output_jsonl_fpath="out.jsonl")

    def test_fallback_limit_matches_config_default(self) -> None:
        from nemo_gym.rollout_collection import _FALLBACK_PER_HOST_CONNECTION_LIMIT
        from nemo_gym.server_utils import GlobalAIOHTTPAsyncClientConfig

        assert (
            _FALLBACK_PER_HOST_CONNECTION_LIMIT
            == GlobalAIOHTTPAsyncClientConfig().global_aiohttp_connector_limit_per_host
        )

    async def test_end_of_run_failure_summary_printed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.post("/run")
        async def run(request: Request):
            row = await request.json()
            if row.get("x") == "fail":
                return JSONResponse(status_code=500, content={"detail": "boom"})
            return {"reward": 1.0, "response": {"usage": {}}}

        adapter = _ExternalAgentASGIAdapter(app)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_aiohttp_client", lambda: adapter)
        monkeypatch.setattr(nemo_gym.rollout_collection, "_EXTERNAL_AGENT_RETRY_SLEEP_SECS", 0)

        input_fpath = tmp_path / "input.jsonl"
        rows = [
            {"responses_create_params": {"input": []}, "x": "ok"},
            {"responses_create_params": {"input": []}, "x": "fail"},
        ]
        input_fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            agent_url="http://localhost:9000",
            upload_rollouts_to_wandb=False,
            num_samples_in_parallel=2,
        )

        class NoServerClientHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                raise AssertionError("pure agent_url runs must not construct a ServerClient")

        await NoServerClientHelper().run_from_config(config)

        out = capsys.readouterr().out
        assert "WARNING: 1 rollout(s) failed this run" in out
        assert "rollouts_failures.jsonl" in out

    async def test_no_failure_summary_when_all_succeed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from fastapi import FastAPI

        app = FastAPI()

        @app.post("/run")
        async def run():
            return {"reward": 1.0, "response": {"usage": {}}}

        adapter = _ExternalAgentASGIAdapter(app)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_aiohttp_client", lambda: adapter)

        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(json.dumps({"responses_create_params": {"input": []}}) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            agent_url="http://localhost:9000",
            upload_rollouts_to_wandb=False,
        )

        class NoServerClientHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                raise AssertionError("pure agent_url runs must not construct a ServerClient")

        await NoServerClientHelper().run_from_config(config)
        assert "WARNING:" not in capsys.readouterr().out


class TestFreshEyesFixes:
    """Pins for the fresh-agent DA round: 3xx handling, userinfo, type gates, retry classes,
    config-aware concurrency cap, sidecar lifecycle, resume guards, and dispatch precedence."""

    async def test_3xx_becomes_failure_row_with_location(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = _FakeAiohttpResponse(301, b"<html>moved</html>", headers={"Location": "https://h:9443/run"})
        _mock_global_client(monkeypatch, AsyncMock(return_value=response))

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert "HTTP 301" in result["error"] and "redirect to https://h:9443/run" in result["error"]

    async def test_aggregate_3xx_skips(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _mock_global_client(monkeypatch, AsyncMock(return_value=_FakeAiohttpResponse(302, b"")))
        result = await _post_external_aggregate_metrics(
            "http://localhost:9000", AggregateMetricsRequest(verify_responses=[])
        )
        assert result is None
        assert "HTTP 302" in capsys.readouterr().out

    async def test_aggregate_malformed_200_body_skips(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _mock_global_client(monkeypatch, AsyncMock(return_value=_FakeAiohttpResponse(200, b"<html>gateway</html>")))
        result = await _post_external_aggregate_metrics(
            "http://localhost:9000", AggregateMetricsRequest(verify_responses=[])
        )
        assert result is None
        assert "Skipping aggregate metrics" in capsys.readouterr().out

    def test_agent_url_userinfo_rejected_without_echoing_secret(self) -> None:
        with pytest.raises(ValidationError) as excinfo:
            RolloutCollectionConfig(
                agent_url="http://alice:hunter2@host:9000",
                input_jsonl_fpath="in.jsonl",
                output_jsonl_fpath="out.jsonl",
            )
        message = str(excinfo.value)
        assert "must not embed credentials" in message
        assert "hunter2" not in message

    @pytest.mark.parametrize(
        "bad_result, expected_fragment",
        [
            ({"reward": 1.0, "response": None}, "invalid types"),
            ({"reward": 1.0, "response": "trace text"}, "invalid types"),
            ({"reward": None, "response": {}}, "invalid types"),
            ({"reward": "high", "response": {}}, "invalid types"),
        ],
        ids=["response-null", "response-str", "reward-null", "reward-str"],
    )
    async def test_wrong_typed_success_becomes_failure_row(
        self, monkeypatch: pytest.MonkeyPatch, bad_result: dict, expected_fragment: str
    ) -> None:
        _mock_global_client(monkeypatch, AsyncMock(return_value=_FakeAiohttpResponse(200, orjson.dumps(bad_result))))
        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)
        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS
        assert expected_fragment in result["error"]

    async def test_client_os_error_retried_bounded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from aiohttp import ClientOSError

        request_mock = AsyncMock(side_effect=ClientOSError("Connection reset by peer"))
        client = _mock_global_client(monkeypatch, request_mock)

        result = await _post_external_agent_run(deepcopy(_URL_ROW), timeout_secs=5.0)

        assert client.request.call_count == 3
        assert result[NG_FAILURE_CLASS_KEY] == EXTERNAL_AGENT_FAILURE_CLASS

    async def test_no_persist_response_dropped_without_record(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from fastapi import FastAPI, Request

        app = FastAPI()

        @app.post("/run")
        async def run(request: Request):
            row = await request.json()
            if row.get("x") == "kill":
                return {"_ng_no_persist": True, "_ng_failure_class": "kill_shaped"}
            return {"reward": 1.0, "response": {"usage": {}}}

        adapter = _ExternalAgentASGIAdapter(app)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_aiohttp_client", lambda: adapter)

        input_fpath = tmp_path / "input.jsonl"
        rows = [
            {"responses_create_params": {"input": []}, "x": "ok"},
            {"responses_create_params": {"input": []}, "x": "kill"},
        ]
        input_fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            agent_url="http://localhost:9000",
            upload_rollouts_to_wandb=False,
        )
        await RolloutCollectionHelper().run_from_config(config)

        # no-persist rows: absent from BOTH files (absence is the resume signal), counted in summary
        assert len([json.loads(line) for line in (tmp_path / "rollouts.jsonl").open()]) == 1
        assert (tmp_path / "rollouts_failures.jsonl").read_bytes() == b""
        out = capsys.readouterr().out
        assert "1 were dropped without a record" in out

    async def test_cap_uses_run_config_override_pre_init(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from fastapi import FastAPI
        from omegaconf import DictConfig

        app = FastAPI()

        @app.post("/run")
        async def run():
            return {"reward": 1.0, "response": {"usage": {}}}

        adapter = _ExternalAgentASGIAdapter(app)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_aiohttp_client", lambda: adapter)
        monkeypatch.setattr(nemo_gym.rollout_collection, "is_global_config_dict_set", lambda: True)
        monkeypatch.setattr(
            nemo_gym.rollout_collection,
            "get_global_config_dict",
            lambda: DictConfig({"global_aiohttp_connector_limit_per_host": 16}),
        )

        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(json.dumps({"responses_create_params": {"input": []}}) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            agent_url="http://localhost:9000",
            upload_rollouts_to_wandb=False,
            num_samples_in_parallel=100,
        )
        await RolloutCollectionHelper().run_from_config(config)

        out = capsys.readouterr().out
        # The run-config override (16), not the class default (1024), bounds the run —
        # asserting the EFFECTIVE concurrency line, not just the cap notice.
        assert "Capping concurrency at 16" in out
        assert "Querying with 16 concurrent requests" in out

    async def test_cap_reads_live_connector_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from nemo_gym.rollout_collection import _effective_per_host_connection_limit

        class FakeConnector:
            limit_per_host = 7

        class FakeClient:
            connector = FakeConnector()

        monkeypatch.setattr(nemo_gym.rollout_collection, "is_global_aiohttp_client_setup", lambda: True)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_aiohttp_client", lambda: FakeClient())
        assert _effective_per_host_connection_limit() == 7

        FakeConnector.limit_per_host = 0  # aiohttp uses 0 to mean unlimited
        assert _effective_per_host_connection_limit() is None

    async def test_fresh_run_clears_stale_failures_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from fastapi import FastAPI

        app = FastAPI()

        @app.post("/run")
        async def run():
            return {"reward": 1.0, "response": {"usage": {}}}

        adapter = _ExternalAgentASGIAdapter(app)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_aiohttp_client", lambda: adapter)

        # Stale sidecar from an earlier, unrelated fresh run: 3 attempts for key (0, 0)
        stale = {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, NG_FAILURE_CLASS_KEY: "external_agent_error"}
        (tmp_path / "rollouts_failures.jsonl").write_text((json.dumps(stale) + "\n") * 3)

        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(json.dumps({"responses_create_params": {"input": []}}) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            agent_url="http://localhost:9000",
            upload_rollouts_to_wandb=False,
        )
        await RolloutCollectionHelper().run_from_config(config)

        # The fresh run cleared the stale attempts; a later resume must not see them.
        assert (tmp_path / "rollouts_failures.jsonl").read_bytes() == b""

    async def test_resume_guard_covers_completed_url_rows(self, tmp_path: Path) -> None:
        # All url rows COMPLETED, none pending: aggregation would still POST to the frozen
        # URL, so resuming without +agent_url must error.
        output_fpath = tmp_path / "rollouts.jsonl"
        row = {
            "responses_create_params": {"input": []},
            AGENT_REF_KEY_NAME: {"url": "http://old-host:1111"},
            TASK_INDEX_KEY_NAME: 0,
            ROLLOUT_INDEX_KEY_NAME: 0,
        }
        done = dict(row, response={"usage": {}}, reward=1.0)
        output_fpath.write_text(json.dumps(done) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "unused.jsonl"),
            output_jsonl_fpath=str(output_fpath),
            resume_from_cache=True,
            upload_rollouts_to_wandb=False,
        )
        with config.materialized_jsonl_fpath.open("wb") as f:
            f.write(orjson.dumps(row) + b"\n")

        with pytest.raises(ValueError, match=r"\+agent_url was not provided"):
            await RolloutCollectionHelper().run_from_config(config)

    async def test_both_key_ref_dispatches_to_named_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Name-first everywhere: a both-key ref must take the NAMED path, matching
        # row_agent_key/canonical_agent_ref/aggregation precedence.
        url_client = _mock_global_client(monkeypatch, AsyncMock())

        named_response = MagicMock()
        named_response.status = 200
        mock_server_client = MagicMock()
        mock_server_client.post = AsyncMock(return_value=named_response)

        async def ok_raise_for_status(_response):
            return None

        async def named_response_json(_response):
            return {"reward": 0.5}

        monkeypatch.setattr(nemo_gym.rollout_collection, "raise_for_status", ok_raise_for_status)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_response_json", named_response_json)

        class MockHelper(RolloutCollectionHelper):
            def setup_server_client(self, *args, **kwargs):
                return mock_server_client

        rows = [
            {
                AGENT_REF_KEY_NAME: {"name": "my_agent", "url": "http://sneaky:1"},
                "responses_create_params": {"input": []},
            }
        ]
        for future in MockHelper().run_examples(rows):
            row, result = await future

        assert result == {"reward": 0.5}
        assert mock_server_client.post.call_args.kwargs["server_name"] == "my_agent"
        url_client.request.assert_not_called()

    def test_row_error_messages_use_real_row_position(self, tmp_path: Path) -> None:
        rows = [
            {"responses_create_params": {"input": []}},
            {"responses_create_params": {"input": []}},
            {"responses_create_params": {"input": []}, AGENT_REF_KEY_NAME: {"url": "http://other-host:1234"}},
        ]
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            agent_url="http://localhost:9000",
        )
        with pytest.raises(ValueError, match="Row 2 carries agent_ref.url"):
            RolloutCollectionHelper()._preprocess_rows_from_config(config)

    def test_num_repeats_url_key_with_trailing_slash_matches(self, tmp_path: Path) -> None:
        input_fpath = tmp_path / "input.jsonl"
        input_fpath.write_text(json.dumps({"responses_create_params": {"input": []}}) + "\n")
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            agent_url="http://localhost:9000",
            num_repeats={"http://localhost:9000/": 3},
        )
        processed = RolloutCollectionHelper()._preprocess_rows_from_config(config)
        assert len(processed) == 3

    def test_failure_log_sampling_head_then_interval(self, capsys: pytest.CaptureFixture[str]) -> None:
        from nemo_gym.rollout_collection import _external_agent_failure_result

        nemo_gym.rollout_collection._NUM_EXTERNAL_AGENT_FAILURES = 0
        for _ in range(7):
            _external_agent_failure_result("http://h:1/run", "boom")
        out = capsys.readouterr().out
        assert "failure #5" in out
        assert "failure #6" not in out and "failure #7" not in out


class TestExternalFailureResumeContract:
    """End-to-end pin for 'recorded in the failures sidecar and retried on resume'."""

    def _agent(self, monkeypatch: pytest.MonkeyPatch, behavior: dict):
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse

        app = FastAPI()

        @app.post("/run")
        async def run(request: Request):
            body = behavior["response"]
            if isinstance(body, int):
                return JSONResponse(status_code=body, content={"detail": "boom"})
            return body

        adapter = _ExternalAgentASGIAdapter(app)
        monkeypatch.setattr(nemo_gym.rollout_collection, "get_global_aiohttp_client", lambda: adapter)
        monkeypatch.setattr(nemo_gym.rollout_collection, "_EXTERNAL_AGENT_RETRY_SLEEP_SECS", 0)

    def _config(self, tmp_path: Path, resume: bool) -> RolloutCollectionConfig:
        input_fpath = tmp_path / "input.jsonl"
        if not input_fpath.exists():
            input_fpath.write_text(json.dumps({"responses_create_params": {"input": []}}) + "\n")
        return RolloutCollectionConfig(
            input_jsonl_fpath=str(input_fpath),
            output_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            agent_url="http://localhost:9000",
            upload_rollouts_to_wandb=False,
            resume_from_cache=resume,
        )

    async def test_failed_row_is_retried_on_resume_and_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        behavior = {"response": 500}
        self._agent(monkeypatch, behavior)

        # Run 1: the row fails → sidecar attempt 1, main jsonl empty
        await RolloutCollectionHelper().run_from_config(self._config(tmp_path, resume=False))
        assert not (tmp_path / "rollouts.jsonl").read_bytes()
        assert len((tmp_path / "rollouts_failures.jsonl").read_text().splitlines()) == 1

        # Run 2 (resume): agent recovered → the SAME row is re-dispatched and succeeds
        behavior["response"] = {"reward": 1.0, "response": {"usage": {}}}
        results = await RolloutCollectionHelper().run_from_config(self._config(tmp_path, resume=True))
        assert len(results) == 1 and results[0]["reward"] == 1.0
        assert len((tmp_path / "rollouts.jsonl").read_text().splitlines()) == 1

    async def test_attempt_cap_gates_after_max_failures(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        behavior = {"response": 500}
        self._agent(monkeypatch, behavior)

        await RolloutCollectionHelper().run_from_config(self._config(tmp_path, resume=False))
        for _ in range(2):
            await RolloutCollectionHelper().run_from_config(self._config(tmp_path, resume=True))
        assert len((tmp_path / "rollouts_failures.jsonl").read_text().splitlines()) == 3

        # Attempt cap (default 3) reached: resume dispatches nothing even though the agent recovered
        behavior["response"] = {"reward": 1.0, "response": {"usage": {}}}
        capsys.readouterr()
        await RolloutCollectionHelper().run_from_config(self._config(tmp_path, resume=True))
        out = capsys.readouterr().out
        assert "1 hit max_attempts=3" in out
        assert not (tmp_path / "rollouts.jsonl").read_bytes()

    async def test_agent_terminal_flag_is_never_retried(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        behavior = {"response": {NG_FAILURE_CLASS_KEY: "task_impossible", "_ng_failure_terminal": True}}
        self._agent(monkeypatch, behavior)

        await RolloutCollectionHelper().run_from_config(self._config(tmp_path, resume=False))
        behavior["response"] = {"reward": 1.0, "response": {"usage": {}}}
        capsys.readouterr()
        await RolloutCollectionHelper().run_from_config(self._config(tmp_path, resume=True))
        out = capsys.readouterr().out
        assert "1 sidecar-terminal" in out
        assert not (tmp_path / "rollouts.jsonl").read_bytes()

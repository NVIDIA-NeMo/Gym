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
from asyncio import Semaphore
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from pydantic import ValidationError

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, SKILLS_REF_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_reverification import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    NG_TERMINAL_KEY,
    CacheKeysByStatus,
    InputRolloutPair,
    OutputPaths,
    RolloutReverificationConfig,
    RolloutReverificationHelper,
    _agent_to_rs_mapping_from_agent_blocks,
    _agent_to_rs_mapping_from_resources_only_config,
    _build_agent_to_resources_server_mapping,
    _build_verify_payload,
    _call_aggregate_metrics,
    _check_reverify_mode,
    _drop_cache_from_payloads,
    _guard_reverify_mode,
    _load_cache_keys_by_status,
    _load_reverified_results,
    _parse_output_line_key,
    _prepare_output_fpaths,
    _prepare_payloads,
    _rollout_verify_debug_summary,
    _run_verification_payloads,
    _yield_inputs_and_rollouts_paired,
    summarize_cache_usage,
)


class TestRolloutReverificationConfig:
    """Field-level validation on RolloutReverificationConfig."""

    def _kwargs(self, **overrides) -> dict:
        return {
            "materialized_inputs_jsonl_fpath": "in.jsonl",
            "rollouts_jsonl_fpath": "r.jsonl",
            "output_jsonl_fpath": "out.jsonl",
            **overrides,
        }

    @pytest.mark.parametrize("field", ["num_samples_in_parallel", "limit"])
    @pytest.mark.parametrize("bad_value", [0, -1])
    def test_non_positive_positive_int_fields_rejected(self, field: str, bad_value: int) -> None:
        """num_samples_in_parallel and limit must be >= 1 (0 is meaningless: Semaphore(0) / re-verify no rows)."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            RolloutReverificationConfig(**self._kwargs(**{field: bad_value}))

    @pytest.mark.parametrize("field", ["num_samples_in_parallel", "limit"])
    def test_positive_int_fields_none_and_positive_allowed(self, field: str) -> None:
        """Omitting the field (None) means unbounded/no-limit; a positive value is kept."""
        assert getattr(RolloutReverificationConfig(**self._kwargs()), field) is None
        assert getattr(RolloutReverificationConfig(**self._kwargs(**{field: 4})), field) == 4

    def test_resume_from_cache_defaults_to_false(self) -> None:
        """resume_from_cache is opt-in: default runs start fresh."""
        assert RolloutReverificationConfig(**self._kwargs()).resume_from_cache is False
        assert RolloutReverificationConfig(**self._kwargs(resume_from_cache=True)).resume_from_cache is True

    def test_overwrite_defaults_to_false(self) -> None:
        """overwrite is opt-in: by default an existing output file is protected (raises)."""
        assert RolloutReverificationConfig(**self._kwargs()).overwrite is False
        assert RolloutReverificationConfig(**self._kwargs(overwrite=True)).overwrite is True


class TestAgentToRsMappingFromAgentBlocks:
    """Tests for RolloutReverificationHelper._agent_to_rs_mapping_from_agent_blocks.

    Covers configs that have responses_api_agents blocks, e.g. benchmarks/birdbench/config.yaml
    """

    def test_single_agent_maps_to_its_resources_server(self) -> None:
        config = {
            "text_to_sql_agent": {
                "responses_api_agents": {
                    "simple_agent": {"resources_server": {"name": "text_to_sql_resources_server"}}
                }
            }
        }
        assert _agent_to_rs_mapping_from_agent_blocks(config) == {"text_to_sql_agent": "text_to_sql_resources_server"}

    def test_multiple_agents_each_map_to_own_resources_server(self) -> None:
        config = {
            "agent_alpha": {"responses_api_agents": {"impl": {"resources_server": {"name": "rs_alpha"}}}},
            "agent_beta": {"responses_api_agents": {"impl": {"resources_server": {"name": "rs_beta"}}}},
        }
        assert _agent_to_rs_mapping_from_agent_blocks(config) == {
            "agent_alpha": "rs_alpha",
            "agent_beta": "rs_beta",
        }

    def test_agent_without_resources_server_name_is_excluded(self) -> None:
        """An agent block missing resources_server.name contributes nothing to the mapping."""
        config = {
            "agent_no_rs": {"responses_api_agents": {"impl": {"resources_server": {}}}},
        }
        assert _agent_to_rs_mapping_from_agent_blocks(config) == {}

    def test_non_agent_blocks_are_ignored(self) -> None:
        """Blocks without responses_api_agents (e.g. resources-only) don't appear in the result."""
        config = {
            "rs_block": {"resources_servers": {"mcqa": {}}},
        }
        assert _agent_to_rs_mapping_from_agent_blocks(config) == {}


class TestAgentToRsMappingFromResourcesOnlyConfig:
    """Tests for RolloutReverificationHelper._agent_to_rs_mapping_from_resources_only_config.

    Covers configs that have only resources_servers blocks with no agent, e.g. resources_servers/mcqa/config.yaml.
    """

    def test_single_resources_server_maps_any_key_to_it(self) -> None:
        config = {"mcqa_resources_server": {"resources_servers": {"mcqa": {"grading_mode": None}}}}
        result = _agent_to_rs_mapping_from_resources_only_config(config)
        assert result["whatever_agent_ran_the_rollout"] == "mcqa_resources_server"
        assert result["another_agent"] == "mcqa_resources_server"

    def test_multiple_resources_servers_raises(self) -> None:
        config = {
            "rs_one": {"resources_servers": {"impl_one": {}}},
            "rs_two": {"resources_servers": {"impl_two": {}}},
        }
        with pytest.raises(ConfigError, match="multiple resources servers"):
            _agent_to_rs_mapping_from_resources_only_config(config)

    def test_no_resources_servers_raises(self) -> None:
        config = {"some_unrelated_block": {"something": "else"}}
        with pytest.raises(ConfigError, match="no resources server found"):
            _agent_to_rs_mapping_from_resources_only_config(config)


class TestBuildAgentToResourcesServerMapping:
    """Tests for _build_agent_to_resources_server_mapping: verifies routing via
    agent blocks when present, falling back to resources-server blocks when not."""

    def test_agent_based_config_routes_via_agent_blocks(self) -> None:
        config = {
            "text_to_sql_agent": {
                "responses_api_agents": {
                    "simple_agent": {"resources_server": {"name": "text_to_sql_resources_server"}}
                }
            }
        }
        assert _build_agent_to_resources_server_mapping(config) == {
            "text_to_sql_agent": "text_to_sql_resources_server"
        }

    def test_resources_only_config_routes_via_resources_server_blocks(self) -> None:
        config = {"mcqa_resources_server": {"resources_servers": {"mcqa": {}}}}
        result = _build_agent_to_resources_server_mapping(config)
        assert result["any_agent"] == "mcqa_resources_server"

    def test_agent_without_rs_name_falls_back_to_resources_server_blocks(self) -> None:
        """Agent blocks with no RS name produce no agent→RS entries; resources-only routing takes over."""
        config = {
            "agent_no_rs": {"responses_api_agents": {"impl": {"resources_server": {}}}},
            "verifier_block": {"resources_servers": {"mcqa": {}}},
        }
        result = _build_agent_to_resources_server_mapping(config)
        assert result["any_agent_name"] == "verifier_block"


class TestRolloutVerifyDebugSummary:
    def test_rollout_verify_debug_summary_compact(self) -> None:
        resources_server_name = "my_rs"
        row = {
            AGENT_REF_KEY_NAME: {"name": "my_agent"},
            TASK_INDEX_KEY_NAME: 12,
            ROLLOUT_INDEX_KEY_NAME: 3,
            "env_specific_metadata": "do not include",
            "responses_create_params": {"input": "large prompt", "tools": ["large schema"]},
        }

        assert _rollout_verify_debug_summary(row, resources_server_name) == {
            "agent_name": "my_agent",
            TASK_INDEX_KEY_NAME: 12,
            ROLLOUT_INDEX_KEY_NAME: 3,
            "resources_server_name": "my_rs",
        }


class TestYieldInputsAndRolloutsPaired:
    def _write_jsonl(self, path: Path, rows: list) -> None:
        path.write_bytes(b"\n".join(orjson.dumps(r) for r in rows) + b"\n")

    def test_pairs_input_and_rollout_by_task_and_rollout_index(self, tmp_path: Path) -> None:
        inputs = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "question": "q0"},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "question": "q1"},
        ]
        rollouts = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0},
        ]
        inputs_path = tmp_path / "inputs.jsonl"
        rollouts_path = tmp_path / "rollouts.jsonl"
        self._write_jsonl(inputs_path, inputs)
        self._write_jsonl(rollouts_path, rollouts)

        pairs = list(_yield_inputs_and_rollouts_paired(inputs_path, rollouts_path))

        assert len(pairs) == 2
        assert pairs[0] == InputRolloutPair(input=inputs[0], rollout=rollouts[0])
        assert pairs[1] == InputRolloutPair(input=inputs[1], rollout=rollouts[1])

    def test_rollouts_in_different_order_than_inputs_still_paired_correctly(self, tmp_path: Path) -> None:
        inputs = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "question": "q0"},
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "question": "q1"},
        ]
        rollouts = [
            {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0},
        ]
        inputs_path = tmp_path / "inputs.jsonl"
        rollouts_path = tmp_path / "rollouts.jsonl"
        self._write_jsonl(inputs_path, inputs)
        self._write_jsonl(rollouts_path, rollouts)

        pairs = list(_yield_inputs_and_rollouts_paired(inputs_path, rollouts_path))

        assert pairs[0].input["question"] == "q1"
        assert pairs[0].rollout["reward"] == 0.0
        assert pairs[1].input["question"] == "q0"
        assert pairs[1].rollout["reward"] == 1.0

    def test_limit_stops_after_n_rollouts(self, tmp_path: Path) -> None:
        inputs = [{TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0, "q": i} for i in range(5)]
        rollouts = [{TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0, "reward": float(i)} for i in range(5)]
        inputs_path = tmp_path / "inputs.jsonl"
        rollouts_path = tmp_path / "rollouts.jsonl"
        self._write_jsonl(inputs_path, inputs)
        self._write_jsonl(rollouts_path, rollouts)

        pairs = list(_yield_inputs_and_rollouts_paired(inputs_path, rollouts_path, limit=3))

        assert len(pairs) == 3
        assert [p.rollout[TASK_INDEX_KEY_NAME] for p in pairs] == [0, 1, 2]

    def test_limit_zero_yields_no_pairs(self, tmp_path: Path) -> None:
        """limit=0 must stop immediately and yield nothing, not iterate all rows."""
        inputs = [{TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0, "q": i} for i in range(3)]
        rollouts = [{TASK_INDEX_KEY_NAME: i, ROLLOUT_INDEX_KEY_NAME: 0, "reward": float(i)} for i in range(3)]
        self._write_jsonl(tmp_path / "inputs.jsonl", inputs)
        self._write_jsonl(tmp_path / "rollouts.jsonl", rollouts)

        pairs = list(
            _yield_inputs_and_rollouts_paired(tmp_path / "inputs.jsonl", tmp_path / "rollouts.jsonl", limit=0)
        )

        assert pairs == []

    def test_rollout_with_no_matching_input_raises_config_error(self, tmp_path: Path) -> None:
        inputs = [{TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "question": "q0"}]
        rollouts = [{TASK_INDEX_KEY_NAME: 99, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0}]
        inputs_path = tmp_path / "inputs.jsonl"
        rollouts_path = tmp_path / "rollouts.jsonl"
        self._write_jsonl(inputs_path, inputs)
        self._write_jsonl(rollouts_path, rollouts)

        with pytest.raises(ConfigError, match="No matching materialized input row"):
            list(_yield_inputs_and_rollouts_paired(inputs_path, rollouts_path))

    def test_multiple_rollouts_per_task_paired_independently(self, tmp_path: Path) -> None:
        inputs = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "question": "q0"},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "question": "q0"},
        ]
        rollouts = [
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0},
            {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 1, "reward": 0.0},
        ]
        inputs_path = tmp_path / "inputs.jsonl"
        rollouts_path = tmp_path / "rollouts.jsonl"
        self._write_jsonl(inputs_path, inputs)
        self._write_jsonl(rollouts_path, rollouts)

        pairs = list(_yield_inputs_and_rollouts_paired(inputs_path, rollouts_path))

        assert len(pairs) == 2
        assert pairs[0].rollout[ROLLOUT_INDEX_KEY_NAME] == 0
        assert pairs[1].rollout[ROLLOUT_INDEX_KEY_NAME] == 1
        # each rollout is keyed by (task_index, rollout_index), so each gets its own input row
        assert pairs[0].input[ROLLOUT_INDEX_KEY_NAME] == 0
        assert pairs[1].input[ROLLOUT_INDEX_KEY_NAME] == 1


class TestBuildVerifyPayload:
    def test_merges_input_row_with_response(self) -> None:
        pair = InputRolloutPair(
            input={"task": "q1", "verifier_metadata": {"answer": 42}},
            rollout={"response": {"output": "hello"}, "reward": 1.0},
        )

        result = _build_verify_payload(pair)

        assert result == {"task": "q1", "verifier_metadata": {"answer": 42}, "response": {"output": "hello"}}

    def test_response_key_overwrites_any_existing_response_in_input(self) -> None:
        pair = InputRolloutPair(
            input={"response": {"output": "stale"}, "task": "q1"},
            rollout={"response": {"output": "fresh"}},
        )

        result = _build_verify_payload(pair)

        assert result["response"] == {"output": "fresh"}

    def test_does_not_mutate_input_row(self) -> None:
        pair = InputRolloutPair(input={"task": "q1"}, rollout={"response": {"output": "x"}})

        _build_verify_payload(pair)

        assert "response" not in pair.input

    def test_extra_rollout_fields_are_not_included(self) -> None:
        pair = InputRolloutPair(
            input={"task": "q1"},
            rollout={"response": {"output": "x"}, "reward": 0.9, "rollout_index": 2},
        )

        result = _build_verify_payload(pair)

        assert set(result.keys()) == {"task", "response"}


# ---------------------------------------------------------------------------
# Cache / resume-from-cache helpers
# ---------------------------------------------------------------------------


class TestParseOutputLineKey:
    def test_extracts_task_and_rollout_index(self) -> None:
        line = orjson.dumps({TASK_INDEX_KEY_NAME: 3, ROLLOUT_INDEX_KEY_NAME: 7, "reward": 1.0})
        assert _parse_output_line_key(line) == (3, 7)

    def test_tolerates_trailing_newline(self) -> None:
        line = orjson.dumps({TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 2}) + b"\n"
        assert _parse_output_line_key(line) == (1, 2)

    def test_missing_indices_return_none_tuple(self) -> None:
        line = orjson.dumps({"reward": 1.0})
        assert _parse_output_line_key(line) == (None, None)


class TestLoadCacheKeysByStatus:
    def _paths(self, tmp_path: Path) -> OutputPaths:
        return OutputPaths(output=tmp_path / "out.jsonl", failures=tmp_path / "out_failures.jsonl")

    def _write(self, path: Path, rows: list[dict]) -> None:
        path.write_bytes(b"".join(orjson.dumps(r) + b"\n" for r in rows))

    def test_no_cache_files_returns_empty_and_prints_skip(self, tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
        cache = _load_cache_keys_by_status(self._paths(tmp_path))
        assert cache.successful_keys == set()
        assert cache.terminal_keys == set()
        assert cache.maxed_out_keys == set()
        assert "Skipping resume_from_cache" in capsys.readouterr().out

    def test_successful_keys_read_from_main_output_file(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        self._write(
            paths.output,
            [
                {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0},
                {TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 2, "reward": 0.0},
            ],
        )
        cache = _load_cache_keys_by_status(paths)
        assert cache.successful_keys == {(0, 0), (1, 2)}

    def test_terminal_keys_flagged_from_failures_sidecar(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        self._write(
            paths.failures,
            [
                {TASK_INDEX_KEY_NAME: 5, ROLLOUT_INDEX_KEY_NAME: 0, NG_TERMINAL_KEY: True},
                {TASK_INDEX_KEY_NAME: 6, ROLLOUT_INDEX_KEY_NAME: 0},  # non-terminal attempt
            ],
        )
        cache = _load_cache_keys_by_status(paths)
        assert cache.terminal_keys == {(5, 0)}

    def test_maxed_out_keys_when_attempts_reach_configured_max(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("nemo_gym.rollout_reverification._get_max_rollout_attempts", lambda: 2)
        paths = self._paths(tmp_path)
        self._write(
            paths.failures,
            [
                {TASK_INDEX_KEY_NAME: 7, ROLLOUT_INDEX_KEY_NAME: 0},  # attempt 1
                {TASK_INDEX_KEY_NAME: 7, ROLLOUT_INDEX_KEY_NAME: 0},  # attempt 2 -> maxed out
                {TASK_INDEX_KEY_NAME: 8, ROLLOUT_INDEX_KEY_NAME: 0},  # attempt 1 -> not maxed out
            ],
        )
        cache = _load_cache_keys_by_status(paths)
        assert cache.maxed_out_keys == {(7, 0)}
        assert (8, 0) not in cache.maxed_out_keys

    def test_failure_rows_without_indices_or_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        paths = self._paths(tmp_path)
        paths.failures.write_bytes(
            orjson.dumps({TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, NG_TERMINAL_KEY: True})
            + b"\n"
            + orjson.dumps({"reward": 0.0})  # no indices -> skipped
            + b"\n"
            + b"\n"  # blank line -> skipped
        )
        cache = _load_cache_keys_by_status(paths)
        assert cache.terminal_keys == {(1, 0)}

    def test_only_failures_file_present_still_loads_sidecar(self, tmp_path: Path) -> None:
        """Output file absent but failures present: successes empty, terminal keys still read."""
        paths = self._paths(tmp_path)
        self._write(paths.failures, [{TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 3, NG_TERMINAL_KEY: True}])
        cache = _load_cache_keys_by_status(paths)
        assert cache.successful_keys == set()
        assert cache.terminal_keys == {(2, 3)}


class TestDropCacheFromPayloads:
    def _payload(self, task: int, rollout: int = 0) -> dict:
        return {TASK_INDEX_KEY_NAME: task, ROLLOUT_INDEX_KEY_NAME: rollout, "data": f"{task}-{rollout}"}

    def test_drops_successful_terminal_and_maxed_out_keeps_the_rest(self) -> None:
        payloads = [self._payload(t) for t in range(5)]  # keys (0,0)..(4,0)
        cache = CacheKeysByStatus(
            successful_keys={(0, 0)},
            terminal_keys={(1, 0)},
            maxed_out_keys={(2, 0)},
        )
        remaining = list(_drop_cache_from_payloads(payloads, cache))
        assert [p[TASK_INDEX_KEY_NAME] for p in remaining] == [3, 4]

    def test_empty_cache_keeps_every_payload(self) -> None:
        payloads = [self._payload(0), self._payload(1)]
        cache = CacheKeysByStatus(successful_keys=set(), terminal_keys=set(), maxed_out_keys=set())
        assert list(_drop_cache_from_payloads(payloads, cache)) == payloads

    def test_drops_at_rollout_index_granularity_within_a_single_task(self) -> None:
        """The cache key is the full (task, rollout) tuple: caching (0,0) must NOT drop (0,1)."""
        payloads = [self._payload(0, 0), self._payload(0, 1), self._payload(0, 2)]
        cache = CacheKeysByStatus(successful_keys={(0, 0)}, terminal_keys=set(), maxed_out_keys=set())
        remaining = list(_drop_cache_from_payloads(payloads, cache))
        assert [p[ROLLOUT_INDEX_KEY_NAME] for p in remaining] == [1, 2]


class TestSummarizeCacheUsage:
    def test_prints_the_expected_counts(self, capsys: pytest.CaptureFixture) -> None:
        cache = CacheKeysByStatus(
            successful_keys={(0, 0)},
            terminal_keys={(1, 0)},
            maxed_out_keys={(2, 0)},
        )
        summarize_cache_usage(cache, all_payloads=[{}] * 10, filtered_payloads=[{}] * 7)
        out = capsys.readouterr().out
        assert "10 total rows to be re-verified" in out
        assert "1 rows already done" in out
        assert "7 rows that still need to be run" in out


class TestPreparePayloads:
    """Tests for _prepare_payloads (the paths are already resolved by run_from_config)."""

    def _paths(self, tmp_path: Path) -> OutputPaths:
        return OutputPaths(output=tmp_path / "out.jsonl", failures=tmp_path / "out_failures.jsonl")

    def _pair(self, task: int, rollout: int = 0) -> InputRolloutPair:
        return InputRolloutPair(
            input={TASK_INDEX_KEY_NAME: task, ROLLOUT_INDEX_KEY_NAME: rollout, "q": task},
            rollout={"response": {"out": task}},
        )

    def _patch_yield(self, monkeypatch: pytest.MonkeyPatch, pairs: list, capture: dict | None = None) -> None:
        def fake_yield(inputs_path, rollouts_path, limit=None):
            if capture is not None:
                capture["inputs_path"] = inputs_path
                capture["rollouts_path"] = rollouts_path
                capture["limit"] = limit
            return iter(pairs)

        monkeypatch.setattr("nemo_gym.rollout_reverification._yield_inputs_and_rollouts_paired", fake_yield)

    def test_returns_one_payload_per_pair_when_not_resuming(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_yield(monkeypatch, [self._pair(0), self._pair(1), self._pair(2)])

        payloads = _prepare_payloads(
            tmp_path / "in.jsonl", tmp_path / "r.jsonl", self._paths(tmp_path), resume_from_cache=False
        )

        assert [p[TASK_INDEX_KEY_NAME] for p in payloads] == [0, 1, 2]
        assert all(p["response"] == {"out": p[TASK_INDEX_KEY_NAME]} for p in payloads)

    def test_limit_is_forwarded_to_yield_pairs(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        capture: dict = {}
        self._patch_yield(monkeypatch, [], capture)

        _prepare_payloads(
            tmp_path / "in.jsonl", tmp_path / "r.jsonl", self._paths(tmp_path), resume_from_cache=False, limit=7
        )

        assert capture["limit"] == 7

    def test_paths_are_passed_through_without_being_swapped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        capture: dict = {}
        self._patch_yield(monkeypatch, [], capture)

        _prepare_payloads(
            tmp_path / "materialized.jsonl",
            tmp_path / "rollouts.jsonl",
            self._paths(tmp_path),
            resume_from_cache=False,
        )

        assert capture["inputs_path"] == tmp_path / "materialized.jsonl"
        assert capture["rollouts_path"] == tmp_path / "rollouts.jsonl"

    def test_passes_full_pair_to_build_verify_payload(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        pair = self._pair(0)
        self._patch_yield(monkeypatch, [pair])
        build_calls: list = []
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_verify_payload",
            lambda p: build_calls.append(p) or {TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
        )

        _prepare_payloads(tmp_path / "in.jsonl", tmp_path / "r.jsonl", self._paths(tmp_path), resume_from_cache=False)

        assert len(build_calls) == 1
        assert build_calls[0] is pair

    def test_resume_from_cache_drops_already_completed_keys(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """With an existing output file marking (0,0) done, resuming re-verifies only the missing keys."""
        paths = self._paths(tmp_path)
        paths.output.write_bytes(orjson.dumps({TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}) + b"\n")
        self._patch_yield(monkeypatch, [self._pair(0), self._pair(1)])

        payloads = _prepare_payloads(tmp_path / "in.jsonl", tmp_path / "r.jsonl", paths, resume_from_cache=True)

        assert [p[TASK_INDEX_KEY_NAME] for p in payloads] == [1]

    def test_resume_from_cache_with_no_cache_files_keeps_all(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_yield(monkeypatch, [self._pair(0), self._pair(1)])

        payloads = _prepare_payloads(
            tmp_path / "in.jsonl", tmp_path / "r.jsonl", self._paths(tmp_path), resume_from_cache=True
        )

        assert [p[TASK_INDEX_KEY_NAME] for p in payloads] == [0, 1]


class TestRunVerificationPayloads:
    """Tests for _run_verification_payloads: routing, success/error paths, and semaphore enforcement."""

    def _make_row(self, agent_name: str, task_idx: int = 0) -> dict:
        return {AGENT_REF_KEY_NAME: {"name": agent_name}, TASK_INDEX_KEY_NAME: task_idx, ROLLOUT_INDEX_KEY_NAME: 0}

    def _patch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        mock_client: MagicMock,
        agent_to_rs: dict,
        raise_for_status_side_effect=None,
        response_json: dict | None = None,
    ) -> None:
        monkeypatch.setattr("nemo_gym.rollout_reverification.setup_server_client", lambda: mock_client)
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping", lambda _: agent_to_rs
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.raise_for_status",
            AsyncMock(side_effect=raise_for_status_side_effect),
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.get_response_json",
            AsyncMock(return_value=response_json or {}),
        )

    async def _collect(self, futures) -> list:
        results = []
        for fut in futures:
            results.append(await fut)
        return results

    async def test_returns_row_and_response_json_for_each_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        rows = [self._make_row("agent_a", i) for i in range(3)]
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        mock_client.post = AsyncMock(return_value=MagicMock())
        self._patch(monkeypatch, mock_client, {"agent_a": "rs_a"}, response_json={"reward": 0.5})

        results = await self._collect(_run_verification_payloads(rows))

        assert len(results) == 3
        returned_rows = [r for r, _ in results]
        for row in rows:
            assert row in returned_rows
        for _, resp in results:
            assert resp == {"reward": 0.5}

    async def test_posts_to_resources_server_matching_agent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row_a = self._make_row("agent_a")
        row_b = self._make_row("agent_b")
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        posted_to: list[tuple[str, str]] = []

        async def capture_post(server_name, url_path, json):  # noqa: ARG001
            posted_to.append((server_name, json[AGENT_REF_KEY_NAME]["name"]))
            return MagicMock()

        mock_client.post = capture_post
        self._patch(monkeypatch, mock_client, {"agent_a": "rs_a", "agent_b": "rs_b"})

        await self._collect(_run_verification_payloads([row_a, row_b]))

        assert set(posted_to) == {("rs_a", "agent_a"), ("rs_b", "agent_b")}

    async def test_failed_response_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        row = self._make_row("agent_a")
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        mock_client.post = AsyncMock(return_value=MagicMock())
        self._patch(
            monkeypatch, mock_client, {"agent_a": "rs_a"}, raise_for_status_side_effect=RuntimeError("HTTP 500")
        )

        with pytest.raises(RuntimeError, match="HTTP 500"):
            await self._collect(_run_verification_payloads([row]))

    async def test_no_debug_output_when_debug_is_disabled(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        row = self._make_row("agent_a")
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        mock_client.post = AsyncMock(return_value=MagicMock())
        self._patch(
            monkeypatch, mock_client, {"agent_a": "rs_a"}, raise_for_status_side_effect=RuntimeError("HTTP 500")
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.is_global_aiohttp_client_request_debug_enabled", lambda: False
        )

        with pytest.raises(RuntimeError):
            await self._collect(_run_verification_payloads([row]))

        assert capsys.readouterr().out == ""

    async def test_failed_response_prints_debug_info_when_debug_enabled(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
    ) -> None:
        row = self._make_row("agent_a")
        mock_response = MagicMock()
        mock_response.status = 500
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        mock_client.post = AsyncMock(return_value=mock_response)
        self._patch(
            monkeypatch, mock_client, {"agent_a": "rs_a"}, raise_for_status_side_effect=RuntimeError("HTTP 500")
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.is_global_aiohttp_client_request_debug_enabled", lambda: True
        )

        with pytest.raises(RuntimeError):
            await self._collect(_run_verification_payloads([row]))

        out = capsys.readouterr().out
        assert "/verify failed" in out
        assert "status=500" in out

    async def test_none_semaphore_imposes_no_concurrency_restriction(self, monkeypatch: pytest.MonkeyPatch) -> None:
        N = 5
        rows = [self._make_row("agent_a", i) for i in range(N)]
        mock_client = MagicMock()
        mock_client.global_config_dict = {}

        active = [0]
        max_active = [0]

        async def slow_post(**_kwargs):
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0)
            active[0] -= 1
            return MagicMock()

        mock_client.post = slow_post
        self._patch(monkeypatch, mock_client, {"agent_a": "rs_a"}, response_json={})

        results = await self._collect(_run_verification_payloads(rows, semaphore=None))

        assert len(results) == N
        assert max_active[0] == N  # no semaphore → all run concurrently

    async def test_routing_mapping_receives_global_config_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        sentinel_config = {"the": "global_config"}
        mock_client.global_config_dict = sentinel_config
        mock_client.post = AsyncMock(return_value=MagicMock())
        monkeypatch.setattr("nemo_gym.rollout_reverification.setup_server_client", lambda: mock_client)
        received: list = []
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping",
            lambda cfg: received.append(cfg) or {"agent_a": "rs_a"},
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification.raise_for_status", AsyncMock())
        monkeypatch.setattr("nemo_gym.rollout_reverification.get_response_json", AsyncMock(return_value={}))

        await self._collect(_run_verification_payloads([self._make_row("agent_a")]))

        assert received[0] is sentinel_config

    async def test_semaphore_limits_concurrent_requests(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Semaphore(2) must prevent more than 2 /verify calls from running concurrently."""
        N = 6
        rows = [self._make_row("agent_a", i) for i in range(N)]
        mock_client = MagicMock()
        mock_client.global_config_dict = {}

        active = [0]
        max_active = [0]

        async def slow_post(**_kwargs):
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
            await asyncio.sleep(0)  # yield so other coroutines can attempt semaphore entry
            active[0] -= 1
            return MagicMock()

        mock_client.post = slow_post
        self._patch(monkeypatch, mock_client, {"agent_a": "rs_a"}, response_json={})

        results = await self._collect(_run_verification_payloads(rows, semaphore=Semaphore(2)))

        assert len(results) == N
        assert max_active[0] <= 2

    async def test_semaphore_fully_released_after_all_tasks_complete(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The semaphore's internal counter must be fully restored once every payload finishes."""
        rows = [self._make_row("agent_a", i) for i in range(3)]
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        mock_client.post = AsyncMock(return_value=MagicMock())
        self._patch(monkeypatch, mock_client, {"agent_a": "rs_a"}, response_json={})

        sem = Semaphore(3)
        await self._collect(_run_verification_payloads(rows, semaphore=sem))

        assert sem._value == 3


class TestPrepareOutputFpaths:
    """Tests for _prepare_output_fpaths: prefixing, parent-dir creation, and resume-aware cleanup."""

    def test_creates_the_parent_directory_not_the_file_path(self, tmp_path: Path) -> None:
        """Regression: the output *file* must remain a writable file, and only its parent dir is created.

        A prior version called .mkdir() on the file path itself, turning output.jsonl into a directory
        and crashing run_from_config on its default (non-resume) path.
        """
        nested = tmp_path / "does" / "not" / "exist" / "out.jsonl"
        paths = _prepare_output_fpaths("", str(nested), resume_from_cache=False, overwrite=False)

        assert paths.output.parent.is_dir()
        assert not paths.output.is_dir()
        # the returned path must be openable as a file
        paths.output.open("ab").close()
        assert paths.output.is_file()

    def test_applies_name_prefix_to_output_and_failures(self, tmp_path: Path) -> None:
        paths = _prepare_output_fpaths(
            "unsafe_", str(tmp_path / "out.jsonl"), resume_from_cache=False, overwrite=False
        )
        assert paths.output.name == "unsafe_out.jsonl"
        assert paths.failures.name == "unsafe_out_failures.jsonl"

    def test_empty_prefix_leaves_names_unchanged(self, tmp_path: Path) -> None:
        paths = _prepare_output_fpaths("", str(tmp_path / "out.jsonl"), resume_from_cache=False, overwrite=False)
        assert paths.output.name == "out.jsonl"
        assert paths.failures.name == "out_failures.jsonl"

    def test_raises_when_existing_and_not_overwrite_and_not_resuming(self, tmp_path: Path) -> None:
        """The safety guard: a fresh run must refuse to clobber an existing output file by default."""
        out = tmp_path / "out.jsonl"
        out.write_text("precious prior rollouts")
        with pytest.raises(ConfigError, match="already exists"):
            _prepare_output_fpaths("", str(out), resume_from_cache=False, overwrite=False)
        # the file is left untouched
        assert out.read_text() == "precious prior rollouts"

    def test_raises_when_only_the_failures_sidecar_exists(self, tmp_path: Path) -> None:
        """The guard also protects the failures sidecar, not just the main output file."""
        failures = tmp_path / "out_failures.jsonl"
        failures.write_text("prior failures")
        with pytest.raises(ConfigError, match="already exists"):
            _prepare_output_fpaths("", str(tmp_path / "out.jsonl"), resume_from_cache=False, overwrite=False)

    def test_deletes_existing_files_when_overwrite_and_not_resuming(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        out = tmp_path / "out.jsonl"
        failures = tmp_path / "out_failures.jsonl"
        out.write_text("stale")
        failures.write_text("stale")

        paths = _prepare_output_fpaths("", str(out), resume_from_cache=False, overwrite=True)

        assert not paths.output.exists()
        assert not paths.failures.exists()
        assert "Deleted existing output file" in capsys.readouterr().out

    def test_no_existing_files_is_fine_without_overwrite(self, tmp_path: Path) -> None:
        """A first run (no prior files) proceeds without needing overwrite."""
        paths = _prepare_output_fpaths("", str(tmp_path / "out.jsonl"), resume_from_cache=False, overwrite=False)
        assert not paths.output.exists()

    def test_keeps_existing_files_when_resuming_regardless_of_overwrite(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        failures = tmp_path / "out_failures.jsonl"
        out.write_text("keep me")
        failures.write_text("keep me too")

        # resume reuses the file; overwrite is ignored and nothing is deleted or raised
        paths = _prepare_output_fpaths("", str(out), resume_from_cache=True, overwrite=False)

        assert paths.output.read_text() == "keep me"
        assert paths.failures.read_text() == "keep me too"


class TestLoadReverifiedResults:
    """Tests for _load_reverified_results: read the main jsonl once, sorted, with minimal routing rows."""

    def test_reads_sorted_results_and_minimal_routing_rows(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        # written out of (task, rollout) order; the helper must sort
        out.write_bytes(
            orjson.dumps(
                {AGENT_REF_KEY_NAME: {"name": "a"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 0.0}
            )
            + b"\n"
            + orjson.dumps(
                {AGENT_REF_KEY_NAME: {"name": "b"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0, "reward": 1.0}
            )
            + b"\n"
        )

        results, rows = _load_reverified_results(out)

        # results sorted by (task, rollout), full payload preserved
        assert [r[TASK_INDEX_KEY_NAME] for r in results] == [0, 1]
        assert results[0]["reward"] == 1.0
        # rows carry only AGENT_REF, aligned with results
        assert rows == [{AGENT_REF_KEY_NAME: {"name": "b"}}, {AGENT_REF_KEY_NAME: {"name": "a"}}]

    def test_empty_and_blank_lines(self, tmp_path: Path) -> None:
        out = tmp_path / "out.jsonl"
        out.write_bytes(b"")
        assert _load_reverified_results(out) == ([], [])

        out.write_bytes(
            orjson.dumps({AGENT_REF_KEY_NAME: {"name": "a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0})
            + b"\n\n"
        )
        results, rows = _load_reverified_results(out)
        assert len(results) == 1
        assert len(rows) == 1


class TestCallAggregateMetrics:
    """Tests for _call_aggregate_metrics.

    Each agent's results must reach only its own resource server (no cross-contamination),
    and the function must correctly use rows for routing, results as the payload,
    and output_fpath to determine the written file path.
    """

    _EMPTY_AGG = {"agent_metrics": {"reward_mean": 0.5}, "key_metrics": {"score": 0.5}, "group_level_metrics": []}

    def _patch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        agent_to_rs: dict,
        post_side_effect=None,
    ) -> MagicMock:
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        mock_client.post = AsyncMock(return_value=MagicMock(), side_effect=post_side_effect)
        monkeypatch.setattr("nemo_gym.rollout_reverification.setup_server_client", lambda: mock_client)
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping",
            lambda _: agent_to_rs,
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification.raise_for_status", AsyncMock())
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.get_response_json",
            AsyncMock(return_value=self._EMPTY_AGG),
        )
        return mock_client

    async def test_each_agent_sent_to_its_own_resource_server_with_only_its_results(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """With two agents mapped to two different RS, each RS must receive only
        the verify responses belonging to its agent — not a mix of both."""
        rows = [
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {"name": "agent_b"}, TASK_INDEX_KEY_NAME: 2, ROLLOUT_INDEX_KEY_NAME: 0},
        ]
        results = [
            {"reward": 1.0, TASK_INDEX_KEY_NAME: 0},
            {"reward": 0.5, TASK_INDEX_KEY_NAME: 1},
            {"reward": 0.0, TASK_INDEX_KEY_NAME: 2},
        ]

        posted: list[tuple[str, list]] = []

        async def capture_post(server_name, url_path, json):  # noqa: ARG001
            posted.append((server_name, json.verify_responses))
            return MagicMock()

        mock_client = self._patch(monkeypatch, {"agent_a": "rs_a", "agent_b": "rs_b"})
        mock_client.post = capture_post

        await _call_aggregate_metrics(results, rows, tmp_path / "rollouts.jsonl")

        assert len(posted) == 2
        by_server = {server: payload for server, payload in posted}

        # rs_a gets the two agent_a results
        assert len(by_server["rs_a"]) == 2
        assert {r[TASK_INDEX_KEY_NAME] for r in by_server["rs_a"]} == {0, 1}

        # rs_b gets the one agent_b result
        assert len(by_server["rs_b"]) == 1
        assert by_server["rs_b"][0][TASK_INDEX_KEY_NAME] == 2

    async def test_rows_drives_routing_results_drives_payload_and_output_fpath_drives_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Verify that:
        - agent routing is read from *rows* (results intentionally lack AGENT_REF_KEY_NAME)
        - *results* are stripped before sending: response body and responses_create_params removed,
          but response.usage preserved
        - the written file path is derived from output_fpath, not hardcoded
        """
        rows = [{AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}]
        results = [
            {
                "reward": 0.8,
                TASK_INDEX_KEY_NAME: 0,
                # no AGENT_REF_KEY_NAME — agent routing must come from rows, not results
                "response": {
                    "output": "a very long model response",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                },
                "responses_create_params": {"input": "large prompt content", "model": "llm"},
            }
        ]

        posted_payloads: list = []

        async def capture_post(**kwargs):
            posted_payloads.append(kwargs["json"].verify_responses)
            return MagicMock()

        mock_client = self._patch(monkeypatch, {"agent_a": "rs_a"})
        mock_client.post = capture_post

        output_fpath = tmp_path / "my_run_rollouts.jsonl"
        returned_path = await _call_aggregate_metrics(results, rows, output_fpath)

        # routing came from rows (not results) — one call was made
        assert len(posted_payloads) == 1
        sent = posted_payloads[0][0]

        # response body stripped, responses_create_params stripped
        assert "responses_create_params" not in sent
        assert sent.get("response") == {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}

        # other fields preserved
        assert sent["reward"] == 0.8

        # output file named after output_fpath, written in the same directory
        assert returned_path is not None
        assert returned_path == tmp_path / "my_run_rollouts_aggregate_metrics.json"
        assert returned_path.exists()

    async def test_returns_none_when_no_results(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """No results → no aggregate file written and None returned."""
        self._patch(monkeypatch, {"agent_a": "rs_a"})
        assert await _call_aggregate_metrics([], [], tmp_path / "out.jsonl") is None

    async def test_rows_without_an_agent_name_are_skipped(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A row whose AGENT_REF has no name contributes nothing (it can't be routed to an RS)."""
        posted: list[str] = []

        async def capture_post(server_name, url_path, json):  # noqa: ARG001
            posted.append(server_name)
            return MagicMock()

        mock_client = self._patch(monkeypatch, {"agent_a": "rs_a"})
        mock_client.post = capture_post

        rows = [
            {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0},
            {AGENT_REF_KEY_NAME: {}, TASK_INDEX_KEY_NAME: 1, ROLLOUT_INDEX_KEY_NAME: 0},  # no name -> skipped
        ]
        results = [{"reward": 1.0, TASK_INDEX_KEY_NAME: 0}, {"reward": 0.0, TASK_INDEX_KEY_NAME: 1}]

        await _call_aggregate_metrics(results, rows, tmp_path / "out.jsonl")

        # only agent_a's RS is contacted; the nameless row never reaches aggregation
        assert posted == ["rs_a"]


class TestCheckReverifyMode:
    """Tests for _check_reverify_mode — queries GET /reverify_mode per unique RS."""

    def _mock_client(self, monkeypatch: pytest.MonkeyPatch, responses_by_rs: dict[str, ReverifyMode]) -> MagicMock:
        """Mock client whose .get embeds the RS name in the response so get_response_json can look it up.

        This makes tests order-independent: _check_reverify_mode iterates a set() whose order
        is non-deterministic, so positional side_effect lists would be fragile.
        """
        mock_client = MagicMock()

        async def fake_get(**kwargs: object) -> str:
            return kwargs["server_name"]  # RS name becomes the "response" object

        mock_client.get = fake_get
        monkeypatch.setattr("nemo_gym.rollout_reverification.raise_for_status", AsyncMock())

        async def get_response_by_rs(rs_name: str) -> ReverifyMode:
            return responses_by_rs[rs_name]

        monkeypatch.setattr("nemo_gym.rollout_reverification.get_response_json", get_response_by_rs)
        return mock_client

    async def test_returns_empty_when_all_stateless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = self._mock_client(monkeypatch, {"rs_a": ReverifyMode.STATELESS, "rs_b": ReverifyMode.STATELESS})

        result = await _check_reverify_mode(mock_client, {"agent_a": "rs_a", "agent_b": "rs_b"})

        assert result == []

    async def test_returns_unsupported_rs_names(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = self._mock_client(
            monkeypatch,
            {"rs_a": ReverifyMode.STATELESS, "rs_b": ReverifyMode.UNSUPPORTED, "rs_c": ReverifyMode.UNSUPPORTED},
        )

        result = await _check_reverify_mode(mock_client, {"agent_a": "rs_a", "agent_b": "rs_b", "agent_c": "rs_c"})

        assert result == ["rs_b", "rs_c"]

    async def test_queries_each_unique_rs_exactly_once(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two agents sharing the same RS must result in only one GET /reverify_mode call."""
        queried: list[str] = []

        async def capturing_get(**kwargs: object) -> MagicMock:
            queried.append(kwargs["server_name"])
            return MagicMock()

        mock_client = MagicMock()
        mock_client.get = capturing_get
        monkeypatch.setattr("nemo_gym.rollout_reverification.raise_for_status", AsyncMock())
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.get_response_json",
            AsyncMock(return_value=ReverifyMode.STATELESS),
        )

        await _check_reverify_mode(mock_client, {"agent_a": "rs_shared", "agent_b": "rs_shared"})

        assert queried == ["rs_shared"]


class TestGuardReverifyMode:
    """Tests for _guard_reverify_mode — raises or returns a warning based on config.force.

    The refactored guard checks *every* resource server declared in the config (not just the
    subset referenced by the current payload batch), reading the force flag from the config.
    """

    def _make_config(self, tmp_path: Path, *, force: bool = False) -> RolloutReverificationConfig:
        return RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath=str(tmp_path / "in.jsonl"),
            rollouts_jsonl_fpath=str(tmp_path / "r.jsonl"),
            output_jsonl_fpath=str(tmp_path / "out.jsonl"),
            force=force,
        )

    def _patch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        unsupported_rs: list[str],
        agent_to_rs: dict[str, str] | None = None,
    ) -> None:
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        monkeypatch.setattr("nemo_gym.rollout_reverification.setup_server_client", lambda: mock_client)
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping",
            lambda _: agent_to_rs or {"agent_a": "rs_a"},
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._check_reverify_mode",
            AsyncMock(return_value=unsupported_rs),
        )

    async def test_returns_none_when_all_stateless(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self._patch(monkeypatch, unsupported_rs=[])
        assert await _guard_reverify_mode(self._make_config(tmp_path)) is None

    async def test_raises_config_error_when_unsupported_and_not_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch, unsupported_rs=["rs_a"])
        with pytest.raises(ConfigError, match="rs_a"):
            await _guard_reverify_mode(self._make_config(tmp_path, force=False))

    async def test_returns_warning_string_when_unsupported_and_force(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch(monkeypatch, unsupported_rs=["rs_a"])
        result = await _guard_reverify_mode(self._make_config(tmp_path, force=True))
        assert result is not None
        assert "WARNING" in result
        assert "rs_a" in result
        assert "unsafe_" in result

    async def test_every_rs_in_config_is_checked_not_just_a_payload_subset(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """All RS from the config mapping are passed to _check_reverify_mode, regardless of payloads."""
        captured: list[dict] = []
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        monkeypatch.setattr("nemo_gym.rollout_reverification.setup_server_client", lambda: mock_client)
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping",
            lambda _: {"agent_a": "rs_a", "agent_b": "rs_b"},
        )

        async def capture_check(_client: object, agent_to_rs: dict) -> list:
            captured.append(dict(agent_to_rs))
            return []

        monkeypatch.setattr("nemo_gym.rollout_reverification._check_reverify_mode", capture_check)

        await _guard_reverify_mode(self._make_config(tmp_path))

        assert captured == [{"agent_a": "rs_a", "agent_b": "rs_b"}]


class TestRolloutReverificationRunFromConfig:
    """Tests for RolloutReverificationHelper.run_from_config.

    All module-level helpers are already unit-tested; these tests cover the orchestration
    logic that lives exclusively inside run_from_config:
      - three-way file routing (success / failure_class / no_persist)
      - metadata stamping from row onto result
      - post-loop sort before aggregate metrics
      - disable_aggregation flag
      - multi-agent multi-RS end-to-end routing
    """

    # ------------------------------------------------------------------ helpers

    def _make_config(
        self,
        tmp_path: Path,
        *,
        disable_aggregation: bool = False,
        num_samples_in_parallel: int | None = None,
    ) -> RolloutReverificationConfig:
        return RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath=str(tmp_path / "inputs.jsonl"),
            rollouts_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            disable_aggregation=disable_aggregation,
            num_samples_in_parallel=num_samples_in_parallel,
        )

    def _make_row(self, agent: str, task: int, rollout: int = 0, *, skills: bool = False) -> dict:
        row = {
            AGENT_REF_KEY_NAME: {"name": agent},
            TASK_INDEX_KEY_NAME: task,
            ROLLOUT_INDEX_KEY_NAME: rollout,
            "verifier_metadata": {"answer": task},
            "responses_create_params": {"input": f"prompt for task {task}"},
        }
        if skills:
            row[SKILLS_REF_KEY_NAME] = ["skill_a"]
        return row

    def _patch_common(self, monkeypatch: pytest.MonkeyPatch, pairs: list[tuple[dict, dict]]) -> None:
        """Patch the delegates that run_from_config calls."""
        payloads = [row for row, _ in pairs]
        monkeypatch.setattr("nemo_gym.rollout_reverification._prepare_payloads", lambda *_a, **_kw: payloads)

        async def fake_future(row: dict, result: dict) -> tuple[dict, dict]:
            return row, result

        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._run_verification_payloads",
            lambda *_args, **_kwargs: [fake_future(row, result) for row, result in pairs],
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification._call_aggregate_metrics", AsyncMock(return_value=None))
        monkeypatch.setattr("nemo_gym.rollout_reverification._guard_reverify_mode", AsyncMock(return_value=None))
        monkeypatch.setattr("nemo_gym.rollout_reverification.get_wandb_run", lambda: None)
        # run_from_config resolves the input paths for real; the files don't exist in these tests, so stub it out.
        monkeypatch.setattr("nemo_gym.rollout_reverification.resolve_input_path", lambda p: Path(p))

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]

    # ------------------------------------------------------------------ tests

    async def test_results_routed_to_correct_output_files_and_metadata_stamped_on_results(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Success, failure_class, and no_persist results each land in the right file.

        Also checks that TASK_INDEX, ROLLOUT_INDEX, AGENT_REF are stamped from the row onto
        every result, and that SKILLS_REF is only stamped when present in the row.
        The output file must contain exactly the same fields as the verify response plus the
        stamped metadata — no extra or missing keys.
        """
        success_row = self._make_row("agent_a", task=0, skills=True)
        failure_row = self._make_row("agent_a", task=1)
        no_persist_row = self._make_row("agent_a", task=2)

        success_result = {"reward": 1.0}
        failure_result = {"reward": 0.0, NG_FAILURE_CLASS_KEY: "timeout"}
        no_persist_result = {"reward": 0.0, NG_NO_PERSIST_KEY: True}

        pairs = [
            (success_row, success_result),
            (failure_row, failure_result),
            (no_persist_row, no_persist_result),
        ]
        self._patch_common(monkeypatch, pairs)
        config = self._make_config(tmp_path, disable_aggregation=True)

        returned = await RolloutReverificationHelper().run_from_config(config)

        output_rows = self._read_jsonl(tmp_path / "output.jsonl")
        failure_rows = self._read_jsonl(tmp_path / "output_failures.jsonl")

        # file row counts: 1 success, 1 failure, 0 no_persist
        assert len(output_rows) == 1
        assert len(failure_rows) == 1

        # success row: exact field set = verify response fields + 3 stamped metadata fields + SKILLS_REF
        stamped = output_rows[0]
        assert stamped[TASK_INDEX_KEY_NAME] == 0
        assert stamped[ROLLOUT_INDEX_KEY_NAME] == 0
        assert stamped[AGENT_REF_KEY_NAME] == {"name": "agent_a"}
        assert stamped[SKILLS_REF_KEY_NAME] == ["skill_a"]
        assert stamped["reward"] == 1.0
        assert set(stamped.keys()) == {
            "reward",
            TASK_INDEX_KEY_NAME,
            ROLLOUT_INDEX_KEY_NAME,
            AGENT_REF_KEY_NAME,
            SKILLS_REF_KEY_NAME,
        }

        # failure row: exact field set = verify response fields + 3 stamped metadata fields (no SKILLS_REF)
        failed = failure_rows[0]
        assert failed[TASK_INDEX_KEY_NAME] == 1
        assert failed[NG_FAILURE_CLASS_KEY] == "timeout"
        assert SKILLS_REF_KEY_NAME not in failed
        assert set(failed.keys()) == {
            "reward",
            NG_FAILURE_CLASS_KEY,
            TASK_INDEX_KEY_NAME,
            ROLLOUT_INDEX_KEY_NAME,
            AGENT_REF_KEY_NAME,
        }

        # run_from_config returns the persisted main-jsonl rows only: exactly the 1 success —
        # NOT the failure (sidecar) or the no_persist row (written nowhere).
        assert len(returned) == 1
        assert returned[0][TASK_INDEX_KEY_NAME] == 0
        assert returned[0]["reward"] == 1.0

    async def test_results_sorted_by_task_and_rollout_index_before_aggregate_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Futures complete in reverse order; run_from_config must sort the results it passes to
        _call_aggregate_metrics into deterministic (task, rollout) order."""
        # futures arrive: task 2, then 1, then 0
        pairs = [
            (self._make_row("agent_a", task=2), {"reward": 0.2}),
            (self._make_row("agent_a", task=1), {"reward": 0.1}),
            (self._make_row("agent_a", task=0), {"reward": 0.0}),
        ]
        self._patch_common(monkeypatch, pairs)
        agg_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("nemo_gym.rollout_reverification._call_aggregate_metrics", agg_mock)
        config = self._make_config(tmp_path)

        returned = await RolloutReverificationHelper().run_from_config(config)

        # all 3 successes are returned, sorted task 0 → 1 → 2 (despite arriving 2 → 1 → 0)
        assert len(returned) == 3
        assert [r[TASK_INDEX_KEY_NAME] for r in returned] == [0, 1, 2]

        # aggregate metrics are computed over the main jsonl, sorted by (task, rollout);
        # the routing rows are aligned with them
        agg_results, agg_rows = agg_mock.call_args.args[0], agg_mock.call_args.args[1]
        assert [r[TASK_INDEX_KEY_NAME] for r in agg_results] == [0, 1, 2]
        assert len(agg_rows) == 3
        assert all(r[AGENT_REF_KEY_NAME] == {"name": "agent_a"} for r in agg_rows)

        # output file on disk is in arrival order (written as futures complete, before the sort)
        output_rows = self._read_jsonl(tmp_path / "output.jsonl")
        assert len(output_rows) == 3

    async def test_num_samples_in_parallel_positive_creates_semaphore_with_that_value(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A positive num_samples_in_parallel must create a Semaphore with that exact value."""
        pairs = [(self._make_row("agent_a", task=0), {"reward": 1.0})]
        captured_semaphore: list = []

        def capturing_run_verification(payloads, semaphore):
            captured_semaphore.append(semaphore)
            return []

        self._patch_common(monkeypatch, pairs)
        monkeypatch.setattr("nemo_gym.rollout_reverification._run_verification_payloads", capturing_run_verification)
        config = self._make_config(tmp_path, disable_aggregation=True, num_samples_in_parallel=3)

        await RolloutReverificationHelper().run_from_config(config)

        assert len(captured_semaphore) == 1
        assert isinstance(captured_semaphore[0], Semaphore)
        assert captured_semaphore[0]._value == 3

    async def test_num_samples_in_parallel_none_uses_unbounded_nullcontext(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """num_samples_in_parallel=None (default) must run unbounded — no Semaphore."""
        pairs = [(self._make_row("agent_a", task=0), {"reward": 1.0})]
        captured_semaphore: list = []

        def capturing_run_verification(payloads, semaphore):
            captured_semaphore.append(semaphore)
            return []

        self._patch_common(monkeypatch, pairs)
        monkeypatch.setattr("nemo_gym.rollout_reverification._run_verification_payloads", capturing_run_verification)
        config = self._make_config(tmp_path, disable_aggregation=True, num_samples_in_parallel=None)

        await RolloutReverificationHelper().run_from_config(config)

        assert len(captured_semaphore) == 1
        assert not isinstance(captured_semaphore[0], Semaphore)

    async def test_disable_aggregation_skips_aggregate_metrics_call(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """disable_aggregation=True must suppress _call_aggregate_metrics entirely.
        The output file must still be written normally."""
        pairs = [(self._make_row("agent_a", task=0), {"reward": 1.0})]
        self._patch_common(monkeypatch, pairs)
        agg_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("nemo_gym.rollout_reverification._call_aggregate_metrics", agg_mock)
        config = self._make_config(tmp_path, disable_aggregation=True)

        returned = await RolloutReverificationHelper().run_from_config(config)

        agg_mock.assert_not_awaited()
        assert len(self._read_jsonl(tmp_path / "output.jsonl")) == 1
        # results are still returned even though aggregation was skipped
        assert len(returned) == 1
        assert returned[0][TASK_INDEX_KEY_NAME] == 0

    async def test_multi_agent_multi_rs_routing_end_to_end(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Two agents backed by two different resource servers.
        Each agent's payloads must reach only its own RS on /verify, all four results
        must appear in the output file with the correct fields, and _call_aggregate_metrics
        must receive results for both agents."""
        pairs = [
            (self._make_row("agent_a", task=0), {"reward": 1.0}),
            (self._make_row("agent_a", task=1), {"reward": 0.8}),
            (self._make_row("agent_b", task=2), {"reward": 0.5}),
            (self._make_row("agent_b", task=3), {"reward": 0.3}),
        ]
        payloads = [row for row, _ in pairs]
        monkeypatch.setattr("nemo_gym.rollout_reverification._prepare_payloads", lambda *_a, **_kw: payloads)

        # Wire _run_verification_payloads through its real impl so routing is exercised.
        # Patch the hooks it relies on instead.
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        verify_calls: list[tuple[str, dict]] = []

        async def capturing_post(server_name, url_path, json):  # noqa: ARG001
            verify_calls.append((server_name, json))
            return MagicMock()

        mock_client.post = capturing_post
        monkeypatch.setattr("nemo_gym.rollout_reverification.setup_server_client", lambda: mock_client)
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping",
            lambda _: {"agent_a": "rs_a", "agent_b": "rs_b"},
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification.raise_for_status", AsyncMock())
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.get_response_json",
            AsyncMock(side_effect=[{"reward": 1.0}, {"reward": 0.8}, {"reward": 0.5}, {"reward": 0.3}]),
        )

        agg_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("nemo_gym.rollout_reverification._call_aggregate_metrics", agg_mock)
        monkeypatch.setattr("nemo_gym.rollout_reverification._guard_reverify_mode", AsyncMock(return_value=None))
        monkeypatch.setattr("nemo_gym.rollout_reverification.get_wandb_run", lambda: None)
        monkeypatch.setattr("nemo_gym.rollout_reverification.resolve_input_path", lambda p: Path(p))
        config = self._make_config(tmp_path)

        returned = await RolloutReverificationHelper().run_from_config(config)

        # /verify called 4 times: 2 to rs_a, 2 to rs_b
        assert len(verify_calls) == 4
        rs_a_calls = [(s, j) for s, j in verify_calls if s == "rs_a"]
        rs_b_calls = [(s, j) for s, j in verify_calls if s == "rs_b"]
        assert len(rs_a_calls) == 2
        assert len(rs_b_calls) == 2

        # each RS only received payloads belonging to its own agent
        for _, payload in rs_a_calls:
            assert payload[AGENT_REF_KEY_NAME]["name"] == "agent_a"
        for _, payload in rs_b_calls:
            assert payload[AGENT_REF_KEY_NAME]["name"] == "agent_b"

        # output file: all 4 results, exact field set matches verify response + stamped metadata
        output_rows = self._read_jsonl(tmp_path / "output.jsonl")
        assert len(output_rows) == 4
        expected_keys = {"reward", TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, AGENT_REF_KEY_NAME}
        for row in output_rows:
            assert set(row.keys()) == expected_keys

        # both agents present in what aggregate metrics receives
        agg_results = agg_mock.call_args.args[0]
        agent_names_in_agg = {r[AGENT_REF_KEY_NAME]["name"] for r in agg_results}
        assert agent_names_in_agg == {"agent_a", "agent_b"}

        # all 4 successes are returned, both agents represented
        assert len(returned) == 4
        assert {r[AGENT_REF_KEY_NAME]["name"] for r in returned} == {"agent_a", "agent_b"}
        assert [r[TASK_INDEX_KEY_NAME] for r in returned] == [0, 1, 2, 3]


class TestRunFromConfigForceFlag:
    """Tests for the --force / unsafe_ prefix integration inside run_from_config."""

    def _make_config(self, tmp_path: Path, *, force: bool = False) -> RolloutReverificationConfig:
        return RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath=str(tmp_path / "inputs.jsonl"),
            rollouts_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            disable_aggregation=True,
            force=force,
        )

    def _patch_common(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *,
        force_warning: str | None,
    ) -> None:
        row = {AGENT_REF_KEY_NAME: {"name": "agent_a"}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}
        result = {"reward": 1.0}

        async def fake_future() -> tuple[dict, dict]:
            return row, result

        monkeypatch.setattr("nemo_gym.rollout_reverification._prepare_payloads", lambda *_a, **_kw: [row])
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._guard_reverify_mode",
            AsyncMock(return_value=force_warning),
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._run_verification_payloads",
            lambda *_a, **_kw: [fake_future()],
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification._call_aggregate_metrics", AsyncMock(return_value=None))
        monkeypatch.setattr("nemo_gym.rollout_reverification.get_wandb_run", lambda: None)
        monkeypatch.setattr("nemo_gym.rollout_reverification.resolve_input_path", lambda p: Path(p))

    async def test_no_unsafe_prefix_when_all_rs_stateless(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        self._patch_common(monkeypatch, force_warning=None)
        await RolloutReverificationHelper().run_from_config(self._make_config(tmp_path))
        assert (tmp_path / "output.jsonl").exists()
        assert not (tmp_path / "unsafe_output.jsonl").exists()

    async def test_unsafe_prefix_applied_to_output_when_force_warning(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        warning = (
            "WARNING: resource server(s) ['rs_a'] have reverify_mode=UNSUPPORTED. Output is prefixed with 'unsafe_'."
        )
        self._patch_common(monkeypatch, force_warning=warning)
        await RolloutReverificationHelper().run_from_config(self._make_config(tmp_path, force=True))

        assert (tmp_path / "unsafe_output.jsonl").exists()
        assert not (tmp_path / "output.jsonl").exists()

    async def test_warning_printed_before_and_after_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        warning = (
            "WARNING: resource server(s) ['rs_a'] have reverify_mode=UNSUPPORTED. Output is prefixed with 'unsafe_'."
        )
        self._patch_common(monkeypatch, force_warning=warning)
        await RolloutReverificationHelper().run_from_config(self._make_config(tmp_path, force=True))

        out = capsys.readouterr().out
        assert out.count(warning) == 2


class TestRunFromConfigResumeFromCache:
    """End-to-end resume behavior for run_from_config.

    _prepare_output_fpaths and _prepare_payloads run for real here; only the input file yield,
    path resolution, the verify dispatch, guard, and aggregate call are stubbed. This exercises
    the full resume path (preserve output, skip cached keys, append only the missing ones).
    """

    def _make_config(
        self, tmp_path: Path, *, resume_from_cache: bool, disable_aggregation: bool = True, overwrite: bool = False
    ) -> RolloutReverificationConfig:
        return RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath=str(tmp_path / "inputs.jsonl"),
            rollouts_jsonl_fpath=str(tmp_path / "rollouts.jsonl"),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            disable_aggregation=disable_aggregation,
            resume_from_cache=resume_from_cache,
            overwrite=overwrite,
        )

    def _row(self, agent: str, task: int, rollout: int = 0) -> dict:
        return {AGENT_REF_KEY_NAME: {"name": agent}, TASK_INDEX_KEY_NAME: task, ROLLOUT_INDEX_KEY_NAME: rollout}

    def _patch(self, monkeypatch: pytest.MonkeyPatch, pairs: list[InputRolloutPair], dispatched: list) -> None:
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._yield_inputs_and_rollouts_paired", lambda *_a, **_kw: iter(pairs)
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification.resolve_input_path", lambda p: Path(p))

        def fake_run(payloads, semaphore=None):
            dispatched.extend(payloads)

            async def fut(p: dict) -> tuple[dict, dict]:
                return p, {"reward": 0.5}

            return [fut(p) for p in payloads]

        monkeypatch.setattr("nemo_gym.rollout_reverification._run_verification_payloads", fake_run)
        monkeypatch.setattr("nemo_gym.rollout_reverification._guard_reverify_mode", AsyncMock(return_value=None))
        monkeypatch.setattr("nemo_gym.rollout_reverification._call_aggregate_metrics", AsyncMock(return_value=None))
        monkeypatch.setattr("nemo_gym.rollout_reverification.get_wandb_run", lambda: None)

    def _read_jsonl(self, path: Path) -> list[dict]:
        return [orjson.loads(line) for line in path.read_bytes().splitlines() if line.strip()]

    async def test_resume_preserves_prior_output_and_reruns_only_missing(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "output.jsonl"
        # task 0 already completed by a prior run, with a distinctive reward
        out.write_bytes(orjson.dumps({**self._row("agent_a", 0), "reward": 1.0}) + b"\n")

        pairs = [
            InputRolloutPair(input=self._row("agent_a", 0), rollout={"response": {"o": 0}}),
            InputRolloutPair(input=self._row("agent_a", 1), rollout={"response": {"o": 1}}),
        ]
        dispatched: list = []
        self._patch(monkeypatch, pairs, dispatched)

        returned = await RolloutReverificationHelper().run_from_config(
            self._make_config(tmp_path, resume_from_cache=True)
        )

        # only the missing task (1) is re-verified
        assert [p[TASK_INDEX_KEY_NAME] for p in dispatched] == [1]

        # The cached task-0 row appears in the file EXACTLY ONCE (append mode preserves it on disk;
        # the dispatch loop only writes newly-verified rows).
        rows = self._read_jsonl(out)
        assert len(rows) == 2
        assert sorted(r[TASK_INDEX_KEY_NAME] for r in rows) == [0, 1]
        assert [r[TASK_INDEX_KEY_NAME] for r in rows].count(0) == 1
        # the pre-existing task-0 row was preserved (reward 1.0, not overwritten by a fresh 0.5)
        task0 = next(r for r in rows if r[TASK_INDEX_KEY_NAME] == 0)
        assert task0["reward"] == 1.0

        # the returned set covers cached + new (full dataset), one row per key, sorted
        assert len(returned) == 2
        assert [r[TASK_INDEX_KEY_NAME] for r in returned] == [0, 1]
        assert next(r for r in returned if r[TASK_INDEX_KEY_NAME] == 0)["reward"] == 1.0

    async def test_overwrite_without_resume_deletes_prior_output_and_reruns_everything(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "output.jsonl"
        out.write_bytes(orjson.dumps({**self._row("agent_a", 0), "reward": 1.0}) + b"\n")

        pairs = [
            InputRolloutPair(input=self._row("agent_a", 0), rollout={"response": {}}),
            InputRolloutPair(input=self._row("agent_a", 1), rollout={"response": {}}),
        ]
        dispatched: list = []
        self._patch(monkeypatch, pairs, dispatched)

        returned = await RolloutReverificationHelper().run_from_config(
            self._make_config(tmp_path, resume_from_cache=False, overwrite=True)
        )

        # both tasks re-run (stale output was cleared first)
        assert sorted(p[TASK_INDEX_KEY_NAME] for p in dispatched) == [0, 1]
        rows = self._read_jsonl(out)
        assert sorted(r[TASK_INDEX_KEY_NAME] for r in rows) == [0, 1]
        # every row is a fresh 0.5 — the old reward 1.0 was discarded with the truncated file
        assert all(r["reward"] == 0.5 for r in rows)
        # exactly the 2 fresh rows are returned (no stale row lingering)
        assert len(returned) == 2
        assert [r[TASK_INDEX_KEY_NAME] for r in returned] == [0, 1]
        assert all(r["reward"] == 0.5 for r in returned)

    async def test_fresh_run_refuses_to_clobber_existing_output_without_overwrite(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The safety guard end-to-end: a default (no resume, no overwrite) run raises on an existing file."""
        out = tmp_path / "output.jsonl"
        out.write_bytes(orjson.dumps({**self._row("agent_a", 0), "reward": 1.0}) + b"\n")

        pairs = [InputRolloutPair(input=self._row("agent_a", 0), rollout={"response": {}})]
        dispatched: list = []
        self._patch(monkeypatch, pairs, dispatched)

        with pytest.raises(ConfigError, match="already exists"):
            await RolloutReverificationHelper().run_from_config(
                self._make_config(tmp_path, resume_from_cache=False, overwrite=False)
            )

        # nothing dispatched and the prior output is untouched
        assert dispatched == []
        assert self._read_jsonl(out)[0]["reward"] == 1.0

    async def test_resume_with_everything_cached_dispatches_nothing_and_leaves_output_intact(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        out = tmp_path / "output.jsonl"
        out.write_bytes(
            orjson.dumps({**self._row("agent_a", 0), "reward": 1.0})
            + b"\n"
            + orjson.dumps({**self._row("agent_a", 1), "reward": 0.0})
            + b"\n"
        )

        pairs = [
            InputRolloutPair(input=self._row("agent_a", 0), rollout={"response": {}}),
            InputRolloutPair(input=self._row("agent_a", 1), rollout={"response": {}}),
        ]
        dispatched: list = []
        self._patch(monkeypatch, pairs, dispatched)

        returned = await RolloutReverificationHelper().run_from_config(
            self._make_config(tmp_path, resume_from_cache=True)
        )

        # nothing left to run; no division-by-zero on an empty payload set
        assert dispatched == []
        # the cached rows are left intact on disk (aggregation would read them from here)
        rows = self._read_jsonl(out)
        assert sorted(r[TASK_INDEX_KEY_NAME] for r in rows) == [0, 1]
        # even with nothing re-verified, the return still covers the full cached dataset
        assert len(returned) == 2
        assert [r[TASK_INDEX_KEY_NAME] for r in returned] == [0, 1]

    async def test_resume_aggregates_over_full_main_jsonl_not_just_new_rows(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """The core resume-metrics guarantee: aggregate metrics are computed over the full main
        jsonl (cached + newly re-verified), not just the re-verified remainder (regression guard
        for the partial-aggregation bug)."""
        out = tmp_path / "output.jsonl"
        # tasks 0 and 1 already done in a prior run
        out.write_bytes(
            orjson.dumps({**self._row("agent_a", 0), "reward": 1.0})
            + b"\n"
            + orjson.dumps({**self._row("agent_a", 1), "reward": 1.0})
            + b"\n"
        )

        pairs = [InputRolloutPair(input=self._row("agent_a", t), rollout={"response": {}}) for t in range(3)]
        dispatched: list = []
        self._patch(monkeypatch, pairs, dispatched)
        agg_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("nemo_gym.rollout_reverification._call_aggregate_metrics", agg_mock)

        await RolloutReverificationHelper().run_from_config(
            self._make_config(tmp_path, resume_from_cache=True, disable_aggregation=False)
        )

        # only task 2 was re-verified...
        assert [p[TASK_INDEX_KEY_NAME] for p in dispatched] == [2]
        # ...but aggregate metrics still see all three (cached 0,1 + fresh 2), read from the file
        agg_results, agg_rows = agg_mock.call_args.args[0], agg_mock.call_args.args[1]
        assert sorted(r[TASK_INDEX_KEY_NAME] for r in agg_results) == [0, 1, 2]
        assert len(agg_rows) == 3
        assert all(r[AGENT_REF_KEY_NAME] == {"name": "agent_a"} for r in agg_rows)

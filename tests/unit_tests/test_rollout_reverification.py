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

from nemo_gym.base_resources_server import ReverifyMode
from nemo_gym.config_types import ConfigError
from nemo_gym.global_config import AGENT_REF_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, SKILLS_REF_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.rollout_reverification import (
    NG_FAILURE_CLASS_KEY,
    NG_NO_PERSIST_KEY,
    InputRolloutPair,
    RolloutReverificationConfig,
    RolloutReverificationHelper,
    _agent_to_rs_mapping_from_agent_blocks,
    _agent_to_rs_mapping_from_resources_only_config,
    _build_agent_to_resources_server_mapping,
    _build_verify_payload,
    _call_aggregate_metrics,
    _check_reverify_mode,
    _guard_output_file,
    _guard_reverify_mode,
    _prepare_payloads_from_config,
    _rollout_verify_debug_summary,
    _run_verification_payloads,
    _setup_server_client,
    _yield_inputs_and_rollouts_paired,
)


#  (
#     _DEFAULT_MAX_ROLLOUT_ATTEMPTS,
#     RolloutAggregationConfig,
#     RolloutAggregationHelper,
#     RolloutCollectionConfig,
#     RolloutCollectionHelper,
#     _expand_input_glob,
#     _get_max_rollout_attempts,
#     _rollout_request_debug_summary,
#     loads_jsonl_line,
# )


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


class TestSetupServerClient:
    def test_returns_server_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.ServerClient.load_from_global_config",
            lambda _: mock_client,
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification.is_global_aiohttp_client_setup", lambda: True)

        result = _setup_server_client()

        assert result is mock_client

    def test_configures_aiohttp_client_when_not_yet_set_up(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        mock_client.global_config_dict = {"max_connections": 10}
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.ServerClient.load_from_global_config",
            lambda _: mock_client,
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification.is_global_aiohttp_client_setup", lambda: False)

        set_aiohttp_calls = []
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.set_global_aiohttp_client",
            lambda cfg: set_aiohttp_calls.append(cfg),
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.GlobalAIOHTTPAsyncClientConfig.model_validate",
            lambda d: d,
        )

        _setup_server_client()

        assert len(set_aiohttp_calls) == 1
        assert set_aiohttp_calls[0] == {"max_connections": 10}

    def test_skips_aiohttp_setup_when_already_configured(self, monkeypatch: pytest.MonkeyPatch) -> None:
        mock_client = MagicMock()
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.ServerClient.load_from_global_config",
            lambda _: mock_client,
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification.is_global_aiohttp_client_setup", lambda: True)

        set_aiohttp_calls = []
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.set_global_aiohttp_client",
            lambda cfg: set_aiohttp_calls.append(cfg),
        )

        _setup_server_client()

        assert set_aiohttp_calls == []

    def test_passes_head_server_config_to_loader(self, monkeypatch: pytest.MonkeyPatch) -> None:
        received = []
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification.ServerClient.load_from_global_config",
            lambda cfg: received.append(cfg) or MagicMock(),
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification.is_global_aiohttp_client_setup", lambda: True)

        sentinel = MagicMock()
        _setup_server_client(head_server_config=sentinel)

        assert received == [sentinel]


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
        monkeypatch.setattr("nemo_gym.rollout_reverification._setup_server_client", lambda: mock_client)
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
        monkeypatch.setattr("nemo_gym.rollout_reverification._setup_server_client", lambda: mock_client)
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


class TestGuardOutputFile:
    def test_raises_if_file_exists_and_overwrite_false(self, tmp_path: Path) -> None:
        fpath = tmp_path / "output.jsonl"
        fpath.write_text("stale")
        with pytest.raises(ConfigError, match="Output file already exists"):
            _guard_output_file(fpath, overwrite=False)

    def test_deletes_file_when_overwrite_true(self, tmp_path: Path) -> None:
        fpath = tmp_path / "output.jsonl"
        fpath.write_text("stale")
        _guard_output_file(fpath, overwrite=True)
        assert not fpath.exists()

    def test_no_error_when_file_absent(self, tmp_path: Path) -> None:
        _guard_output_file(tmp_path / "output.jsonl", overwrite=False)


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


class TestPreparePayloadsFromConfig:
    def _make_config(self, limit: int | None = None) -> RolloutReverificationConfig:
        return RolloutReverificationConfig(
            materialized_inputs_jsonl_fpath="/inputs.jsonl",
            rollouts_jsonl_fpath="/rollouts.jsonl",
            output_jsonl_fpath="/output.jsonl",
            limit=limit,
        )

    def _patch_yield(self, monkeypatch: pytest.MonkeyPatch, pairs: list, capture: dict | None = None) -> None:
        def fake_yield(inputs_path, rollouts_path, limit=None):
            if capture is not None:
                capture["inputs_path"] = inputs_path
                capture["rollouts_path"] = rollouts_path
                capture["limit"] = limit
            return iter(pairs)

        monkeypatch.setattr("nemo_gym.rollout_reverification._yield_inputs_and_rollouts_paired", fake_yield)

    def test_resolves_both_paths_before_pairing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        resolved = {}

        def fake_resolve(path):
            r = Path(f"/resolved{path}")
            resolved[path] = r
            return r

        capture: dict = {}
        monkeypatch.setattr("nemo_gym.rollout_reverification.resolve_input_path", fake_resolve)
        self._patch_yield(monkeypatch, [], capture)
        monkeypatch.setattr("nemo_gym.rollout_reverification._build_verify_payload", lambda _pair: {})

        _prepare_payloads_from_config(self._make_config())

        assert capture["inputs_path"] == resolved["/inputs.jsonl"]
        assert capture["rollouts_path"] == resolved["/rollouts.jsonl"]

    def test_materialized_inputs_and_rollouts_paths_are_not_swapped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.rollout_reverification.resolve_input_path", lambda p: Path(p))
        capture: dict = {}
        self._patch_yield(monkeypatch, [], capture)

        _prepare_payloads_from_config(self._make_config())

        assert str(capture["inputs_path"]) == "/inputs.jsonl"
        assert str(capture["rollouts_path"]) == "/rollouts.jsonl"

    def test_limit_is_forwarded_to_yield_pairs(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("nemo_gym.rollout_reverification.resolve_input_path", lambda p: Path(p))
        capture: dict = {}
        self._patch_yield(monkeypatch, [], capture)

        _prepare_payloads_from_config(self._make_config(limit=7))

        assert capture["limit"] == 7

    def test_returns_one_payload_per_pair(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pairs = [InputRolloutPair(input={"q": i}, rollout={"response": {"out": i}}) for i in range(3)]
        monkeypatch.setattr("nemo_gym.rollout_reverification.resolve_input_path", lambda p: Path(p))
        self._patch_yield(monkeypatch, pairs)
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_verify_payload",
            lambda pair: {"payload_for": pair.input["q"]},
        )

        payloads = _prepare_payloads_from_config(self._make_config())

        assert payloads == [{"payload_for": 0}, {"payload_for": 1}, {"payload_for": 2}]

    def test_passes_full_pair_to_build_verify_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        pair = InputRolloutPair(input={"task": "x"}, rollout={"response": {"out": "y"}})
        monkeypatch.setattr("nemo_gym.rollout_reverification.resolve_input_path", lambda p: Path(p))
        self._patch_yield(monkeypatch, [pair])
        build_calls: list = []
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_verify_payload",
            lambda p: build_calls.append(p) or {},
        )

        _prepare_payloads_from_config(self._make_config())

        assert len(build_calls) == 1
        assert build_calls[0] is pair


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
        monkeypatch.setattr("nemo_gym.rollout_reverification._setup_server_client", lambda: mock_client)
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
        """Patch the five delegates that run_from_config calls."""
        payloads = [row for row, _ in pairs]
        monkeypatch.setattr("nemo_gym.rollout_reverification._prepare_payloads_from_config", lambda *_: payloads)

        async def fake_future(row: dict, result: dict) -> tuple[dict, dict]:
            return row, result

        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._run_verification_payloads",
            lambda *_args, **_kwargs: [fake_future(row, result) for row, result in pairs],
        )
        monkeypatch.setattr("nemo_gym.rollout_reverification._call_aggregate_metrics", AsyncMock(return_value=None))
        monkeypatch.setattr("nemo_gym.rollout_reverification._guard_reverify_mode", AsyncMock(return_value=None))
        monkeypatch.setattr("nemo_gym.rollout_reverification.get_wandb_run", lambda: None)
        monkeypatch.setattr("nemo_gym.rollout_reverification._guard_output_file", lambda *_: None)

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

        # run_from_config accumulates all 3 results regardless of routing
        assert len(returned) == 3

    async def test_results_sorted_by_task_and_rollout_index_before_aggregate_metrics(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Futures complete in reverse order; run_from_config must sort before calling
        _call_aggregate_metrics and before returning — so both the returned list and
        the args passed to aggregate metrics are in deterministic (task, rollout) order."""
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

        returned: list[dict] = await RolloutReverificationHelper().run_from_config(config)  # type: ignore[assignment]

        # returned list must be sorted task 0 → 1 → 2
        assert [r[TASK_INDEX_KEY_NAME] for r in returned] == [0, 1, 2]

        # both positional args to _call_aggregate_metrics must be sorted
        agg_results, agg_rows = agg_mock.call_args.args[0], agg_mock.call_args.args[1]
        assert [r[TASK_INDEX_KEY_NAME] for r in agg_results] == [0, 1, 2]
        assert [r[TASK_INDEX_KEY_NAME] for r in agg_rows] == [0, 1, 2]

        # output file on disk must also be in arrival order (written before the sort)
        # but the in-memory returned list is sorted — these are independent guarantees
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
        The output file must still be written and the results returned normally."""
        pairs = [(self._make_row("agent_a", task=0), {"reward": 1.0})]
        self._patch_common(monkeypatch, pairs)
        agg_mock = AsyncMock(return_value=None)
        monkeypatch.setattr("nemo_gym.rollout_reverification._call_aggregate_metrics", agg_mock)
        config = self._make_config(tmp_path, disable_aggregation=True)

        returned = await RolloutReverificationHelper().run_from_config(config)

        agg_mock.assert_not_awaited()
        assert len(returned) == 1
        assert len(self._read_jsonl(tmp_path / "output.jsonl")) == 1

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
        monkeypatch.setattr("nemo_gym.rollout_reverification._prepare_payloads_from_config", lambda *_: payloads)

        # Wire _run_verification_payloads through its real impl so routing is exercised.
        # Patch the three hooks it relies on instead.
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        verify_calls: list[tuple[str, dict]] = []

        async def capturing_post(server_name, url_path, json):  # noqa: ARG001
            verify_calls.append((server_name, json))
            return MagicMock()

        mock_client.post = capturing_post
        monkeypatch.setattr("nemo_gym.rollout_reverification._setup_server_client", lambda: mock_client)
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
        monkeypatch.setattr("nemo_gym.rollout_reverification._guard_output_file", lambda *_: None)
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
        assert len(returned) == 4

    @pytest.mark.parametrize("bad_value", [0, -1])
    async def test_non_positive_num_samples_in_parallel_raises(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bad_value: int
    ) -> None:
        """A non-positive concurrency raises ConfigError instead of building an unacquirable Semaphore(0)."""
        self._patch_common(monkeypatch, pairs=[(self._make_row("rs_a", task=0), {"reward": 1.0})])
        config = self._make_config(tmp_path, num_samples_in_parallel=bad_value)
        with pytest.raises(ConfigError, match="num_samples_in_parallel must be a positive integer"):
            await RolloutReverificationHelper().run_from_config(config)


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
    """Tests for _guard_reverify_mode — raises or returns warning based on force flag."""

    def _make_payload(self, agent: str) -> dict:
        return {AGENT_REF_KEY_NAME: {"name": agent}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}

    def _patch(
        self,
        monkeypatch: pytest.MonkeyPatch,
        unsupported_rs: list[str],
        agent_to_rs: dict[str, str] | None = None,
    ) -> None:
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        monkeypatch.setattr("nemo_gym.rollout_reverification._setup_server_client", lambda: mock_client)
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping",
            lambda _: agent_to_rs or {"agent_a": "rs_a"},
        )
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._check_reverify_mode",
            AsyncMock(return_value=unsupported_rs),
        )

    async def test_returns_none_when_all_stateless(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, unsupported_rs=[])
        result = await _guard_reverify_mode([self._make_payload("agent_a")], force=False)
        assert result is None

    async def test_raises_config_error_when_unsupported_and_not_force(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, unsupported_rs=["rs_a"])
        with pytest.raises(ConfigError, match="rs_a"):
            await _guard_reverify_mode([self._make_payload("agent_a")], force=False)

    async def test_returns_warning_string_when_unsupported_and_force(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch(monkeypatch, unsupported_rs=["rs_a"])
        result = await _guard_reverify_mode([self._make_payload("agent_a")], force=True)
        assert result is not None
        assert "WARNING" in result
        assert "rs_a" in result
        assert "unsafe_" in result

    async def test_only_rs_referenced_by_payloads_are_checked(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Payloads only reference agent_a; rs_b (from agent_b) must not be queried."""
        captured: list[dict] = []
        mock_client = MagicMock()
        mock_client.global_config_dict = {}
        monkeypatch.setattr("nemo_gym.rollout_reverification._setup_server_client", lambda: mock_client)
        monkeypatch.setattr(
            "nemo_gym.rollout_reverification._build_agent_to_resources_server_mapping",
            lambda _: {"agent_a": "rs_a", "agent_b": "rs_b"},
        )

        async def capture_check(_client: object, agent_to_rs: dict) -> list:
            captured.append(dict(agent_to_rs))
            return []

        monkeypatch.setattr("nemo_gym.rollout_reverification._check_reverify_mode", capture_check)

        await _guard_reverify_mode([self._make_payload("agent_a")], force=False)

        assert captured == [{"agent_a": "rs_a"}]


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

        monkeypatch.setattr("nemo_gym.rollout_reverification._prepare_payloads_from_config", lambda *_: [row])
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
        monkeypatch.setattr("nemo_gym.rollout_reverification._guard_output_file", lambda *_: None)

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

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rollout-aware snapshot dirs.

With num_repeats > 1, all rollouts of a task share _ng_task_index, so the old
snap path (sample_{task_index}/attempt_{a}_*) made concurrent rollouts clobber
and interleave each other's snapshot files — per-reset context extraction was
unattributable (found on the mercor x4 collection run 5106853). The fix keys
the sample dir by rollout too: sample_{task_index}_r{rollout_index}, while
single-rollout runs (no rollout_index in metadata) keep the legacy naming
byte-for-byte.
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent, BrowsecompAgentRunRequest
from responses_api_agents.browsecomp_agent.tests.test_app import (
    _make_config,
    _make_model_response,
    _make_msg,
)


def _agent_with_one_answer(tmp_path):
    agent = BrowsecompAgent(
        config=_make_config(snap_dir=str(tmp_path), save_model_call_using_vllm_tokenize_endpoint=False),
        server_client=MagicMock(spec=ServerClient),
    )
    model = _make_model_response([_make_msg("Exact Answer: DONE")])
    http = MagicMock()
    http.ok = True
    http.status = 200
    http.cookies = {}
    http.read = AsyncMock(return_value=json.dumps(model).encode())
    agent.server_client.post = AsyncMock(return_value=http)
    return agent


def _mocks():
    request_mock = MagicMock()
    request_mock.cookies = {}
    response_mock = MagicMock()
    response_mock.set_cookie = MagicMock()
    return request_mock, response_mock


@pytest.mark.asyncio
async def test_snapshot_dir_includes_rollout_index(tmp_path) -> None:
    agent = _agent_with_one_answer(tmp_path)
    request_mock, response_mock = _mocks()
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "q"}],
        metadata={"task_index": "7", "attempt": "0", "rollout_index": "2"},
    )
    await agent.responses(request_mock, response_mock, body)

    assert (tmp_path / "sample_7_r2" / "attempt_0_final.jsonl").exists()
    assert (tmp_path / "sample_7_r2" / "attempt_0_trajectory.jsonl").exists()
    assert not (tmp_path / "sample_7").exists()  # nothing leaks into the legacy dir


@pytest.mark.asyncio
async def test_snapshot_dir_without_rollout_index_keeps_legacy_naming(tmp_path) -> None:
    agent = _agent_with_one_answer(tmp_path)
    request_mock, response_mock = _mocks()
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "q"}],
        metadata={"task_index": "7", "attempt": "0"},
    )
    await agent.responses(request_mock, response_mock, body)

    assert (tmp_path / "sample_7" / "attempt_0_final.jsonl").exists()
    assert not list(tmp_path.glob("sample_7_r*"))


@pytest.mark.asyncio
async def test_run_seeds_rollout_index_metadata(tmp_path) -> None:
    """run() forwards the row's _ng_rollout_index into responses metadata (snap_dir set)."""
    agent = BrowsecompAgent(
        config=_make_config(snap_dir=str(tmp_path)),
        server_client=MagicMock(spec=ServerClient),
    )
    attempt0 = _make_model_response([_make_msg("Exact Answer: DONE")])
    verify_json = {
        "reward": 1.0,
        "response": attempt0,
        "responses_create_params": {"input": [{"role": "user", "content": "q"}]},
    }

    def _http(read_bytes=None):
        m = MagicMock()
        m.ok = True
        m.cookies = {}
        if read_bytes is not None:
            m.read = AsyncMock(return_value=read_bytes)
        return m

    agent.server_client.post = AsyncMock(
        side_effect=[_http(), _http(json.dumps(attempt0).encode()), _http(json.dumps(verify_json).encode())]
    )
    request_mock = MagicMock()
    request_mock.cookies = {}
    body = BrowsecompAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "q"}]),
        _ng_task_index=7,
        _ng_rollout_index=2,
    )
    await agent.run(request_mock, body)

    responses_call = agent.server_client.post.call_args_list[1]
    metadata = responses_call.kwargs["json"].metadata
    assert metadata["task_index"] == "7"
    assert metadata["rollout_index"] == "2"

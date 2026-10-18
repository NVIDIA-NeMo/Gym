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
"""The response carries stop_reason: why the agent loop ended (answered / max_steps / context_full / timeout)."""

import json
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent
from responses_api_agents.browsecomp_agent.tests.test_app import _make_config, _make_fn_call, _make_model_response, _make_msg


async def _run(config_kwargs: dict, model_turn: dict):
    agent = BrowsecompAgent(config=_make_config(**config_kwargs), server_client=MagicMock(spec=ServerClient))
    http = MagicMock()
    http.ok = True
    http.status = 200
    http.read = AsyncMock(return_value=json.dumps(model_turn).encode())
    http.content.read = AsyncMock(return_value=b'{"results_string": "r"}')
    http.cookies = {}
    agent.server_client.post = AsyncMock(return_value=http)
    request_mock = MagicMock()
    request_mock.cookies = {}
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q?"}])
    return await agent.responses(request_mock, MagicMock(), body)


async def test_answer_sets_answered() -> None:
    result = await _run({}, _make_model_response([_make_msg("Exact Answer: Paris")]))
    assert result.stop_reason == "answered"


async def test_step_budget_sets_max_steps() -> None:
    turn = _make_model_response([_make_fn_call("search", args={"queries": ["q"]})])
    result = await _run({"max_steps": 2, "nudge_steps": False}, turn)
    assert result.stop_reason == "max_steps"


async def test_length_cut_sets_context_full() -> None:
    # The policy-model server answers a context overflow with an empty, length-cut response.
    turn = _make_model_response([])
    turn["status"] = "incomplete"
    turn["incomplete_details"] = {"reason": "max_output_tokens"}
    result = await _run({}, turn)
    assert result.stop_reason == "context_full"


async def test_deadline_sets_timeout() -> None:
    turn = _make_model_response([_make_fn_call("search", args={"queries": ["q"]})])
    result = await _run({"rollout_timeout_s": 0.0}, turn)
    assert result.timed_out is True
    assert result.stop_reason == "timeout"

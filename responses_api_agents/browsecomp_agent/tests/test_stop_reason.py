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
"""The response carries stop_reason: why the agent loop ended.

Values: answered / max_steps / max_output_tokens / context_full / timeout.
"""

import json
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent
from responses_api_agents.browsecomp_agent.tests.test_app import (
    _make_config,
    _make_fn_call,
    _make_model_response,
    _make_msg,
)


async def _run(config_kwargs: dict, model_turn: dict, max_output_tokens: int | None = None):
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
    body = NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "user", "content": "Q?"}], max_output_tokens=max_output_tokens
    )
    return await agent.responses(request_mock, MagicMock(), body)


async def test_answer_sets_answered() -> None:
    result = await _run({}, _make_model_response([_make_msg("Exact Answer: Paris")]))
    assert result.stop_reason == "answered"


async def test_step_budget_sets_max_steps() -> None:
    turn = _make_model_response([_make_fn_call("search", args={"queries": ["q"]})])
    result = await _run({"max_steps": 2, "nudge_steps": False}, turn)
    assert result.stop_reason == "max_steps"


def _length_cut(output_tokens: int | None) -> dict:
    turn = _make_model_response([])
    turn["status"] = "incomplete"
    turn["incomplete_details"] = {"reason": "max_output_tokens"}
    if output_tokens is not None:
        turn["usage"] = {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": output_tokens,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 10 + output_tokens,
        }
    return turn


async def test_context_overflow_sets_context_full() -> None:
    # The policy-model server answers a context overflow with an empty, length-cut response and zero usage.
    result = await _run({}, _length_cut(0), max_output_tokens=100)
    assert result.stop_reason == "context_full"


async def test_cut_before_the_output_budget_sets_context_full() -> None:
    # Generation stopped at the context window before the requested budget was spent.
    result = await _run({}, _length_cut(40), max_output_tokens=100)
    assert result.stop_reason == "context_full"


async def test_cut_at_the_output_budget_sets_max_output_tokens() -> None:
    result = await _run({}, _length_cut(100), max_output_tokens=100)
    assert result.stop_reason == "max_output_tokens"


async def test_cut_without_usage_or_budget_sets_context_full() -> None:
    assert (await _run({}, _length_cut(None), max_output_tokens=100)).stop_reason == "context_full"
    assert (await _run({}, _length_cut(100))).stop_reason == "context_full"


async def test_deadline_sets_timeout() -> None:
    turn = _make_model_response([_make_fn_call("search", args={"queries": ["q"]})])
    result = await _run({"rollout_timeout_s": 0.0}, turn)
    assert result.timed_out is True
    assert result.stop_reason == "timeout"

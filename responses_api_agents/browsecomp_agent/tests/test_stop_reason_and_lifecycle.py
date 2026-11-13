# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""stop_reason on a timeout after the model already answered, and agent sandbox lifecycle records."""

import asyncio
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent
from responses_api_agents.browsecomp_agent.sandbox import BrowsecompSandboxSession
from responses_api_agents.browsecomp_agent.tests.test_agent_sessions import _seed
from responses_api_agents.browsecomp_agent.tests.test_app import _make_config, _make_fn_call, _make_model_response


async def test_tool_call_past_the_deadline_labels_the_model_response_timeout() -> None:
    # The model answered once, so the loop ends on its response rather than a synthetic resp_timeout_*.
    agent = BrowsecompAgent(config=_make_config(rollout_timeout_s=0.5), server_client=MagicMock(spec=ServerClient))
    turn = _make_model_response([_make_fn_call("search", args={"queries": ["q"]})])

    async def post(**kwargs):
        http = MagicMock(ok=True, status=200, cookies={})
        if kwargs["url_path"].endswith("/v1/responses"):
            http.read = AsyncMock(return_value=json.dumps(turn).encode())
            return http
        await asyncio.sleep(5)  # the search outlives the rollout deadline
        raise AssertionError("unreachable")

    agent.server_client.post = AsyncMock(side_effect=post)
    request = MagicMock(cookies={})
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q?"}])

    result = await agent.responses(request, MagicMock(), body)

    assert not result.id.startswith("resp_timeout_")
    assert result.timed_out is True
    assert result.status == "incomplete"
    assert result.stop_reason == "timeout"


@pytest.mark.parametrize(("owns_sandbox", "event"), [(True, "sandbox_stop"), (False, "sandbox_disconnect")])
async def test_agent_close_logs_once_per_successful_release(caplog, owns_sandbox: bool, event: str) -> None:
    caplog.set_level(logging.INFO)
    session = SimpleNamespace(
        sandbox=SimpleNamespace(provider_name="docker", sandbox_id="agent-box"),
        owns_sandbox=owns_sandbox,
        close=AsyncMock(side_effect=[RuntimeError("stop failed"), None, None]),
    )
    seed = _seed()
    state = BrowsecompSandboxSession(
        request=seed, session=session, tool_access=None, resources_cookies={}, started_at=time.monotonic()
    )

    with pytest.raises(RuntimeError, match="stop failed"):
        await state.close(timeout=1)
    await state.close(timeout=1)  # the retry that succeeds
    await state.close(timeout=1)  # a repeated close after success

    lines = [r.message for r in caplog.records if "role=agent" in r.message]
    assert len(lines) == 2, lines
    assert all(line.startswith(f"[browsecomp][{event}] ") for line in lines)
    assert "status=failed:RuntimeError" in lines[0] and "status=ok" in lines[1]
    assert all(f"session={seed.agent_session_id} provider=docker id=agent-box" in line for line in lines)
    assert float(lines[1].rsplit("lifetime_s=", 1)[1]) >= 0.0

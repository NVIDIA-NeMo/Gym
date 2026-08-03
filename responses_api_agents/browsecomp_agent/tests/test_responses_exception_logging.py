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
"""An exception out of responses() must name itself in the log.

Diagnostic instrumentation, added because job 5794208 lost 6/400 samples to
`ClientResponseError: 500` from the agent's own /v1/responses and **the log contained no
traceback at all** — no `🚨` marker, no app.py frame, nothing. The malformed-JSON bug
(bdfba2f4) printed a full ExceptionGroup via exception_handling_middleware, so the absence
here means the failure does not surface as an ordinary application exception.

Without a traceback the cause can only be bounded, not named. These tests pin the
requirement that responses() logs whatever escapes it — and, critically, that it still
RE-RAISES, so run()'s containment (b7715fc8) keeps working exactly as before.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent, BrowsecompAgentConfig


def _make_agent() -> BrowsecompAgent:
    config = BrowsecompAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="test_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        nudge_steps=False,
    )
    return BrowsecompAgent(config=config, server_client=MagicMock(spec=ServerClient))


async def _call(agent: BrowsecompAgent):
    request_mock = MagicMock()
    request_mock.cookies = {}
    response_mock = MagicMock()
    response_mock.set_cookie = MagicMock()
    body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "Q?"}])
    return await agent.responses(request_mock, response_mock, body)


class _Boom(RuntimeError):
    pass


class TestResponsesLogsWhatEscapes:
    async def test_traceback_is_printed(self, capfd) -> None:
        agent = _make_agent()
        agent._responses_impl = AsyncMock(side_effect=_Boom("kaboom"))
        with pytest.raises(_Boom):
            await _call(agent)
        # readouterr() DRAINS the buffer — capture once. print() goes to stdout,
        # traceback.print_exc() to stderr, so both halves are needed.
        cap = capfd.readouterr()
        out = cap.out + cap.err
        assert "_Boom" in out or "kaboom" in out, "the exception must name itself in the log"
        assert "Traceback" in out, "a full traceback must be printed, not just the message"

    async def test_exception_still_propagates(self, capfd) -> None:
        """Logging must NOT swallow it — run()'s containment depends on it re-raising."""
        agent = _make_agent()
        agent._responses_impl = AsyncMock(side_effect=_Boom("kaboom"))
        with pytest.raises(_Boom):
            await _call(agent)

    async def test_marker_is_greppable(self, capfd) -> None:
        """A stable prefix so the next run can be grepped without knowing the exception."""
        agent = _make_agent()
        agent._responses_impl = AsyncMock(side_effect=_Boom("kaboom"))
        with pytest.raises(_Boom):
            await _call(agent)
        cap = capfd.readouterr()
        out = cap.out + cap.err
        assert "[browsecomp][responses_exc]" in out


class TestHappyPathUnaffected:
    async def test_successful_call_passes_through_untouched(self, capfd) -> None:
        agent = _make_agent()
        sentinel = object()
        agent._responses_impl = AsyncMock(return_value=sentinel)
        assert await _call(agent) is sentinel
        out = capfd.readouterr().out
        assert "[browsecomp][responses_exc]" not in out, "a clean call must log nothing"

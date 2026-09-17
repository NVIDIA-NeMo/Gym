# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Malformed model output must reach tool validation without a logger crash."""

from unittest.mock import MagicMock

import pytest
from stirrup.core.models import AssistantMessage, ToolCall

from responses_api_agents.stirrup_agent.nemo_agent import NeMoAgent


@pytest.mark.parametrize("arguments", [r'{"cmd":"bad\escape"}', '{"cmd":"unfinished', '{"cmd":"echo ok"}'])
def test_default_logger_preserves_tool_arguments(arguments, capsys):
    agent = NeMoAgent(client=MagicMock(), name="test_agent", tools=[])
    message = AssistantMessage(
        content="Run a command",
        tool_calls=[ToolCall(tool_call_id="call-1", name="code_exec", arguments=arguments)],
    )
    before = message.model_dump()

    agent._logger.assistant_message(1, 2, message)

    output = capsys.readouterr().out
    assert message.model_dump() == before
    if arguments != '{"cmd":"echo ok"}':
        assert "Cannot format tool arguments as JSON" in output
        assert "Run a command" in output
        assert "raw arguments" in output
    else:
        assert "Cannot format" not in output


def test_explicit_logger_is_preserved():
    logger = MagicMock()
    agent = NeMoAgent(client=MagicMock(), name="test_agent", tools=[], logger=logger)
    assert agent._logger is logger

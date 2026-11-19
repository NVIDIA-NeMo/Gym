# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from resources_servers.opencode_multimodal_benchmarks.tests import tool_adapter_checks
from resources_servers.opencode_multimodal_benchmarks.tests.test_rewards import tool_response
from resources_servers.string_match.app import (
    StringMatchResourcesServer,
    StringMatchResourcesServerConfig,
    StringMatchVerifyRequest,
)
from resources_servers.string_match_tool_rewards.app import (
    ToolRewardConfig,
    ToolRewardServer,
    ToolRewardVerifyRequest,
)


class TestAdapter(tool_adapter_checks.AdapterChecks):
    component = "string_match_tool_rewards"
    parent = "string_match"
    invalid_fields = {"extraction_mode": "invalid", "case_sensitive": "invalid"}


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_rewards", [None, {}])
@pytest.mark.parametrize("expected_answer,expected_reward", [("12\n34", 1.0), ("1234", 0.0)])
@pytest.mark.parametrize("strip_reasoning", [False, True])
async def test_split_answer_preserves_native_string_match(
    tool_rewards, expected_answer, expected_reward, strip_reasoning
):
    response = tool_response("12")
    response.output = response.output[-1:]
    message = response.output[0]
    message.content.append(message.content[0].model_copy(update={"text": "34"}))
    original = response.model_dump(mode="json")
    row = dict(
        responses_create_params={"input": []},
        response=response,
        expected_answer=expected_answer,
        extraction_mode="full_response",
    )
    client = MagicMock(spec=ServerClient)
    native = StringMatchResourcesServer(
        config=StringMatchResourcesServerConfig(name="native", host="127.0.0.1", port=1, entrypoint="app.py"),
        server_client=client,
    )
    adapter = ToolRewardServer(
        config=ToolRewardConfig(
            name="adapter",
            host="127.0.0.1",
            port=2,
            entrypoint="app.py",
            tool_rewards=tool_rewards,
            strip_reasoning_for_grading=strip_reasoning,
        ),
        server_client=client,
    )
    native_result = await native.verify(StringMatchVerifyRequest(**row))
    result = await adapter.verify(ToolRewardVerifyRequest(**row))
    assert result.reward == native_result.reward == expected_reward
    assert result.response.model_dump(mode="json") == original
    assert response.model_dump(mode="json") == original

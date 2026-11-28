# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from resources_servers.visgym.app import VisGymResourcesServer
from resources_servers.visgym.schemas import (
    VisGymAgentVerifyRequest,
    VisGymNeMoGymResponse,
    VisGymResourcesServerConfig,
)


@pytest.mark.parametrize(
    "mode,success,expected,masked",
    [
        ("training", True, 0.4, False),
        ("task_success", True, 1.0, False),
        ("task_success", False, 0.0, False),
        ("task_success", None, 0.0, True),
    ],
)
def test_score_contract(mode, success, expected, masked):
    server = VisGymResourcesServer.model_construct(
        config=VisGymResourcesServerConfig(
            name="test", host="127.0.0.1", port=12345, entrypoint="app.py", reward_mode=mode
        ),
        env_id_to_total_reward={"episode": 0.4},
        env_id_to_task_success={"episode": success},
    )
    response = VisGymNeMoGymResponse(
        id="response",
        created_at=0,
        model="policy",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="auto",
        tools=[],
        env_id="episode",
    )
    request = VisGymAgentVerifyRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input=[]), response=response
    )
    result = server._verify_session(request)
    assert result.reward == expected
    assert result.mask_sample is masked
    assert result.response.id == "response"
    if masked:
        assert result.failure_kind == "visgym:success_unavailable"
    repeated = server._verify_session(request)
    assert repeated.mask_sample is True
    assert repeated.failure_kind == "session_lost"

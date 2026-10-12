# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.sandbox import agent_tools


@pytest.mark.parametrize("host", ["localhost", "localhost.", "127.0.0.1", "127.0.0.2", "[::1]", "0.0.0.0", "[::]"])
def test_remote_endpoint_rejects_unreachable_bind(host, monkeypatch):
    monkeypatch.setattr(agent_tools, "get_server_url", lambda _: f"http://{host}:8000")
    with pytest.raises(ValueError, match="use_absolute_ip=true"):
        agent_tools.sandbox_server_url("model", require_reachable=True)


@pytest.mark.parametrize(
    "host,expected",
    [("127.0.0.1", "127.0.0.1"), ("localhost", "localhost"), ("0.0.0.0", "127.0.0.1"), ("[::]", "[::1]")],
)
def test_inherited_network_preserves_host_network_connectivity(host, expected, monkeypatch):
    monkeypatch.setattr(agent_tools, "get_server_url", lambda _: f"http://{host}:8000")
    assert agent_tools.sandbox_server_url("model") == f"http://{expected}:8000"


@pytest.mark.parametrize("reward,masked", [(1, False), (0, True)])
async def test_failure_zero_rejects_incompatible_verifier(reward, masked):
    original = NeMoGymResponse.model_validate(
        dict(
            id="x",
            created_at=0,
            model="test",
            object="response",
            output=[],
            tool_choice="auto",
            tools=[],
            parallel_tool_calls=True,
        )
    )
    body = BaseRunRequest(responses_create_params={"input": "x"})
    reply = SimpleNamespace(
        ok=True,
        read=AsyncMock(return_value=json.dumps(dict(reward=reward, mask_sample=masked)).encode()),
        raise_for_status=lambda: None,
    )
    with pytest.raises(ValueError, match="unmasked zero"):
        await agent_tools.verify_agent_response(
            SimpleNamespace(post=AsyncMock(return_value=reply)),
            ResourcesServerRef(type="resources_servers", name="math"),
            body,
            original,
            {},
            force_zero_reward=True,
        )

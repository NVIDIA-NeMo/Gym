# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from nemo_gym.agents import agent_tools
from nemo_gym.config.config_types import ResourcesServerRef
from nemo_gym.core.responses_api_types import NeMoGymResponse
from nemo_gym.server_runtime.base_resources_server import BaseRunRequest


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


def test_openshell_network_policy_allows_only_named_endpoints():
    policy = agent_tools.restricted_network_policy(
        "openshell",
        ["http://10.1.2.3:8000/v1", "https://model.example", "http://tools.example:9000", "http://10.1.2.3:8000"],
    )
    rule = policy["network_policies"]["nemo_gym_endpoints"]
    assert policy["version"] == 1
    assert rule["binaries"] == [{"path": "/**"}]
    assert rule["endpoints"] == [
        {"host": "10.1.2.3", "port": 8000, "allowed_ips": ["10.1.2.3/32"]},
        {"host": "model.example", "port": 443},
        {"host": "tools.example", "port": 9000},
    ]


def test_openshell_network_policy_public_ip_needs_no_allowed_ips():
    rule = agent_tools.restricted_network_policy("openshell", ["http://8.8.8.8:80"])["network_policies"][
        "nemo_gym_endpoints"
    ]
    assert rule["endpoints"] == [{"host": "8.8.8.8", "port": 80}]


@pytest.mark.parametrize("url", ["http://localhost:8000", "http://127.0.0.1:8000", "http://0.0.0.0:8000"])
def test_openshell_network_policy_rejects_unreachable_hosts(url):
    with pytest.raises(ValueError, match="reachable"):
        agent_tools.restricted_network_policy("openshell", [url])


def test_openshell_network_policy_requires_a_derivable_port():
    with pytest.raises(ValueError, match="port"):
        agent_tools.restricted_network_policy("openshell", ["ftp://model.example"])

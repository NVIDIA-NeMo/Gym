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


class TestPhaseNetworkPolicy:
    """`phase_network_policy`: a Harbor phase's `network_mode` as the OpenSandbox egress policy, failing closed."""

    def test_public_phases_have_no_policy(self):
        assert agent_tools.phase_network_policy("opensandbox", None) is None
        assert agent_tools.phase_network_policy("docker", "public") is None
        assert agent_tools.phase_network_policy(None, "public", []) is None
        with pytest.raises(ValueError, match="only valid when network_mode='allowlist'"):
            agent_tools.phase_network_policy("opensandbox", "public", ["example.org"])

    def test_no_network_denies_everything(self):
        assert agent_tools.phase_network_policy("opensandbox", "no-network") == {"defaultAction": "deny", "egress": []}
        with pytest.raises(ValueError, match="only valid when network_mode='allowlist'"):
            agent_tools.phase_network_policy("opensandbox", "no-network", ["example.org"])

    def test_allowlist_hosts_become_ordered_allow_rules(self):
        policy = agent_tools.phase_network_policy(
            "opensandbox", "allowlist", ["PyPI.org", "*.nvidia.com", "pypi.org.", "10.0.0.7", "2001:DB8:0::1"]
        )
        assert policy == {
            "defaultAction": "deny",
            "egress": [
                {"action": "allow", "target": "pypi.org"},
                {"action": "allow", "target": "*.nvidia.com"},
                {"action": "allow", "target": "10.0.0.7"},
                {"action": "allow", "target": "2001:db8::1"},
            ],
        }
        # An empty allowlist is a deny-all policy, as Harbor resolves it.
        assert agent_tools.phase_network_policy("opensandbox", "allowlist", None) == {
            "defaultAction": "deny",
            "egress": [],
        }

    @pytest.mark.parametrize(
        ("host", "reason"),
        [
            ("10.0.0.0/8", "CIDR ranges are not supported"),
            ("", "must not be empty"),
            ("http://pypi.org", "not a host name"),
            ("pypi org", "not a host name"),
            ("*pypi.org", "not a host name"),
            # The spellings Harbor's `normalize_allowed_hosts` refuses at task parse are refused here too, rather
            # than forwarded verbatim as rules the sidecar never matches.
            ("*.10.0.0.1", "wildcards must target host names, not IP addresses"),
            ("fe80::1%eth0", "zone ids are not supported"),
            ("[2001:db8::1]", "written without brackets"),
        ],
    )
    def test_entries_the_policy_cannot_express_are_refused(self, host, reason):
        with pytest.raises(ValueError, match=reason):
            agent_tools.phase_network_policy("opensandbox", "allowlist", ["ok.example", host])

    @pytest.mark.parametrize("provider", ["docker", "daytona", None])
    def test_other_providers_fail_closed(self, provider):
        with pytest.raises(ValueError, match="requires the OpenSandbox network-policy provider"):
            agent_tools.phase_network_policy(provider, "no-network")
        with pytest.raises(ValueError, match="requires the OpenSandbox"):
            agent_tools.phase_network_policy(provider, "allowlist", ["example.org"])

    def test_unknown_mode_is_refused(self):
        with pytest.raises(ValueError, match="Unknown network_mode"):
            agent_tools.phase_network_policy("opensandbox", "firewalled")

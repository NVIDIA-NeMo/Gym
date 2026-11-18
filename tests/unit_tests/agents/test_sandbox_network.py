# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from nemo_gym.agents.base_responses_api_agent import AgentSeedRejectedError
from nemo_gym.agents.utils.sandbox_network import restrict_sandbox_network
from nemo_gym.sandbox.providers.base import SandboxSpec


MODEL_URL = "http://model.example:8000/v1"
TOOL_URLS = ["http://tools.example:9000/mcp", "http://search.example/mcp"]
# The helper reads only the provider's name until providers declare egress capabilities.
OPENSANDBOX = SimpleNamespace(name="opensandbox")
DOCKER = SimpleNamespace(name="docker")


@pytest.mark.parametrize("provider_options", [{}, {"network_policy": {"defaultAction": "allow"}}])
def test_inherit_returns_the_spec_unchanged(provider_options: dict[str, object]) -> None:
    spec = SandboxSpec(image="task", provider_options=provider_options)
    restricted = restrict_sandbox_network(
        spec, network_access="inherit", provider=OPENSANDBOX, model_base_url=MODEL_URL, tool_urls=TOOL_URLS
    )
    assert restricted is spec


@pytest.mark.parametrize(
    ("network_access", "allowed"),
    [
        ("model_only", ["model.example"]),
        ("model_and_tools", ["model.example", "search.example", "tools.example"]),
    ],
)
def test_restricted_access_denies_all_egress_but_the_allowed_hosts(network_access: str, allowed: list[str]) -> None:
    spec = SandboxSpec(image="task", provider_options={"other": 1})
    restricted = restrict_sandbox_network(
        spec, network_access=network_access, provider=OPENSANDBOX, model_base_url=MODEL_URL, tool_urls=TOOL_URLS
    )
    assert restricted.provider_options == {
        "other": 1,
        "network_policy": {"defaultAction": "deny", "egress": [{"action": "allow", "target": h} for h in allowed]},
    }
    assert restricted.image == "task"
    assert spec.provider_options == {"other": 1}


def test_restricted_access_rejects_an_explicit_network_policy() -> None:
    spec = SandboxSpec(image="task", provider_options={"network_policy": {"defaultAction": "allow"}})
    with pytest.raises(AgentSeedRejectedError, match="not both") as refused:
        restrict_sandbox_network(spec, network_access="model_only", provider=OPENSANDBOX, model_base_url=MODEL_URL)
    # A configuration error: the seed fails once with HTTP 422 instead of being retried.
    assert refused.value.status_code == 422


def test_restricted_access_fails_closed_on_a_provider_without_network_policies() -> None:
    with pytest.raises(AgentSeedRejectedError, match="requires the OpenSandbox") as refused:
        restrict_sandbox_network(
            SandboxSpec(image="task"), network_access="model_only", provider=DOCKER, model_base_url=MODEL_URL
        )
    assert refused.value.status_code == 422


@pytest.mark.parametrize("model_base_url", ["http://127.0.0.1:8000/v1", "http://localhost:8000/v1"])
def test_restricted_access_refuses_an_endpoint_the_sandbox_cannot_reach(model_base_url: str) -> None:
    with pytest.raises(AgentSeedRejectedError, match="sandbox-reachable host") as refused:
        restrict_sandbox_network(
            SandboxSpec(image="task"),
            network_access="model_only",
            provider=OPENSANDBOX,
            model_base_url=model_base_url,
        )
    assert refused.value.status_code == 422


def test_a_trailing_dot_on_a_host_is_dropped_from_its_rule() -> None:
    # The egress sidecar strips a query name's trailing dot, so "model.example." would never match.
    restricted = restrict_sandbox_network(
        SandboxSpec(image="task"),
        network_access="model_only",
        provider=OPENSANDBOX,
        model_base_url="http://model.example.:8000/v1",
    )
    assert restricted.provider_options["network_policy"]["egress"] == [{"action": "allow", "target": "model.example"}]


def test_unset_provider_options_are_restricted_like_empty_ones() -> None:
    spec = SandboxSpec(image="task")
    object.__setattr__(spec, "provider_options", None)
    restricted = restrict_sandbox_network(
        spec, network_access="model_only", provider=OPENSANDBOX, model_base_url=MODEL_URL
    )
    assert restricted.provider_options["network_policy"]["egress"] == [{"action": "allow", "target": "model.example"}]


@pytest.mark.parametrize("network_access", ["model_only", "model_and_tools"])
def test_a_required_tool_grant_needs_model_and_tools(network_access: str) -> None:
    spec = SandboxSpec(image="task")

    def restrict() -> SandboxSpec:
        return restrict_sandbox_network(
            spec,
            network_access=network_access,
            provider=OPENSANDBOX,
            model_base_url=MODEL_URL,
            tool_urls=TOOL_URLS,
            required_tool_urls=TOOL_URLS[:1],
        )

    if network_access == "model_only":
        # model_only blocks the grant's host, so the session would fail only after the sandbox exists.
        with pytest.raises(AgentSeedRejectedError, match="model_and_tools"):
            restrict()
        return
    assert {rule["target"] for rule in restrict().provider_options["network_policy"]["egress"]} >= {"tools.example"}

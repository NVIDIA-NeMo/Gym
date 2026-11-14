# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import logging
import os
import subprocess
import sys
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


def test_a_restriction_is_logged_with_its_mode_and_hosts(caplog: pytest.LogCaptureFixture) -> None:
    # The log line is the only record of whether a run's sandbox was restricted.
    with caplog.at_level(logging.INFO, logger="nemo_gym.agents.utils.sandbox_network"):
        restrict_sandbox_network(
            SandboxSpec(image="task"),
            network_access="model_and_tools",
            provider=OPENSANDBOX,
            model_base_url=MODEL_URL,
            tool_urls=TOOL_URLS,
        )
    [record] = [r for r in caplog.records if r.name == "nemo_gym.agents.utils.sandbox_network"]
    # Where INFO reaches a handler, the record stays at INFO.
    assert record.levelno == logging.INFO
    assert "network_access=model_and_tools" in record.getMessage()
    assert "model.example, search.example, tools.example" in record.getMessage()


# An agent server process as Gym starts it: the supervisor injects the config, which configures logging only with
# `verbose`, and uvicorn then configures its own loggers.
AGENT_PROCESS = """
import logging.config
import sys
from types import SimpleNamespace

import uvicorn.config

from nemo_gym.agents.utils.sandbox_network import restrict_sandbox_network
from nemo_gym.config.global_config import get_global_config_dict
from nemo_gym.sandbox.providers.base import SandboxSpec

get_global_config_dict()
logging.config.dictConfig(uvicorn.config.LOGGING_CONFIG)
restrict_sandbox_network(
    SandboxSpec(image="task"),
    network_access=sys.argv[1],
    provider=SimpleNamespace(name="opensandbox"),
    model_base_url="http://model.example:8000/v1?key=model-secret",
    tool_urls=["https://user:tool-secret@tools.example/mcp/path?q=1"],  # pragma: allowlist secret
)
"""


@pytest.mark.parametrize("verbose", [False, True])
@pytest.mark.parametrize(
    ("network_access", "hosts"),
    [("model_only", "model.example"), ("model_and_tools", "model.example, tools.example")],
)
def test_an_agent_server_process_prints_one_restriction_record(network_access: str, hosts: str, verbose: bool) -> None:
    env = {key: value for key, value in os.environ.items() if not key.startswith(("NEMO_GYM_OTEL_", "NEMO_LENS_"))}
    env["NEMO_GYM_CONFIG_DICT"] = f"{{verbose: {str(verbose).lower()}}}"
    result = subprocess.run(
        [sys.executable, "-c", AGENT_PROCESS, network_access],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=True,
    )
    [line] = [line for line in result.stderr.splitlines() if "Restricting a new" in line]
    assert line.endswith(
        f"Restricting a new opensandbox sandbox to network_access={network_access}: egress allowed only to {hosts}"
    )
    for leaked in ("secret", "user", "/mcp", "/v1", "q=1", "8000"):
        assert leaked not in line


def test_inherit_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="nemo_gym.agents.utils.sandbox_network"):
        restrict_sandbox_network(
            SandboxSpec(image="task"), network_access="inherit", provider=OPENSANDBOX, model_base_url=MODEL_URL
        )
    assert not [r for r in caplog.records if r.name == "nemo_gym.agents.utils.sandbox_network"]


def test_a_rejected_restriction_logs_no_restriction(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="nemo_gym.agents.utils.sandbox_network"):
        with pytest.raises(AgentSeedRejectedError):
            restrict_sandbox_network(
                SandboxSpec(image="task"), network_access="model_only", provider=DOCKER, model_base_url=MODEL_URL
            )
    assert not [r for r in caplog.records if r.name == "nemo_gym.agents.utils.sandbox_network"]

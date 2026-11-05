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
"""Compose a session harness's `configs/profiles/benchmark.yaml` onto HLE and seed it there.

Shared by the profile checks in `tests/unit_tests/environment` and each adapter's own seed test, which runs in the
adapter's environment so its modules count toward that adapter's suite.
"""

import importlib
from pathlib import Path
from typing import Any

import httpx
import pytest
from omegaconf import DictConfig, OmegaConf

from environment_servers.single_agent_turn.app import SessionHandles
from environment_servers.single_agent_turn_legacy.app import (
    SingleAgentTurnLegacyEnvironmentServer,
    SingleAgentTurnLegacyEnvironmentServerConfig,
)
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig
from nemo_gym.config.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
)
from nemo_gym.server_runtime.base_resources_server import ResourcesSeedSessionResponse


REPO_ROOT = Path(__file__).resolve().parents[1]
HLE_ENVIRONMENT = "hle_equivalence_llm_judge_environment_server"
# The override each adapter README gives for HLE: its judge grants the agent no tools.
HLE_PAIRING = {
    HLE_ENVIRONMENT: {"environment_servers": {"single_agent_turn_legacy": {"resources_tool_transports": []}}}
}
IMAGE = "registry.example.invalid/sandbox:1"
# Values for the harness settings a restricted profile requires (`???`): a restricted sandbox cannot download the
# harness, so the image carries its runtime or binary.
REQUIRED_SETTING_VALUES = {
    "sandbox_runtime_dir": "/opt/harness-runtime",
    "sandbox_executable_path": "/opt/harness/bin/harness",
    "sandbox_executable_sha256": "0" * 64,
}

# Adapter directory name -> (agent class, its config class).
HARNESSES = {
    "pi_agent": ("PiAgent", "PiAgentConfig"),
    "opencode_agent": ("OpenCodeAgent", "OpenCodeAgentConfig"),
    "codex_agent": ("CodexAgent", "CodexAgentConfig"),
    "openclaw_agent": ("OpenClawAgent", "OpenClawAgentConfig"),
    "hermes_agent": ("HermesAgent", "HermesAgentConfig"),
    "claude_code_agent": ("ClaudeCodeAgent", "ClaudeCodeAgentConfig"),
}


def profile_settings(agent: str) -> dict[str, Any]:
    """The agent block of the adapter's benchmark profile, as written, with ``???`` kept for a required value."""
    profile = OmegaConf.load(REPO_ROOT / "responses_api_agents" / agent / "configs" / "profiles" / "benchmark.yaml")
    return OmegaConf.to_container(profile[agent].responses_api_agents[agent], resolve=False)


# Read from each profile, so a harness that restricts its profile changes only its own files.
MODEL_ONLY_PROFILES = {agent for agent in HARNESSES if profile_settings(agent).get("network_access") == "model_only"}
# The harness settings each profile requires the caller to supply.
REQUIRED_SETTINGS = {
    agent: sorted(key for key in REQUIRED_SETTING_VALUES if profile_settings(agent).get(key) == "???")
    for agent in HARNESSES
}


def compose_profile_on_hle(
    agent: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    pairing: bool = True,
    image: bool = True,
    required: bool = True,
    cli_args: tuple = (),
) -> DictConfig:
    """Compose the agent's benchmark profile onto HLE, with the README's override unless `pairing` is False."""
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "unused")
    composed = f"hle_equivalence_llm_judge_{agent}"
    # Set under the composed agent's name, as the missing-value error for the profile's `image: ???` suggests.
    if image:
        cli_args = (f"++{composed}.responses_api_agents.{agent}.sandbox_config.image={IMAGE}", *cli_args)
    if required:
        settings = f"++{composed}.responses_api_agents.{agent}"
        cli_args = (
            *(f"{settings}.{key}={REQUIRED_SETTING_VALUES[key]}" for key in REQUIRED_SETTINGS[agent]),
            *cli_args,
        )
    monkeypatch.setattr("sys.argv", ["gym", *cli_args])
    return GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            skip_load_from_cli=not cli_args,
            skip_load_from_dotenv=True,
            offline=True,
            initial_global_config_dict=OmegaConf.create(
                {
                    "config_paths": [
                        "responses_api_models/vllm_model/configs/vllm_model.yaml",
                        "benchmarks/hle/config.yaml",
                        f"responses_api_agents/{agent}/configs/profiles/benchmark.yaml",
                    ],
                    "policy_base_url": "http://unused.invalid/v1",
                    "policy_api_key": "unused",
                    "policy_model_name": "test-model",
                    **(HLE_PAIRING if pairing else {}),
                }
            ),
        )
    )


def server_config(config: DictConfig, name: str) -> dict[str, Any]:
    return OmegaConf.to_container(get_first_server_config_dict(config, name), resolve=True) | {"name": name}


async def seed_through_hle(
    agent: str, config: DictConfig, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict[str, Any], httpx.Response | None, bool]:
    """Seed the composed agent with the request HLE's Environment Server sends it.

    Returns that request, the agent's reply if it gave one, and whether the agent went on to create its own sandbox.
    """
    module = importlib.import_module(f"responses_api_agents.{agent}.app")
    agent_class, config_class = (getattr(module, name) for name in HARNESSES[agent])
    agent_name = f"hle_equivalence_llm_judge_{agent}"
    sandbox_requests: list[str] = []
    # An offline parse assigns no ports; the direct_http grant carries the Resources server's URL.
    get_first_server_config_dict(config, "hle_equivalence_llm_judge_resources_server").port = 12001

    def resolve_provider_config(provider_ref: str, *args: Any, **kwargs: Any) -> None:
        sandbox_requests.append(provider_ref)
        raise RuntimeError("the test stops at sandbox creation")

    # Every adapter resolves its provider's config only after it accepts the seed, to create its own sandbox.
    monkeypatch.setattr(module, "resolve_provider_config", resolve_provider_config)
    client = ServerClient(head_server_config=BaseServerConfig(host="127.0.0.1", port=11000), global_config_dict=config)
    harness = agent_class(config=config_class.model_validate(server_config(config, agent_name)), server_client=client)
    environment = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(server_config(config, HLE_ENVIRONMENT)),
        server_client=client,
    )
    seeds: list[dict[str, Any]] = []
    replies: list[httpx.Response] = []

    async def post(self: ServerClient, server_name: str, url_path: str, **kwargs: Any) -> httpx.Response:
        assert (server_name, url_path) == (agent_name, "/v1/agent_sessions")
        seeds.append(kwargs["json"])
        transport = httpx.ASGITransport(app=harness.setup_webserver())
        async with httpx.AsyncClient(transport=transport, base_url="http://agent") as agent_http:
            replies.append(await agent_http.post(url_path, json=kwargs["json"]))
        return replies[-1]

    monkeypatch.setattr(ServerClient, "post", post)
    request = environment._episode_request_from_row(
        {"_ng_task_index": 0, "_ng_rollout_index": 0, "responses_create_params": {"input": "task"}}
    )
    handles = SessionHandles(resources_session_id="resources-session", agent_session_id="agent-session")
    handles.resources_cookies = {"session": "resources"}
    try:
        # HLE's Resources server lends no sandbox.
        await environment._seed_agent(
            request, handles, ResourcesSeedSessionResponse(resources_session_id="resources-session")
        )
    except Exception:
        # The agent's reply is read directly; the Environment Server's handling of it is not under test here.
        if not seeds:
            raise
    return seeds[-1], (replies[-1] if replies else None), bool(sandbox_requests)


async def check_hle_seed(agent: str, pairing: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Seed the composed agent through HLE's Environment Server, with or without the README's override.

    With it, the seed carries no grants and the agent goes on to create its own sandbox. Without it, the seed carries
    HLE's required direct_http grant, which no session harness can use, so the agent refuses it with a 422.
    """
    config = compose_profile_on_hle(agent, monkeypatch, pairing=pairing)
    seed, reply, created_sandbox = await seed_through_hle(agent, config, monkeypatch)
    assert seed["sandbox_access"] is None
    if pairing:
        assert seed["tool_accesses"] == []
        # The agent accepts the seed and goes on to create its own sandbox.
        assert created_sandbox, reply and reply.text
    else:
        # HLE's judge grants its Resources server over direct HTTP, which no session harness can use.
        assert [(access["kind"], access["name"], access["required"]) for access in seed["tool_accesses"]] == [
            ("direct_http", "hle_equivalence_llm_judge_resources_server.direct_http", True)
        ]
        assert reply is not None and reply.status_code == 422, reply and reply.text
        assert not created_sandbox

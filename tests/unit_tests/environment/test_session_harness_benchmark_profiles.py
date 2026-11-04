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
"""The session harnesses' `configs/profiles/benchmark.yaml`, composed onto HLE as the adapter READMEs describe."""

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
from nemo_gym.config.config_types import BaseServerConfig, ConfigMissingValuesError
from nemo_gym.config.global_config import (
    GlobalConfigDictParser,
    GlobalConfigDictParserConfig,
    get_first_server_config_dict,
)
from nemo_gym.environment.agent_registry import discover_agents
from nemo_gym.server_runtime.base_resources_server import ResourcesSeedSessionResponse


REPO_ROOT = Path(__file__).resolve().parents[3]
HLE_ENVIRONMENT = "hle_equivalence_llm_judge_environment_server"
IMAGE = "registry.example.invalid/sandbox:1"

# Adapter directory name -> (agent class, its config class).
HARNESSES = {
    "pi_agent": ("PiAgent", "PiAgentConfig"),
    "opencode_agent": ("OpenCodeAgent", "OpenCodeAgentConfig"),
    "codex_agent": ("CodexAgent", "CodexAgentConfig"),
    "openclaw_agent": ("OpenClawAgent", "OpenClawAgentConfig"),
    "hermes_agent": ("HermesAgent", "HermesAgentConfig"),
    "claude_code_agent": ("ClaudeCodeAgent", "ClaudeCodeAgentConfig"),
}


def _compose(
    agent: str,
    monkeypatch: pytest.MonkeyPatch,
    *,
    transports: list[str] | None = None,
    image: bool = True,
    cli_args: tuple = (),
) -> DictConfig:
    """Compose the profile onto HLE as shipped, or with `transports` set on HLE's Environment Server."""
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("OPENSANDBOX_API_KEY", "unused")
    composed = f"hle_equivalence_llm_judge_{agent}"
    # Set under the composed agent's name, as the missing-value error for the profile's `image: ???` suggests.
    if image:
        cli_args = (f"++{composed}.responses_api_agents.{agent}.sandbox_config.image={IMAGE}", *cli_args)
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
                    **(
                        {
                            HLE_ENVIRONMENT: {
                                "environment_servers": {
                                    "single_agent_turn_legacy": {"resources_tool_transports": transports}
                                }
                            }
                        }
                        if transports is not None
                        else {}
                    ),
                }
            ),
        )
    )


def _server_config(config: DictConfig, name: str) -> dict[str, Any]:
    return OmegaConf.to_container(get_first_server_config_dict(config, name), resolve=True) | {"name": name}


async def _seed_through_hle(
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
    harness = agent_class(config=config_class.model_validate(_server_config(config, agent_name)), server_client=client)
    environment = SingleAgentTurnLegacyEnvironmentServer(
        config=SingleAgentTurnLegacyEnvironmentServerConfig.model_validate(_server_config(config, HLE_ENVIRONMENT)),
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


@pytest.mark.parametrize("agent", HARNESSES)
def test_profile_is_not_a_standalone_agent_variant(agent: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    assert list(discover_agents()[agent].variants) == [agent]


@pytest.mark.parametrize("agent", HARNESSES)
def test_profile_composes_onto_hle_with_its_limits_and_a_loopback_head_server(
    agent: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _compose(agent, monkeypatch, cli_args=("+default_host=192.0.2.30",))
    composed = f"hle_equivalence_llm_judge_{agent}"
    assert OmegaConf.missing_keys(config) == set()
    assert list(config[composed].responses_api_agents) == [agent]
    settings = get_first_server_config_dict(config, composed)
    assert settings.resources_server.name == "hle_equivalence_llm_judge_resources_server"
    assert settings.sandbox_provider == "sandbox"
    assert settings.sandbox_config.image == IMAGE
    assert settings.sandbox_config.ready_timeout_s == 1200
    # Hermes derives its TTL from its own timeouts; the others would get OpenSandbox's 10-minute default.
    assert settings.sandbox_config.get("ttl_s") == (None if agent == "hermes_agent" else 14400)
    if agent == "claude_code_agent":
        assert settings.claude_code_version == "2.1.108"
        assert settings.model_server.name == "policy_model"
    environment = get_first_server_config_dict(config, HLE_ENVIRONMENT)
    assert environment.agent_server.name == composed
    # HLE's judge serves no tools, so its Environment Server sets no transports and grants none.
    assert list(environment.get("resources_tool_transports", [])) == []
    # Sandboxes reach the servers at the caller's address; the head server's config endpoint stays on loopback.
    assert config.head_server.host == "127.0.0.1"
    assert settings.host == "192.0.2.30"


@pytest.mark.parametrize("agent", HARNESSES)
def test_profile_without_an_image_fails_at_resolve(agent: str, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ConfigMissingValuesError, match=rf"hle_equivalence_llm_judge_{agent}\..*sandbox_config\.image"):
        _compose(agent, monkeypatch, image=False)


@pytest.mark.parametrize("transports", [None, ["direct_http"]], ids=["as_shipped", "direct_http_granted"])
@pytest.mark.parametrize("agent", HARNESSES)
async def test_hle_seed_reaches_the_agents_own_sandbox_unless_a_direct_http_grant_is_required(
    agent: str, transports: list[str] | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    if agent == "hermes_agent":
        pytest.importorskip("model_tools", reason="hermes_agent imports hermes-agent, which its own venv installs")
    config = _compose(agent, monkeypatch, transports=transports)
    seed, reply, created_sandbox = await _seed_through_hle(agent, config, monkeypatch)
    assert seed["sandbox_access"] is None
    if transports is None:
        # HLE's judge serves no tools, so its Environment Server grants none.
        assert seed["tool_accesses"] == []
        # The agent accepts the seed and goes on to create its own sandbox.
        assert created_sandbox, reply and reply.text
    else:
        # A pairing whose Environment Server grants direct HTTP access, which no session harness can use.
        assert [(access["kind"], access["name"], access["required"]) for access in seed["tool_accesses"]] == [
            ("direct_http", "hle_equivalence_llm_judge_resources_server.direct_http", True)
        ]
        assert reply is not None and reply.status_code == 422, reply and reply.text
        assert not created_sandbox

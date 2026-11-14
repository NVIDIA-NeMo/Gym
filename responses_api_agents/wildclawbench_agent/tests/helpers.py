# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Construct an agent with deterministic policy and image metadata for component tests."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf

from nemo_gym.client import ServerClient
from responses_api_agents.wildclawbench_agent.app import WildClawAgent, WildClawConfig


def make_agent(tmp_path: Path, worker_python: str) -> WildClawAgent:
    """Build the real agent while replacing startup policy/image inspection."""
    config = WildClawConfig(
        name="wildclaw_agent",
        host="127.0.0.1",
        port=0,
        entrypoint="app.py",
        source_root=str(tmp_path),
        worker_python=worker_python,
        output_root=str(tmp_path / "episodes"),
        model_name="policy",
        model_server=dict(type="responses_api_models", name="policy_model"),
        sandbox_runtime="docker",
    )
    with (
        patch(
            "responses_api_agents.wildclawbench_agent.app.get_global_config_dict",
            return_value=OmegaConf.create(
                {
                    "policy_model": {
                        "responses_api_models": {
                            "vllm_model": {
                                "entrypoint": "wildclawbench.py",
                                "uses_reasoning_parser": True,
                                "propagate_context_overflow_errors": True,
                            }
                        }
                    }
                }
            ),
        ),
        patch(
            "responses_api_agents.wildclawbench_agent.execution.subprocess.run",
            return_value=SimpleNamespace(stdout="sha256:" + "a" * 64),
        ),
    ):
        agent = WildClawAgent(config=config, server_client=MagicMock(spec=ServerClient))
        agent._image_identity = "sha256:" + "a" * 64
        return agent


def make_pipeline(tmp_path, worker_python, monkeypatch):
    """Exercise the actual Environment/Resources/Agent lifecycle with a local worker fixture.

    Only the native asset validation and HTTP transport are replaced. Requests still pass through the
    production seed/activate/close/verify methods and their ownership records.
    """
    import json
    import socket
    from http.cookies import SimpleCookie
    from types import SimpleNamespace

    from starlette.requests import Request

    from environment_servers.wildclawbench_environment.app import WildClawEnvironment, WildClawEnvironmentConfig
    from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedSessionRequest
    from nemo_gym.client import SESSION_ID_KEY
    from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
    from nemo_gym.server_runtime import BaseVerifyRequest, ResourcesCloseSessionRequest, ResourcesSeedSessionRequest
    from resources_servers.wildclawbench.app import WildClawResources, WildClawResourcesConfig
    from responses_api_agents.wildclawbench_agent.execution import ExecutionConfig, episode_output

    agent = make_agent(tmp_path, worker_python)
    monkeypatch.setattr(
        WildClawAgent, "resolve_model_base_url", lambda self, name, rollout: f"http://localhost/rollouts/{rollout}/v1"
    )
    execution = ExecutionConfig.model_validate(agent.config.model_dump())
    with patch("resources_servers.wildclawbench.app.image_identity", return_value=agent._image_identity):
        resources = WildClawResources(
            config=WildClawResourcesConfig(
                name="wildclaw_resources", host="127.0.0.1", port=1, entrypoint="app.py", execution=execution
            ),
            server_client=agent.server_client,
        )

    def lease(config, key, metadata, prompt, episode, task):
        output = episode_output(config, key)
        output.mkdir(parents=True, exist_ok=True)
        (output / "lease.json").write_text(
            json.dumps(
                dict(
                    execution=config.model_dump(mode="json"),
                    owner_host=socket.gethostname(),
                    metadata=metadata,
                    prompt=prompt,
                    episode_id=episode,
                    task_id=task,
                )
            )
        )
        return output

    monkeypatch.setattr("resources_servers.wildclawbench.app.create_lease", lease)
    client = agent.server_client
    calls = []
    jars = {}

    async def post(server_name, url_path, **kwargs):
        calls.append(url_path)
        body = kwargs["json"]
        if hasattr(body, "model_dump"):
            body = body.model_dump(mode="json")
        jar = jars.setdefault(server_name, {})
        request = Request(
            dict(
                type="http",
                headers=[],
                session=jar,
                path_params={"rollout_id": url_path.split("/")[2] if url_path.startswith("/ng-rollout/") else None},
            )
        )
        if url_path == "/seed_session":
            jar.setdefault(SESSION_ID_KEY, "initial-cookie")
            result = await resources.seed_session(ResourcesSeedSessionRequest.model_validate(body), request=request)
        elif url_path == "/v1/agent_sessions":
            result = await agent.seed_agent_session(request, AgentSeedSessionRequest.model_validate(body))
        elif url_path == "/v1/agent_sessions/close":
            result = await agent.close_agent_session(request, AgentCloseSessionRequest.model_validate(body))
        elif url_path == "/verify":
            result = await resources.verify(request, BaseVerifyRequest.model_validate(body))
        elif url_path == "/close_session":
            result = await resources.close_resources_session(ResourcesCloseSessionRequest.model_validate(body))
        else:
            result = await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming.model_validate(body))
        cookies = SimpleCookie()
        cookies["session"] = "fixture-cookie"

        async def read():
            return json.dumps(result.model_dump(mode="json")).encode()

        # Fresh HTTP cookies are per episode, as each real environment handles its own signed jars.
        if url_path == "/close_session":
            jars.clear()
        return SimpleNamespace(
            read=read, ok=True, status=200, cookies=cookies, headers={"content-type": "application/json"}
        )

    client.post.side_effect = post
    client._resolve_base_url.return_value = "http://local:8000"
    # Parent consults the bound agent config to build its activation URL.
    client.global_config_dict = OmegaConf.create({
        "wildclaw_agent": {"responses_api_agents": {"wildclawbench_agent": {"entrypoint": "app.py"}}},
        "wildclaw_resources": {"resources_servers": {"wildclawbench": {"entrypoint": "app.py", "execution": execution.model_dump(mode="json")}}},
    })
    environment = WildClawEnvironment(
        config=WildClawEnvironmentConfig(
            name="wildclaw_environment_server",
            host="127.0.0.1",
            port=2,
            entrypoint="app.py",
            agent_server={"type": "responses_api_agents", "name": "wildclaw_agent"},
            resources_server={"type": "resources_servers", "name": "wildclaw_resources"},
            tool_servers=[],
            output_root=execution.output_root,
            max_concurrent_episodes=1,
            queue_timeout_seconds=10,
            default_episode_timeout_seconds=10,
            cleanup_timeout_seconds=10,
        ),
        server_client=client,
    )
    return environment, agent, resources, calls

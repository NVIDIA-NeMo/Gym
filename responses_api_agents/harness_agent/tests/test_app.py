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

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec
from nemo_gym.sandbox.providers.local import LocalProvider
from nemo_gym.server_utils import ServerClient
from responses_api_agents.harness_agent.app import (
    _RUN_CONTEXT,
    HarnessAgent,
    HarnessAgentConfig,
    HarnessAgentRunRequest,
    stage_and_run_eval,
)


def _config(**kwargs) -> HarnessAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="sbx",
        resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        agent="codex",
        agent_kwargs={
            "model": {
                "model": "gym-policy-model",
                "base_url": "__SANDBOX_MODEL_URL__/v1",
            }
        },
        sandbox_provider={"opensandbox": {}},
    )
    base.update(kwargs)
    return HarnessAgentConfig(**base)


def _make_agent(**cfg_kwargs) -> HarnessAgent:
    # skip provider creation and gym tar build (both side effects) during construction
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=MagicMock()),
        patch.object(HarnessAgent, "_build_gym_tar", return_value=None),
    ):
        return HarnessAgent(config=_config(**cfg_kwargs), server_client=MagicMock(spec=ServerClient))


def test_config_defaults():
    cfg = _config()
    assert cfg.sandbox_image == "python:3.12-slim"
    assert cfg.sandbox_python == "python3"


def test_named_sandbox_provider_is_resolved_with_metadata():
    provider = MagicMock()
    global_config = {
        "sandbox": {
            "default_metadata": {"cluster": "cell3"},
            "opensandbox": {"connection": {"domain": "sandbox.example"}},
        }
    }
    with (
        patch("responses_api_agents.harness_agent.app.get_global_config_dict", return_value=global_config),
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=provider) as create,
        patch.object(HarnessAgent, "_build_gym_tar", return_value=None),
    ):
        agent = HarnessAgent(config=_config(sandbox_provider="sandbox"), server_client=MagicMock(spec=ServerClient))

    create.assert_called_once_with({"opensandbox": {"connection": {"domain": "sandbox.example"}}})
    assert agent._sandbox_default_metadata == {"cluster": "cell3"}


def test_runner_config_carries_agent_name():
    agent = _make_agent(agent="codex", sandbox_python="/deps/bin/python3")
    script, runner_config, cmd = agent._runner()
    assert runner_config["agent"] == "codex"
    assert "runner_config.json" in script
    assert '"model_base_url": harness.config.model.base_url' in script
    assert 'rc["agent"] == "claude_code"' in script
    assert 'rc["agent"] == "codex"' in script
    compile(script, "<agent_runner>", "exec")
    assert cmd == "/deps/bin/python3 runner.py"


async def test_local_provider_provisions_in_its_workspace(tmp_path):
    server = await asyncio.start_server(lambda *_: None, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    agent = _make_agent()
    agent._provider = LocalProvider(workspace_root=str(tmp_path))
    agent._gym_tar = None

    try:
        handle = await agent._provision_box("", {"/work/request.json": "{}"}, f"http://127.0.0.1:{port}")
        assert (handle.raw["workspace"] / "work" / "request.json").read_text() == "{}"
        await agent._close_box(handle)
        assert not handle.raw["workspace"].exists()
    finally:
        server.close()
        await server.wait_closed()


async def test_local_provider_stages_skills_in_its_workspace(tmp_path):
    server = await asyncio.start_server(lambda *_: None, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    skills = tmp_path / "source-skills"
    skills.mkdir()
    (skills / "SKILL.md").write_text("skill")
    agent = _make_agent()
    agent._provider = LocalProvider(workspace_root=str(tmp_path / "sandboxes"))
    agent._gym_tar = None
    try:
        handle = await agent._provision_box("", {}, f"http://127.0.0.1:{port}", str(skills))
        assert (handle.raw["workspace"] / "work" / "skills" / "SKILL.md").read_text() == "skill"
        await agent._close_box(handle)
    finally:
        server.close()
        await server.wait_closed()


def test_sandbox_model_url_preserves_remote_hostname_and_port():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client._build_server_base_url = MagicMock(return_value="http://model-host:8000")
    agent.server_client.global_config_dict = MagicMock()
    with (
        patch("responses_api_agents.harness_agent.app.get_first_server_config_dict", return_value={}),
        patch("responses_api_agents.harness_agent.app.socket.gethostbyname") as resolve,
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://model-host:8000"
    resolve.assert_not_called()


def test_sandbox_model_url_prefers_backend_base_url_and_strips_v1():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client.global_config_dict = MagicMock()
    with (
        patch(
            "responses_api_agents.harness_agent.app.get_first_server_config_dict",
            return_value={"base_url": "http://vllm-node:9000/v1"},
        ),
        patch("responses_api_agents.harness_agent.app.socket.gethostbyname") as resolve,
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://vllm-node:9000"
    resolve.assert_not_called()


def test_sandbox_model_url_keeps_loopback_on_dns_failure():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client._build_server_base_url = MagicMock(return_value="http://localhost:8000")
    agent.server_client.global_config_dict = MagicMock()
    with (
        patch("responses_api_agents.harness_agent.app.get_first_server_config_dict", return_value={}),
        patch("responses_api_agents.harness_agent.app.socket.gethostbyname", side_effect=OSError),
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://localhost:8000"


def test_sandbox_model_url_uses_direct_harness_endpoint_without_model_server():
    agent = _make_agent(
        model_server=None,
        agent_kwargs={"model": {"model": "direct", "base_url": "https://provider.example/v1"}},
    )

    assert agent._sandbox_model_url(MagicMock()) == "https://provider.example"


def test_sandbox_model_url_preserves_training_rollout_prefix():
    agent = _make_agent()
    agent.server_client.global_config_dict = MagicMock()
    request = SimpleNamespace(path_params={}, url=SimpleNamespace(path="/run"))
    token = _RUN_CONTEXT.set({"url_prefix": "/ng-rollout/rollout-1/training-token-capture"})
    try:
        with patch(
            "responses_api_agents.harness_agent.app.get_first_server_config_dict",
            return_value={"base_url": "http://model:8000/v1"},
        ):
            url = agent._sandbox_model_url(request)
    finally:
        _RUN_CONTEXT.reset(token)
    assert url == "http://model:8000/ng-rollout/rollout-1/training-token-capture"


async def test_run_forwards_mcp_skills_rollout_and_verifier_fields(tmp_path):
    agent = _make_agent()
    agent.server_client.post = AsyncMock(side_effect=[MagicMock(cookies={}), MagicMock(cookies={})])
    agent.server_client._build_server_base_url = MagicMock(return_value="http://resources:8001")
    agent.server_client.global_config_dict = {}
    body = HarnessAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hello"),
        _ng_rollout_id="rollout-1",
        skills_ref={"path": str(tmp_path)},
    )
    response = {
        "id": "response-1",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }
    verify = body.model_dump() | {"response": response, "reward": 1.0, "custom_metric": 7}
    seen = {}

    async def responses(_, __, ___):
        seen.update(_RUN_CONTEXT.get())
        return NeMoGymResponse.model_validate(response)

    with (
        patch.object(HarnessAgent, "responses", new=responses),
        patch.object(HarnessAgent, "url_path_for_run", return_value="/ng-rollout/rollout-1") as path,
        patch("responses_api_agents.harness_agent.app.raise_for_status", new=AsyncMock()),
        patch(
            "responses_api_agents.harness_agent.app.get_first_server_config_dict",
            return_value={"host": "resources", "port": 8001},
        ),
        patch(
            "responses_api_agents.harness_agent.app.get_response_json",
            new=AsyncMock(
                side_effect=[
                    {
                        "mcp": {
                            "server_name": "tools",
                            "url_path": "/mcp",
                            "headers": {"Authorization": "Bearer session"},
                        }
                    },
                    verify,
                ]
            ),
        ),
    ):
        result = await agent.run(MagicMock(cookies={}), body)
    path.assert_called_once_with("", body)
    assert seen["url_prefix"] == "/ng-rollout/rollout-1"
    assert seen["skills_path"] == str(tmp_path)
    assert seen["mcp"] == {
        "server_name": "tools",
        "url": "http://resources:8001/mcp",
        "transport": "http",
        "headers": {"Authorization": "Bearer session"},
    }
    assert result.custom_metric == 7


async def test_grading_command_failure_is_not_reward_zero():
    agent = _make_agent()
    agent._provider.exec = AsyncMock(return_value=SandboxExecResult("", "grader crashed", 2))
    agent._provider.download_file = AsyncMock()

    with pytest.raises(RuntimeError, match="eval command failed.*grader crashed"):
        await agent._grade_in_box(MagicMock(), {"eval_command": "false"})
    agent._provider.download_file.assert_not_awaited()


async def test_empty_reward_file_is_not_reward_zero():
    provider = MagicMock()
    provider.exec = AsyncMock(return_value=SandboxExecResult("", "", 0))
    provider.download_file = AsyncMock(side_effect=lambda _, __, path: path.write_text(""))

    with pytest.raises(RuntimeError, match="reward file.*is empty"):
        await stage_and_run_eval(provider, MagicMock(), {}, "true", "/reward", 30)


async def test_local_grading_stays_in_workspace(tmp_path):
    provider = LocalProvider(workspace_root=str(tmp_path))
    handle = await provider.create(SandboxSpec(image=""))
    try:
        reward = await stage_and_run_eval(
            provider,
            handle,
            {"/tests/test.sh": "true"},
            "bash /tests/test.sh && mkdir -p /logs/verifier && echo 1 > /logs/verifier/reward.txt",
            "/logs/verifier/reward.txt",
            30,
        )
        assert reward == 1.0
        assert (handle.raw["workspace"] / "logs" / "verifier" / "reward.txt").is_file()
    finally:
        await provider.close(handle)


async def test_setup_failure_closes_sandbox():
    agent = _make_agent(setup_commands=["install deps"])
    handle = MagicMock()
    agent._provider.create = AsyncMock(return_value=handle)
    agent._provider.exec = AsyncMock(
        side_effect=[SandboxExecResult("", "", 0), SandboxExecResult("", "install failed", 2)]
    )
    agent._provider.close = AsyncMock()

    with pytest.raises(RuntimeError, match="setup failed.*install failed"):
        await agent._provision_box("image", {}, "https://model.example")
    agent._provider.close.assert_awaited_once_with(handle)


async def test_agent_runner_failure_reports_its_log():
    agent = _make_agent()
    handle = MagicMock()
    agent._provision_box = AsyncMock(return_value=handle)
    agent._provider.exec = AsyncMock(
        side_effect=[
            SandboxExecResult("", "exit status 1", 1),
            SandboxExecResult("runner traceback", "", 0),
        ]
    )
    agent._provider.close = AsyncMock()
    agent._download_json = AsyncMock()
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")

    with (
        patch.object(agent, "_sandbox_model_url", return_value="https://model.example"),
        pytest.raises(RuntimeError, match="runner failed.*runner traceback"),
    ):
        await agent.responses(MagicMock(), body)

    agent._download_json.assert_not_awaited()
    agent._provider.close.assert_awaited_once_with(handle)


async def test_download_json_requires_exactly_one_row():
    agent = _make_agent()
    agent._provider.download_file = AsyncMock(side_effect=lambda _, __, path: path.write_text("{}\n{}\n"))

    with pytest.raises(RuntimeError, match="expected one JSON row.*got 2"):
        await agent._download_json(MagicMock(), "/work/rollouts.jsonl")


def test_gym_tar_built_on_init():
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=MagicMock()),
        patch.object(HarnessAgent, "_build_gym_tar", return_value="/tmp/fake.tar.gz"),
    ):
        agent = HarnessAgent(config=_config(), server_client=MagicMock(spec=ServerClient))
        assert agent._gym_tar == "/tmp/fake.tar.gz"


def test_gym_source_prebuilt_path_and_url():
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=MagicMock()),
        patch.object(HarnessAgent, "_build_gym_tar") as build,
    ):
        prebuilt = HarnessAgent(
            config=_config(gym_source="/tmp/prebuilt.tar.gz"), server_client=MagicMock(spec=ServerClient)
        )
        assert str(prebuilt._gym_tar) == "/tmp/prebuilt.tar.gz"
        assert prebuilt._gym_source_url is None
        remote = HarnessAgent(
            config=_config(gym_source="https://example.com/gym.tar.gz"), server_client=MagicMock(spec=ServerClient)
        )
        assert remote._gym_tar is None
        assert remote._gym_source_url == "https://example.com/gym.tar.gz"
        build.assert_not_called()

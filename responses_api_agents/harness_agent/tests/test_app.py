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
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.base import ConnectableProvider, SandboxExecResult, SandboxSpec
from nemo_gym.sandbox.providers.local import LocalProvider
from nemo_gym.server_utils import ServerClient
from responses_api_agents.harness_agent.app import (
    _RUN_CONTEXT,
    HarnessAgent,
    HarnessAgentConfig,
    HarnessAgentRunRequest,
    resolve_agent,
    stage_and_run_eval,
)


def _response() -> dict:
    return {
        "id": "response-1",
        "created_at": 0,
        "model": "model",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "none",
        "tools": [],
    }


def test_runner_prefers_staged_agent_after_gym_import(tmp_path):
    root = Path(__file__).resolve().parents[3]
    mount = tmp_path / "gym_mount"
    mount.mkdir()
    (mount / "nemo_gym").symlink_to(root / "nemo_gym", target_is_directory=True)
    installed = tmp_path / "installed"
    for directory, value in [(mount, "staged"), (installed, "installed")]:
        package = directory / "responses_api_agents"
        package.mkdir(parents=True)
        (package / "runner_probe.py").write_text(f"source = {value!r}\n")
    runner = tmp_path / "agent_runner.py"
    runner.write_text((root / "responses_api_agents/harness_agent/agent_runner.py").read_text())
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys, runpy\nsys.path.insert(0, {str(installed)!r})\n"
            f"runpy.run_path({str(runner)!r})\n"
            "from responses_api_agents.runner_probe import source\nassert source == 'staged'",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def _config(**kwargs) -> HarnessAgentConfig:
    base = dict(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="sbx",
        resources_server=ResourcesServerRef(type="resources_servers", name="rs"),
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        agent="opencode",
        sandbox_provider={"opensandbox": {}},
    )
    base.update(kwargs)
    return HarnessAgentConfig(**base)


def _make_agent(**cfg_kwargs) -> HarnessAgent:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    provider = MagicMock(spec=ConnectableProvider)
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=provider),
        patch.object(HarnessAgent, "_build_gym_tar", return_value=None),
    ):
        return HarnessAgent(config=_config(**cfg_kwargs), server_client=server_client)


def test_config_defaults():
    cfg = _config()
    assert cfg.sandbox_image == "python:3.12-slim"
    assert cfg.sandbox_python == "python3"


def test_registry_exposes_all_harness_agents():
    assert resolve_agent("opencode") == (
        "responses_api_agents.opencode_agent.app",
        "OpenCodeAgent",
        "OpenCodeAgentConfig",
    )
    for name in ("deepagents", "mini_swe"):
        assert resolve_agent(name) == (
            "responses_api_agents.nemo_fabric_agent.app",
            "NeMoFabricAgent",
            "NeMoFabricAgentConfig",
        )


def test_unknown_agent_is_rejected():
    with pytest.raises(ValueError, match="Unknown agent: unknown"):
        resolve_agent("unknown")


def test_named_sandbox_provider_is_resolved_with_metadata():
    provider = MagicMock()
    global_config = {
        "sandbox": {
            "default_metadata": {"cluster": "cell3"},
            "opensandbox": {"connection": {"domain": "sandbox.example"}},
        }
    }
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = global_config
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=provider) as create,
        patch.object(HarnessAgent, "_build_gym_tar", return_value=None),
    ):
        agent = HarnessAgent(config=_config(sandbox_provider="sandbox"), server_client=server_client)
    create.assert_called_once_with({"opensandbox": {"connection": {"domain": "sandbox.example"}}})
    assert agent._sandbox_metadata == {"cluster": "cell3"}


def test_sandbox_model_url_preserves_remote_hostname_and_port():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client._build_server_base_url = MagicMock(return_value="http://model-host:8000")
    agent.server_client.global_config_dict = MagicMock()
    with patch("responses_api_agents.harness_agent.app.get_first_server_config_dict", return_value={}):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://model-host:8000"


def test_sandbox_model_url_uses_gym_proxy_instead_of_backend():
    agent = _make_agent(model_server={"type": "responses_api_models", "name": "policy_model"})
    agent.server_client.global_config_dict = MagicMock()
    agent.server_client._build_server_base_url = MagicMock(return_value="http://gym-proxy:8000")
    with patch(
        "responses_api_agents.harness_agent.app.get_first_server_config_dict",
        return_value={"base_url": "http://vllm-node:9000/v1"},
    ):
        url = agent._sandbox_model_url(MagicMock())
    assert url == "http://gym-proxy:8000"


@pytest.mark.parametrize("hostname", ["sandbox-gateway", "localhost"])
def test_sandbox_model_url_uses_explicit_override(hostname):
    agent = _make_agent(sandbox_model_base_url=f"http://{hostname}:8000/v1")
    token = _RUN_CONTEXT.set({"url_prefix": "/ng-rollout/rollout-1"})
    try:
        with (
            patch("responses_api_agents.harness_agent.app.get_first_server_config_dict") as get_config,
            patch("responses_api_agents.harness_agent.app.socket.gethostbyname") as resolve,
        ):
            url = agent._sandbox_model_url(MagicMock())
    finally:
        _RUN_CONTEXT.reset(token)
    assert url == f"http://{hostname}:8000/ng-rollout/rollout-1"
    get_config.assert_not_called()
    resolve.assert_not_called()


def test_sandbox_model_url_preserves_training_rollout_prefix():
    agent = _make_agent()
    agent.server_client._build_server_base_url = MagicMock(return_value="http://model:8000")
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


async def test_run_preserves_rollout_path_and_verifier_fields():
    agent = _make_agent()
    agent.server_client.post = AsyncMock(side_effect=[MagicMock(cookies={}), MagicMock(cookies={})])
    body = HarnessAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hello"),
        _ng_rollout_id="rollout-1",
    )
    response = _response()
    verify = body.model_dump() | {"response": response, "reward": 1.0, "custom_metric": 7}
    seen = {}

    async def responses(_, __, ___):
        seen.update(_RUN_CONTEXT.get())
        return NeMoGymResponse.model_validate(response)

    with (
        patch.object(HarnessAgent, "responses", new=responses),
        patch.object(
            HarnessAgent,
            "url_path_for_run",
            return_value="/ng-rollout/rollout-1/training-token-capture",
        ) as path,
        patch("responses_api_agents.harness_agent.app.raise_for_status", new=AsyncMock()),
        patch(
            "responses_api_agents.harness_agent.app.get_response_json",
            new=AsyncMock(side_effect=[{"sandbox_handle": "box-1"}, verify]),
        ),
    ):
        result = await agent.run(MagicMock(cookies={}), body)
    path.assert_called_once_with("", body)
    assert seen.pop("rollout_id")
    assert seen == {
        "sandbox_descriptor": {"sandbox_id": "box-1"},
        "url_prefix": "/ng-rollout/rollout-1/training-token-capture",
    }
    assert result.custom_metric == 7


async def test_verify_failure_closes_resource_sandbox():
    agent = _make_agent()
    agent.server_client.post = AsyncMock(side_effect=[MagicMock(cookies={}), MagicMock(cookies={})])
    agent._provider.close = AsyncMock()
    body = HarnessAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hello"),
    )
    response = _response()
    handle = MagicMock(provider_name="opensandbox")
    agent._provider.connect = AsyncMock(return_value=handle)

    async def responses(_, __, ___):
        return NeMoGymResponse.model_validate(response)

    with (
        patch.object(HarnessAgent, "responses", new=responses),
        patch(
            "responses_api_agents.harness_agent.app.raise_for_status",
            new=AsyncMock(side_effect=[None, RuntimeError("verify failed")]),
        ),
        patch(
            "responses_api_agents.harness_agent.app.get_response_json",
            new=AsyncMock(return_value={"sandbox_handle": "box-1"}),
        ),
    ):
        with pytest.raises(RuntimeError, match="verify failed"):
            await agent.run(MagicMock(cookies={}), body)
    agent._provider.close.assert_awaited_once_with(handle)


@pytest.mark.parametrize("exception_type", [RuntimeError, asyncio.CancelledError])
async def test_pre_provision_failure_closes_resource_sandbox(exception_type):
    agent = _make_agent()
    agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
    handle = MagicMock(provider_name="opensandbox")
    agent._provider.connect = AsyncMock(return_value=handle)
    agent._provider.close = AsyncMock()
    responses = AsyncMock(side_effect=exception_type())
    body = HarnessAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hello"),
    )
    with (
        patch.object(HarnessAgent, "responses", new=responses),
        patch("responses_api_agents.harness_agent.app.raise_for_status", new=AsyncMock()),
        patch(
            "responses_api_agents.harness_agent.app.get_response_json",
            new=AsyncMock(return_value={"sandbox_handle": "box-1"}),
        ),
        pytest.raises(exception_type),
    ):
        await agent.run(MagicMock(cookies={}), body)
    agent._provider.connect.assert_awaited_once_with({"sandbox_id": "box-1"})
    agent._provider.close.assert_awaited_once_with(handle)


async def test_response_serialization_failure_closes_resource_sandbox():
    agent = _make_agent()
    agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
    handle = MagicMock(provider_name="opensandbox")
    agent._provider.connect = AsyncMock(return_value=handle)
    agent._provider.close = AsyncMock()
    response = MagicMock()
    response.model_dump.side_effect = RuntimeError("serialize failed")
    responses = AsyncMock(return_value=response)
    body = HarnessAgentRunRequest(
        responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hello"),
    )
    with (
        patch.object(HarnessAgent, "responses", new=responses),
        patch("responses_api_agents.harness_agent.app.raise_for_status", new=AsyncMock()),
        patch(
            "responses_api_agents.harness_agent.app.get_response_json",
            new=AsyncMock(return_value={"sandbox_handle": "box-1"}),
        ),
        pytest.raises(RuntimeError, match="serialize failed"),
    ):
        await agent.run(MagicMock(cookies={}), body)
    agent._provider.connect.assert_awaited_once_with({"sandbox_id": "box-1"})
    agent._provider.close.assert_awaited_once_with(handle)


async def test_resource_sandbox_survives_until_verification():
    agent = _make_agent(agent_kwargs={"repo_dir": "/testbed"})
    agent._gym_tar = None
    handle = MagicMock(provider_name="opensandbox")
    agent._provider.connect = AsyncMock(return_value=handle)
    agent._provider.create = AsyncMock()
    runner = SandboxExecResult("", "", 0)
    logs = SandboxExecResult("RUNNER_DONE", "", 0)
    agent._provider.exec = AsyncMock(
        side_effect=[SandboxExecResult("", "", 0), SandboxExecResult("/testbed\n", "", 0), runner, logs]
    )
    uploaded = {}

    async def upload(_, source, target):
        uploaded[target] = source.read_text()

    agent._provider.upload_file = AsyncMock(side_effect=upload)
    agent._provider.close = AsyncMock()

    response = _response()
    agent._provider.download_file = AsyncMock(side_effect=lambda _, __, path: path.write_text(json.dumps(response)))
    token = _RUN_CONTEXT.set({"sandbox_descriptor": {"sandbox_id": "box-1"}})
    try:
        with patch.object(agent, "_sandbox_model_url", return_value="https://model.example"):
            await agent.responses(MagicMock(), NeMoGymResponseCreateParamsNonStreaming(input="hello"))
            agent._provider.close.assert_not_awaited()
    finally:
        _RUN_CONTEXT.reset(token)
    agent._provider.connect.assert_awaited_once_with({"sandbox_id": "box-1"})
    agent._provider.create.assert_not_awaited()
    assert json.loads(uploaded["/work/runner_config.json"])["cwd"] == "/testbed"
    assert json.loads(uploaded["/work/agent_config.json"])["repo_dir"] == "/testbed"


async def test_resource_sandbox_requires_connectable_provider(tmp_path):
    agent = _make_agent()
    agent._provider = LocalProvider(workspace_root=str(tmp_path))
    token = _RUN_CONTEXT.set({"sandbox_descriptor": {"sandbox_id": "box-1"}})
    try:
        with pytest.raises(TypeError, match="cannot connect"):
            await agent._provision_box("image", {}, "https://model.example")
    finally:
        _RUN_CONTEXT.reset(token)


async def test_cancelled_setup_closes_owned_sandbox():
    agent = _make_agent()
    handle = MagicMock(provider_name="opensandbox")
    agent._provider.create = AsyncMock(return_value=handle)
    agent._provider.exec = AsyncMock(side_effect=asyncio.CancelledError())
    agent._provider.close = AsyncMock()
    with pytest.raises(asyncio.CancelledError):
        await agent._provision_box("image", {}, "https://model.example")
    agent._provider.close.assert_awaited_once_with(handle)


async def test_grading_command_failure_is_not_reward_zero():
    agent = _make_agent()
    agent._provider.exec = AsyncMock(return_value=SandboxExecResult("", "grader crashed", 2))
    agent._provider.download_file = AsyncMock()

    with pytest.raises(RuntimeError, match="eval command failed.*grader crashed"):
        await stage_and_run_eval(agent._provider, MagicMock(), {}, "false", "/reward", 30)
    agent._provider.download_file.assert_not_awaited()


async def test_empty_reward_file_is_not_reward_zero():
    provider = MagicMock()
    provider.exec = AsyncMock(return_value=SandboxExecResult("", "", 0))
    provider.download_file = AsyncMock(side_effect=lambda _, __, path: path.write_text(""))

    with pytest.raises(RuntimeError, match="reward file.*is empty"):
        await stage_and_run_eval(provider, MagicMock(), {}, "true", "/reward", 30)


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


async def test_local_provider_provisions_in_its_workspace(tmp_path):
    server = await asyncio.start_server(lambda *_: None, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    agent = _make_agent(sandbox_spec={"files": {"/work/environment.txt": "Network disabled."}})
    agent._provider = LocalProvider(workspace_root=str(tmp_path))
    agent._gym_tar = None
    try:
        handle = await agent._provision_box("", {"/work/request.json": "{}"}, f"http://127.0.0.1:{port}")
        assert (handle.raw["workspace"] / "work" / "request.json").read_text() == "{}"
        assert (handle.raw["workspace"] / "work" / "environment.txt").read_text() == "Network disabled."
        await agent._close_box(handle)
        assert not handle.raw["workspace"].exists()
    finally:
        server.close()
        await server.wait_closed()


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


async def test_agent_runner_failure_reports_its_log():
    agent = _make_agent(agent="opencode")
    handle = MagicMock(provider_name="opensandbox")
    agent._provision_box = AsyncMock(return_value=handle)
    agent._provider.exec = AsyncMock(
        side_effect=[SandboxExecResult("", "exit status 1", 1), SandboxExecResult("runner traceback", "", 0)]
    )
    agent._provider.close = AsyncMock()
    body = NeMoGymResponseCreateParamsNonStreaming(input="hello")
    with (
        patch.object(agent, "_sandbox_model_url", return_value="https://model.example"),
        pytest.raises(RuntimeError, match="runner failed.*runner traceback"),
    ):
        await agent.responses(MagicMock(), body)
    files = agent._provision_box.await_args.args[1]
    config = json.loads(files["/work/agent_config.json"])
    assert config["resources_server"]["name"] == "rs"
    assert config["model_server"]["name"] == "model"
    assert "adapter_id" not in config


def test_gym_tar_built_on_init():
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=MagicMock()),
        patch.object(HarnessAgent, "_build_gym_tar", return_value="/tmp/fake.tar.gz"),
    ):
        agent = HarnessAgent(config=_config(), server_client=server_client)
        assert agent._gym_tar == "/tmp/fake.tar.gz"


def test_runner_marker_survives_redirected_stdout(tmp_path):
    root = Path(__file__).resolve().parents[3]
    mount = tmp_path / "gym_mount"
    mount.mkdir()
    (mount / "nemo_gym").symlink_to(root / "nemo_gym", target_is_directory=True)
    (mount / "stub_agent.py").write_text(
        "import io, sys\nfrom types import SimpleNamespace\n"
        "Config = SimpleNamespace\n"
        "class Agent:\n"
        "    def __init__(self, **kwargs): pass\n"
        "    async def responses(self, request, params):\n"
        "        sys.stdout = io.StringIO()\n"
        "        return SimpleNamespace(model_dump_json=lambda: '{}')\n"
    )
    for name, body in {
        "runner_config.json": {"agent_module": "stub_agent", "agent_class": "Agent", "agent_config_class": "Config"},
        "agent_config.json": {},
        "request.json": {"input": "hello"},
    }.items():
        (tmp_path / name).write_text(json.dumps(body))
    (tmp_path / "model_url.txt").write_text("http://127.0.0.1:8000/v1")
    runner = tmp_path / "agent_runner.py"
    runner.write_text((root / "responses_api_agents/harness_agent/agent_runner.py").read_text())
    result = subprocess.run([sys.executable, str(runner)], capture_output=True, text=True, check=True)
    assert "RUNNER_DONE" in result.stdout
    assert json.loads((tmp_path / "response.json").read_text()) == {}


def test_runner_config_carries_agent_symbols():
    agent = _make_agent(
        agent="opencode",
        sandbox_python="/deps/bin/python3",
    )
    script, runner_config, cmd = agent._runner()
    assert runner_config["agent_module"] == "responses_api_agents.opencode_agent.app"
    assert runner_config["agent_class"] == "OpenCodeAgent"
    assert runner_config["agent_config_class"] == "OpenCodeAgentConfig"
    assert "runner_config.json" in script
    compile(script, "<agent_runner>", "exec")
    assert cmd == "/deps/bin/python3 runner.py"


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


async def test_download_json_requires_exactly_one_row():
    agent = _make_agent()
    agent._provider.download_file = AsyncMock(side_effect=lambda _, __, path: path.write_text("{}\n{}\n"))

    with pytest.raises(RuntimeError, match="expected one JSON row.*got 2"):
        await agent._download_json(MagicMock(), "/work/rollouts.jsonl")


def test_gym_source_prebuilt_path_and_url():
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    with (
        patch("responses_api_agents.harness_agent.app.create_provider", return_value=MagicMock()),
        patch.object(HarnessAgent, "_build_gym_tar") as build,
    ):
        prebuilt = HarnessAgent(config=_config(gym_source="/tmp/prebuilt.tar.gz"), server_client=server_client)
        assert str(prebuilt._gym_tar) == "/tmp/prebuilt.tar.gz"
        assert prebuilt._gym_source_url is None
        remote = HarnessAgent(config=_config(gym_source="https://example.com/gym.tar.gz"), server_client=server_client)
        assert remote._gym_tar is None
        assert remote._gym_source_url == "https://example.com/gym.tar.gz"
        build.assert_not_called()


@pytest.mark.parametrize("target", ["http://127.0.0.1:8000", "http://[::1]:8000", "http://localhost:8000"])
async def test_restricted_network_rejects_loopback_before_creation(target):
    agent = _make_agent(network_access="model_only")
    agent._provider.name = "opensandbox"
    agent._provider.create = AsyncMock()
    with pytest.raises(ValueError, match="sandbox-reachable"):
        await agent._provision_box("image", {}, target)
    agent._provider.create.assert_not_awaited()


async def test_restricted_network_overrides_permissive_policy_without_mutating_config():
    spec = {
        "provider_options": {"network_policy": {"defaultAction": "allow"}},
        "resources": {"cpu": 2, "memory_mib": 8192},
    }
    agent = _make_agent(network_access="model_only", sandbox_spec=spec)
    agent._provider.name = "opensandbox"
    handle = MagicMock(provider_name="opensandbox")
    agent._provider.create = AsyncMock(return_value=handle)
    agent._provider.exec = AsyncMock(return_value=SandboxExecResult("", "", 0))
    agent._provider.upload_file = AsyncMock()
    await agent._provision_box("science-image", {}, "https://gym-proxy.example")
    created = agent._provider.create.await_args.args[0]
    assert created.provider_options["network_policy"] == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "gym-proxy.example"}],
    }
    assert created.resources.memory_mib == 8192
    assert agent.config.sandbox_spec == spec


async def test_restricted_network_rejects_unverifiable_resource_sandbox():
    agent = _make_agent(network_access="model_only")
    agent._provider.connect = AsyncMock()
    token = _RUN_CONTEXT.set({"sandbox_descriptor": {"sandbox_id": "box"}})
    try:
        with pytest.raises(ValueError, match="Cannot verify"):
            await agent._provision_box("image", {}, "https://gym-proxy.example")
    finally:
        _RUN_CONTEXT.reset(token)
    agent._provider.connect.assert_not_awaited()


async def test_restricted_network_rejects_provider_without_enforcement():
    agent = _make_agent(network_access="model_only")
    agent._provider.name = "local"
    with pytest.raises(ValueError, match="network-policy provider"):
        await agent._provision_box("image", {}, "https://gym-proxy.example")


@pytest.mark.parametrize("status,failed", [("completed", False), ("failed", True), ("incomplete", True)])
async def test_generation_preserves_observations_and_receipt_before_judging(tmp_path, status, failed):
    from hashlib import sha256

    from nemo_gym.rollout_observability import AgentInvocation, AgentObservationBundle, ObservationGap

    agent = _make_agent(artifacts_dir=str(tmp_path), execution_failure_reward_zero=True)
    handle = MagicMock(provider_name="opensandbox", sandbox_id="box-test")
    agent._provision_box = AsyncMock(return_value=handle)
    agent._provider.exec = AsyncMock(
        side_effect=[SandboxExecResult("", "", 0), SandboxExecResult("RUNNER_DONE", "", 0)]
    )
    agent._provider.close = AsyncMock()
    response = _response() | {
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "id": "answer",
                "content": [{"type": "output_text", "text": "42", "annotations": []}],
            }
        ]
    }
    response["_ng_agent_observations"] = AgentObservationBundle(
        source="pi",
        records=[AgentInvocation(invocation_id="rollout-id", status=status)],
        gaps=[ObservationGap(code="no_sandbox_runtime")],
    ).model_dump(mode="json")

    async def download(_, remote, local):
        local.write_text("runner logs" if remote.endswith("runner.out") else json.dumps(response))

    agent._provider.download_file = AsyncMock(side_effect=download)
    # Deliberately path-like ID: the receipt must stay in artifacts_dir.
    context = {"rollout_id": "../../rollout-id"}
    token = _RUN_CONTEXT.set(context)
    try:
        with patch.object(agent, "_sandbox_model_url", return_value="https://gym-proxy.example"):
            result = await agent.responses(MagicMock(), NeMoGymResponseCreateParamsNonStreaming(input="hello"))
    finally:
        _RUN_CONTEXT.reset(token)
    assert "_ng_agent_observations" not in result.model_dump()
    assert context["harness_failed"] is failed
    assert all(gap.code != "no_sandbox_runtime" for gap in context["observations"].gaps)
    assert context["observations"].records[-1].sandbox_id == "box-test"
    dest = tmp_path / sha256(b"../../rollout-id").hexdigest()
    assert json.loads((dest / "generation.json").read_text())["harness_failed"] is failed
    assert (dest / "runner.log").read_text() == "runner logs"
    agent._provider.close.assert_awaited_once_with(handle)


async def test_completed_harness_failure_skips_judge():
    agent = _make_agent(execution_failure_reward_zero=True)
    agent.server_client.post = AsyncMock(return_value=MagicMock(cookies={}))
    body = HarnessAgentRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hello"))

    async def failed_response(_, __, ___):
        _RUN_CONTEXT.get()["harness_failed"] = True
        return NeMoGymResponse.model_validate(_response())

    with (
        patch.object(HarnessAgent, "responses", new=failed_response),
        patch("responses_api_agents.harness_agent.app.raise_for_status", new=AsyncMock()),
        patch("responses_api_agents.harness_agent.app.get_response_json", new=AsyncMock(return_value={})),
    ):
        result = await agent.run(MagicMock(cookies={}), body)
    assert result.reward == 0 and result.harness_failed
    assert agent.server_client.post.await_count == 1
    assert agent.server_client.post.await_args.kwargs["url_path"] == "/seed_session"


async def test_tool_session_requires_authenticated_metadata_and_preserves_cookies():
    agent = _make_agent(tool_servers=[{"type": "resources_servers", "name": "search"}])
    seeded = MagicMock()
    seeded.read = AsyncMock(
        return_value=json.dumps({"mcp": {"url_path": "/mcp", "headers": {"Authorization": "test-token"}}}).encode()
    )
    agent.server_client.post = AsyncMock(return_value=seeded)
    body = HarnessAgentRunRequest(responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="hello"))
    with patch.object(agent, "_tool_server_url", return_value="https://tools.example"):
        result = await agent._seed_tool_servers(body, {"session": "one"})
    assert result["search"]["url"] == "https://tools.example/mcp"
    assert result["search"]["headers"] == {"Authorization": "test-token"}
    assert agent.server_client.post.await_args.kwargs["cookies"] == {"session": "one"}
    seeded.read.return_value = b'{"mcp": {"url_path": "/mcp"}}'
    with pytest.raises(ValueError, match="authenticated MCP"):
        await agent._seed_tool_servers(body, {})


async def test_restricted_tools_allow_only_model_and_tool_hosts():
    agent = _make_agent(
        network_access="model_and_tools", tool_servers=[{"type": "resources_servers", "name": "search"}]
    )
    agent._provider.name = "opensandbox"
    agent._provider.create = AsyncMock(return_value=MagicMock(provider_name="opensandbox"))
    agent._provider.exec = AsyncMock(return_value=SandboxExecResult("", "", 0))
    agent._provider.upload_file = AsyncMock()
    with patch.object(agent, "_tool_server_url", return_value="https://tools.example"):
        await agent._provision_box("image", {}, "https://model.example")
    policy = agent._provider.create.await_args.args[0].provider_options["network_policy"]
    assert policy == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "model.example"}, {"action": "allow", "target": "tools.example"}],
    }
    with patch.object(agent, "_tool_server_url", return_value="http://127.0.0.1:80"):
        with pytest.raises(ValueError, match="sandbox-reachable"):
            await agent._provision_box("image", {}, "https://model.example")


async def test_concurrent_rollouts_keep_mcp_tokens_out_of_shared_config():
    agent = _make_agent(agent="opencode", agent_kwargs={"opencode_config": {"mcp": {"existing": {"enabled": False}}}})
    agent._provision_box = AsyncMock(return_value=MagicMock(provider_name="opensandbox"))
    agent._provider.exec = AsyncMock(return_value=SandboxExecResult("RUNNER_DONE", "", 0))
    agent._provider.close = AsyncMock()
    agent._download_json = AsyncMock(return_value=_response())

    async def run_one(identifier):
        token = _RUN_CONTEXT.set(
            {"rollout_id": identifier, "mcp": {"search": {"headers": {"Authorization": identifier}}}}
        )
        try:
            await asyncio.sleep(0)
            await agent.responses(MagicMock(), NeMoGymResponseCreateParamsNonStreaming(input=identifier))
        finally:
            _RUN_CONTEXT.reset(token)

    with patch.object(agent, "_sandbox_model_url", return_value="https://model.example"):
        await asyncio.gather(run_one("one"), run_one("two"))
    assert agent.config.agent_kwargs == {"opencode_config": {"mcp": {"existing": {"enabled": False}}}}
    for call in agent._provision_box.await_args_list:
        files = call.args[1]
        identifier = json.loads(files["/work/runner_config.json"])["rollout_id"]
        mcp = json.loads(files["/work/agent_config.json"])["opencode_config"]["mcp"]
        assert mcp["search"]["headers"]["Authorization"] == identifier
        assert mcp["existing"] == {"enabled": False}

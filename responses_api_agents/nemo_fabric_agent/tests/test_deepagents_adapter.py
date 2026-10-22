# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import socket
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest


with patch.object(socket.socket, "connect", side_effect=AssertionError("network is forbidden")):
    import deepagents
    from deepagents.middleware.filesystem import FilesystemMiddleware
    from nemo_fabric import Fabric, FabricConfig
    from nemo_fabric.errors import FabricConfigError
    from nemo_fabric_adapter_contract import models as contract
    from nemo_fabric_adapters.deepagents import adapter as native
    from pydantic import ValidationError

    from responses_api_agents.nemo_fabric_agent.deepagents_adapter import DeepAgentsRuntime


@pytest.fixture(autouse=True)
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", Mock(side_effect=AssertionError("network is forbidden")))


@pytest.fixture
def start_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(native, "preflight_check", Mock())
    monkeypatch.setattr(native, "build_chat_model", Mock(return_value=(None, "offline-model", None)))
    created = {}
    monkeypatch.setattr(deepagents, "create_deep_agent", lambda **kwargs: created.update(kwargs))
    workspace = tmp_path / "app"
    workspace.mkdir()
    monkeypatch.chdir(workspace)

    async def start(root: str | None = None):
        payload = {
            "base_dir": str(workspace),
            "config": contract.AgentConfig.from_mapping(
                {
                    "harness": {"settings": {} if root is None else {"filesystem_root": root}},
                    "models": {"default": {"provider": "openai-compatible", "model": "offline-model"}},
                }
            ),
            "runtime_context": {
                "runtime_id": "offline-runtime",
                "request_id": "offline-request",
                "invocation_id": "offline-invocation",
                "environment": {
                    "environment_id": "offline-environment",
                    "provider": "local",
                    "workspace": str(workspace),
                    "control_location": "in_env_control",
                    "ownership": "caller_owned",
                },
                "artifacts": {},
            },
        }
        original = deepcopy(payload)
        cwd = Path.cwd()
        runtime = DeepAgentsRuntime()
        await runtime.start(payload)
        try:
            assert payload == original
            assert Path.cwd() == cwd
            assert runtime._start_payload["base_dir"] == str(workspace)
            return created["backend"]
        finally:
            await runtime.stop()

    return start


def test_default_root_reproduces_workspace_virtualization(tmp_path: Path, start_backend) -> None:
    backend = asyncio.run(start_backend())
    result = backend.write("/logs/agent/answer.txt", "offline answer")
    assert result.error is None
    assert (tmp_path / "app/logs/agent/answer.txt").read_text() == "offline answer"
    assert not (tmp_path / "logs/agent/answer.txt").exists()
    assert backend.ls("/app").error is not None


def test_explicit_root_reads_app_and_writes_physical_logs(tmp_path: Path, start_backend) -> None:
    (tmp_path / "app/source.txt").write_text("offline source")
    backend = asyncio.run(start_backend(str(tmp_path)))
    assert backend._resolve_path("/app/source.txt") == tmp_path / "app/source.txt"
    assert backend.read("/app/source.txt").file_data["content"] == "offline source"
    tool = FilesystemMiddleware(backend=backend)._create_write_file_tool()
    result = tool.func(
        file_path="/logs/agent/answer.txt",
        content="offline answer",
        runtime=SimpleNamespace(tool_call_id="offline-write"),
    )
    assert result.status == "success"
    assert (tmp_path / "logs/agent/answer.txt").read_text() == "offline answer"
    assert not (tmp_path / "app/logs").exists()
    assert list((tmp_path / "app/artifacts/deepagents/.fabric/runtimes").glob("*.sqlite"))


def test_root_slash_preserves_real_absolute_paths(tmp_path: Path, start_backend) -> None:
    source = tmp_path / "app/source.txt"
    source.write_text("offline source")
    backend = asyncio.run(start_backend("/"))
    answer = tmp_path / "logs/agent/answer.txt"
    assert backend._resolve_path(str(source)) == source
    assert backend.read(str(source)).file_data["content"] == "offline source"
    assert backend._resolve_path(str(answer)) == answer
    assert backend.write(str(answer), "offline answer").error is None
    assert answer.read_text() == "offline answer"


def test_missing_content_and_traversal_still_fail_before_write(tmp_path: Path, start_backend) -> None:
    backend = asyncio.run(start_backend(str(tmp_path)))
    tool = FilesystemMiddleware(backend=backend)._create_write_file_tool()
    with pytest.raises(ValidationError, match="content"):
        tool.invoke({"file_path": "/logs/agent/answer.txt"})
    with pytest.raises(ValueError, match="traversal"):
        backend._resolve_path("/../answer.txt")
    assert not (tmp_path / "logs").exists()
    assert backend.virtual_mode is True


def test_explicit_descriptor_plans_without_shared_registry_edit(tmp_path: Path) -> None:
    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.nemo_fabric_agent.app import NeMoFabricAgent, NeMoFabricAgentConfig

    descriptor = Path(__file__).parents[1] / "deepagents.fabric-adapter.json"
    agent = NeMoFabricAgent(
        config=NeMoFabricAgentConfig(
            host="127.0.0.1",
            port=0,
            name="offline-deepagents",
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "offline-resources"},
            model_server={"type": "responses_api_models", "name": "offline-model"},
            adapter_id="nemo-gym.deepagents",
            model_provider="openai-compatible",
            model="offline-model",
            harness_settings={"filesystem_root": "/"},
            fabric_config={"discovery": {"local_paths": [str(descriptor)]}},
        ),
        server_client=Mock(spec=ServerClient),
    )
    config = agent._fabric_config(
        model_base_url="http://offline-model/v1",
        workspace=tmp_path,
        system_prompt=None,
        mcp_servers={},
        skills=[],
    )
    plan = Fabric().plan(config, base_dir=tmp_path).to_mapping()
    assert plan["adapter"]["adapter_id"] == "nemo-gym.deepagents"
    assert plan["adapter"]["runner"]["module"] == "responses_api_agents.nemo_fabric_agent.deepagents_adapter"
    assert config.to_mapping()["environment"]["workspace"] == str(tmp_path)
    local = json.loads(descriptor.read_text())
    assert "deepagents" in local["settings_schema"]["properties"]


@pytest.mark.parametrize("root", ["relative/path", "", None, 3])
def test_descriptor_rejects_invalid_root(tmp_path: Path, root: object) -> None:
    descriptor = Path(__file__).parents[1] / "deepagents.fabric-adapter.json"
    with pytest.raises(FabricConfigError, match="filesystem_root"):
        Fabric().plan(
            FabricConfig.from_mapping(
                {
                    "metadata": {"name": "offline-deepagents"},
                    "harness": {"adapter_id": "nemo-gym.deepagents", "settings": {"filesystem_root": root}},
                    "discovery": {"local_paths": [str(descriptor)]},
                }
            ),
            base_dir=tmp_path,
        )

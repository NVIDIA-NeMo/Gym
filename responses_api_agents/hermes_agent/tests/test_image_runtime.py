# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Runtime selection and configuration; doubles do not prove installed native image support."""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml

from nemo_gym.agents import AgentSeedSessionRequest, SandboxSession
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from responses_api_agents.hermes_agent import sandbox, sandbox_runner
from responses_api_agents.hermes_agent.app import HermesAgent, HermesAgentConfig


@pytest.mark.parametrize("image", [False, True])
def test_image_runtime_config(image: bool) -> None:
    config = HermesAgentConfig(
        host="127.0.0.1",
        port=8000,
        name="hermes",
        entrypoint="app.py",
        model_server={"type": "responses_api_models", "name": "model"},
        image_input=image,
        context_length=131072,
    )
    agent = HermesAgent(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    native = yaml.safe_load(agent._build_config())
    assert native["model"]["context_length"] == 131072
    if image:
        assert native["model"]["supports_vision"] is True
        assert native["model"]["provider"] == "custom"
        assert native["agent"]["image_input_mode"] == "native"
    else:
        assert "supports_vision" not in native["model"]
        assert "image_input_mode" not in native["agent"]
    assert "sandbox_python" not in HermesAgentConfig.model_fields
    assert HermesAgentConfig.model_fields["image_input"].default is False


def test_image_runtime_has_independent_pins() -> None:
    runtime = sandbox._IMAGE_RUNTIME
    assert "0ee05d72f2b30385d4861fd8e30511c6d51100e4" in runtime.requirements[0]
    assert runtime.constraints is not None
    assert "mcp==1.26.0" in runtime.pins
    assert "openai==2.24.0" in runtime.pins
    assert "decide_image_input_mode" in runtime.import_check
    assert runtime.directory != sandbox._HERMES_RUNTIME.directory


@pytest.mark.parametrize("image", [False, True])
@pytest.mark.parametrize("installed", [False, True])
async def test_seed_reuses_or_installs_selected_runtime(monkeypatch, image: bool, installed: bool) -> None:
    box = AsyncMock()
    box.exec.return_value = SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")
    selected = sandbox._IMAGE_RUNTIME if image else sandbox._HERMES_RUNTIME
    runtime = SimpleNamespace(
        directory=selected.directory,
        python=selected.python,
        is_installed=AsyncMock(return_value=installed),
        install=AsyncMock(),
    )
    other = SimpleNamespace(is_installed=AsyncMock(), install=AsyncMock())
    monkeypatch.setattr(sandbox, "_IMAGE_RUNTIME" if image else "_HERMES_RUNTIME", runtime)
    monkeypatch.setattr(sandbox, "_HERMES_RUNTIME" if image else "_IMAGE_RUNTIME", other)
    state = sandbox.HermesSandboxSession(
        request=AgentSeedSessionRequest(
            agent_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="fixture", task_id="1"),
        ),
        session=SandboxSession(sandbox=box, workdir="/workspace", session_dir="/tmp/session", harness="Hermes"),
        image_input=image,
    )
    await state.install_runtime(install_timeout=30)
    runtime.is_installed.assert_awaited_once_with(box)
    if installed:
        runtime.install.assert_not_awaited()
    else:
        runtime.install.assert_awaited_once_with(box, timeout_s=30)
    other.is_installed.assert_not_awaited()
    other.install.assert_not_awaited()
    destinations = [call.args[1] for call in box.upload.await_args_list]
    assert destinations == [
        f"{runtime.directory}/{name}" for name in ("sandbox_runner.py", "sandbox_observer.py", "model_kwargs.py")
    ]
    command = await state.stage_activation({})
    assert command.argv[0] == runtime.python
    assert command.argv[3] == f"{runtime.directory}/sandbox_runner.py"


@pytest.mark.parametrize("modern", [False, True])
def test_streaming_constructor_mapping(monkeypatch: pytest.MonkeyPatch, modern: bool) -> None:
    if modern:

        class Agent:
            def __init__(self, *, base_url: str) -> None:
                self.base_url = base_url
    else:

        class Agent:
            def __init__(self, *, base_url: str, use_streaming: bool = True) -> None:
                self.base_url = base_url
                self.use_streaming = use_streaming

    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=Agent))
    sandbox_runner._use_model_server("http://fixture/v1")
    agent = Agent(base_url="http://fixture/v1")
    assert agent.base_url == "http://fixture/v1"
    assert agent._disable_streaming if modern else not agent.use_streaming


def test_image_runtime_check_does_not_inherit_task_hermes_home(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "task-hermes"))
    prefix = sandbox._IMAGE_RUNTIME.import_check.split("import run_agent;", 1)[0]
    exec(prefix, {})
    assert os.environ["HERMES_HOME"] == os.path.join(os.environ["HOME"], ".hermes")
    assert not (tmp_path / "task-hermes").exists()


@pytest.mark.parametrize("image_model", [None, "fixture-policy"])
def test_primary_runtime_binding_is_image_only(monkeypatch, image_model: str | None) -> None:
    from agent import auxiliary_client

    binding = MagicMock()
    monkeypatch.setattr(auxiliary_client, "set_runtime_main", binding, raising=False)

    class Agent:
        def __init__(self, *, base_url: str) -> None:
            self.base_url = base_url

    monkeypatch.setitem(sys.modules, "run_agent", SimpleNamespace(AIAgent=Agent))
    sandbox_runner._use_model_server("http://primary/v1", image_model=image_model)
    if image_model is None:
        binding.assert_not_called()
    else:
        binding.assert_called_once_with(
            "custom", "fixture-policy", base_url="http://primary/v1", api_key="gym", api_mode="chat_completions"
        )

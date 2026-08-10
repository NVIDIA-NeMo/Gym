# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml
from harbor.models.agent.context import AgentContext

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymEasyInputMessage, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.terminus_2_agent.app import (
    LocalEnvironment,
    LocalTmuxSession,
    StandaloneTerminus2,
    Terminus2Agent,
    Terminus2AgentConfig,
    Terminus2NemoGym,
    _extract_instruction,
)


def _config(**kwargs) -> Terminus2AgentConfig:
    return Terminus2AgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="terminus_2_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        **kwargs,
    )


def _make_agent(**kwargs) -> Terminus2Agent:
    with patch("responses_api_agents.terminus_2_agent.app.Terminus2Agent.model_post_init"):
        agent = Terminus2Agent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


class TestInstructionExtraction:
    def test_string_input(self) -> None:
        assert _extract_instruction("do the task") == ("do the task", None)

    def test_system_and_latest_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="system", content="be careful"),
            NeMoGymEasyInputMessage(role="user", content="first"),
            NeMoGymEasyInputMessage(role="user", content="second"),
        ]
        assert _extract_instruction(items) == ("second", "be careful")


class TestLocalEnvironment:
    @pytest.mark.asyncio
    async def test_exec_uses_workspace_and_env(self, tmp_path: Path) -> None:
        environment = LocalEnvironment(tmp_path, command_timeout_sec=5)
        result = await environment.exec('printf \'%s:%s\' "$PWD" "$VALUE"', env={"VALUE": "ok"})
        assert result.return_code == 0
        assert result.stdout == f"{tmp_path}:ok"

    @pytest.mark.asyncio
    async def test_exec_timeout_is_reported(self, tmp_path: Path) -> None:
        environment = LocalEnvironment(tmp_path, command_timeout_sec=0.01)
        result = await environment.exec("sleep 1")
        assert result.return_code == 124
        assert "timed out" in (result.stderr or "")


class TestLocalTmuxSession:
    def test_macos_launcher_does_not_use_linux_script_flags(self, tmp_path: Path) -> None:
        session = LocalTmuxSession(
            session_name="terminus-test",
            environment=MagicMock(),
            logging_path=tmp_path / "pane.log",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        with patch("responses_api_agents.terminus_2_agent.app.sys.platform", "darwin"):
            command = session._tmux_start_session
        assert "script -qc" not in command
        assert "tmux new-session" in command

    def test_linux_launcher_stays_upstream(self, tmp_path: Path) -> None:
        session = LocalTmuxSession(
            session_name="terminus-test",
            environment=MagicMock(),
            logging_path=tmp_path / "pane.log",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        with patch("responses_api_agents.terminus_2_agent.app.sys.platform", "linux"):
            command = session._tmux_start_session
        assert "script -qc" in command


class TestStandaloneTerminus2:
    @pytest.mark.asyncio
    async def test_tracks_two_consecutive_completion_claims(self) -> None:
        agent = object.__new__(StandaloneTerminus2)
        agent._consecutive_completion_claims = 0
        non_completion = ([], False, "", "analysis", "plan", MagicMock())
        completion = ([], True, "", "analysis", "plan", MagicMock())

        with patch.object(
            Terminus2NemoGym, "_handle_llm_interaction", new=AsyncMock(side_effect=[completion, completion])
        ):
            await agent._handle_llm_interaction()
            assert agent.finished_naturally is False
            await agent._handle_llm_interaction()
            assert agent.finished_naturally is True

        with patch.object(Terminus2NemoGym, "_handle_llm_interaction", new=AsyncMock(return_value=non_completion)):
            await agent._handle_llm_interaction()
        assert agent.finished_naturally is False


class TestResponses:
    @pytest.mark.asyncio
    async def test_converts_terminus_trajectory(self) -> None:
        agent = _make_agent(system_prompt="system config", model="test-model")
        trajectory = {
            "steps": [
                {"step_id": 1, "source": "user", "message": "task"},
                {
                    "step_id": 2,
                    "source": "agent",
                    "message": "Analysis: inspect\nPlan: list files",
                    "tool_calls": [
                        {
                            "tool_call_id": "call_0_1",
                            "function_name": "bash_command",
                            "arguments": {"keystrokes": "ls\n", "duration": 0.1},
                        }
                    ],
                    "observation": {"results": [{"content": "file.txt"}]},
                    "metrics": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            ],
            "final_metrics": {"total_prompt_tokens": 10, "total_completion_tokens": 4},
        }
        context = AgentContext(n_input_tokens=10, n_output_tokens=4, n_cache_tokens=0)
        agent._run_terminus = AsyncMock(return_value=(trajectory, context, {}, False, True))

        body = NeMoGymResponseCreateParamsNonStreaming(
            model="ignored-request-model",
            input=[
                NeMoGymEasyInputMessage(role="system", content="input system"),
                NeMoGymEasyInputMessage(role="user", content="solve it"),
            ],
        )
        with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
            response = await agent.responses(request=None, body=body)

        assert response.model == "test-model"
        assert [item.type for item in response.output] == ["message", "function_call", "function_call_output"]
        assert response.usage.total_tokens == 14
        assert '"finished_naturally": true' in response.metadata["terminus_2"]
        called_instruction = agent._run_terminus.await_args.args[1]
        assert called_instruction == "system config\n\ninput system\n\nsolve it"

    @pytest.mark.asyncio
    async def test_empty_trajectory_gets_assistant_message(self) -> None:
        agent = _make_agent()
        agent._run_terminus = AsyncMock(return_value=({"steps": []}, AgentContext(), {}, False, False))
        body = NeMoGymResponseCreateParamsNonStreaming(model="model", input="task")

        with patch.object(Terminus2Agent, "resolve_model_base_url", return_value="http://model/v1"):
            response = await agent.responses(request=None, body=body)

        assert len(response.output) == 1
        assert response.output[0].type == "message"


class TestConfig:
    def test_defaults(self) -> None:
        config = _config()
        assert config.parser_name == "json"
        assert config.concurrency == 8
        assert config.record_terminal_session is False
        assert config.trajectory_config == {"raw_content": False}

    @pytest.mark.parametrize(
        "relative_path,config_name",
        [
            ("configs/terminus_2_agent.yaml", "terminus_2_agent"),
            ("../anyterminal_agent/configs/anyterminal_terminus_2.yaml", "anyterminal_terminus_2"),
        ],
    )
    def test_yaml_parses(self, relative_path: str, config_name: str) -> None:
        path = Path(__file__).resolve().parent.parent / relative_path
        data = yaml.safe_load(path.read_text())
        assert config_name in data

    def test_requirements_share_harbor_pin(self) -> None:
        agent_dir = Path(__file__).resolve().parent.parent
        terminus_requirement = next(
            line for line in (agent_dir / "requirements.txt").read_text().splitlines() if line.startswith("harbor @")
        )
        harbor_requirement = next(
            line
            for line in (agent_dir.parent / "harbor_agent" / "requirements.txt").read_text().splitlines()
            if line.startswith("harbor @")
        )
        assert terminus_requirement == harbor_requirement

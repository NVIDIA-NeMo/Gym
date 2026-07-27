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
from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pi_agent.app import (
    PiAgent,
    PiAgentConfig,
    ResourcesServerRef,
    _extract_instruction,
    parse_pi_events,
)


def _config(**kwargs) -> PiAgentConfig:
    return PiAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> PiAgent:
    with patch("responses_api_agents.pi_agent.app.PiAgent.model_post_init"):
        agent = PiAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _msg_end(role, content, **extra) -> str:
    return json.dumps({"type": "message_end", "message": {"role": role, "content": content, **extra}})


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 8
        assert cfg.command == "pi"
        assert cfg.command_parts == ["pi"]

    def test_semaphore_initialized(self) -> None:
        agent = _make_agent(concurrency=4)
        assert agent.sem._value == 4


class TestExtractInstruction:
    def test_user_only(self) -> None:
        user, system = _extract_instruction([NeMoGymEasyInputMessage(role="user", content="hello")])
        assert user == "hello"
        assert system is None

    def test_system_plus_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="system", content="be concise"),
            NeMoGymEasyInputMessage(role="user", content="hi"),
        ]
        user, system = _extract_instruction(items)
        assert user == "hi"
        assert system == "be concise"

    def test_empty(self) -> None:
        user, system = _extract_instruction([])
        assert user == ""
        assert system is None


class TestParsePiEvents:
    def test_empty(self) -> None:
        items, usage = parse_pi_events("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_assistant_text_and_usage(self) -> None:
        line = _msg_end(
            "assistant",
            [{"type": "text", "text": "the answer is 4"}],
            usage={"input": 100, "output": 20, "cacheRead": 5},
        )
        items, usage = parse_pi_events(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"
        assert usage["input_tokens"] == 105
        assert usage["output_tokens"] == 20

    def test_user_messages_ignored(self) -> None:
        line = _msg_end("user", [{"type": "text", "text": "hi"}])
        assert parse_pi_events(line)[0] == []

    def test_non_message_end_events_ignored(self) -> None:
        line = json.dumps({"type": "message_update", "message": {"role": "assistant", "content": []}})
        assert parse_pi_events(line)[0] == []

    def test_tool_call_and_result(self) -> None:
        lines = "\n".join(
            [
                _msg_end(
                    "assistant", [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "echo 6"}}]
                ),
                _msg_end("toolResult", [{"type": "text", "text": "6\n"}], toolCallId="c1", toolName="bash"),
                _msg_end("assistant", [{"type": "text", "text": "answer is 6"}]),
            ]
        )
        items, _ = parse_pi_events(lines)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "bash"
        assert json.loads(items[0].arguments)["command"] == "echo 6"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output
        assert isinstance(items[2], NeMoGymResponseOutputMessage)

    def test_malformed_lines_skipped(self) -> None:
        line = "not-json\n" + _msg_end("assistant", [{"type": "text", "text": "ok"}])
        items, _ = parse_pi_events(line)
        assert len(items) == 1


class TestEnv:
    def test_env_passthrough(self) -> None:
        agent = _make_agent(env={"NVIDIA_API_KEY": "k", "EMPTY": ""})
        env = agent._env(Path("/tmp/h"))
        assert env["NVIDIA_API_KEY"] == "k"
        assert env["HOME"] == "/tmp/h"
        assert "EMPTY" not in env


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pi_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "pi_agent" in data
        inner = data["pi_agent"]["responses_api_agents"]["pi_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 8
        assert inner["command"] == "pi"

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
from responses_api_agents.codex_agent.app import (
    CodexAgent,
    CodexAgentConfig,
    ResourcesServerRef,
    _extract_instruction,
    parse_codex_events,
)


def _config(**kwargs) -> CodexAgentConfig:
    return CodexAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> CodexAgent:
    with patch("responses_api_agents.codex_agent.app.CodexAgent.model_post_init"):
        agent = CodexAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _ev(type_, **kwargs) -> str:
    return json.dumps({"type": type_, **kwargs})


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 8
        assert cfg.command == "codex"
        assert cfg.wire_api == "responses"
        assert cfg.command_parts == ["codex"]

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


class TestParseCodexEvents:
    def test_empty(self) -> None:
        items, usage = parse_codex_events("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_agent_message(self) -> None:
        line = _ev("item.completed", item={"id": "i1", "type": "agent_message", "text": "the answer is 4"})
        items, _ = parse_codex_events(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"

    def test_command_execution_becomes_tool_call(self) -> None:
        line = _ev(
            "item.completed",
            item={
                "id": "c1",
                "type": "command_execution",
                "command": "python3 -c 'print(6)'",
                "aggregated_output": "6\n",
            },
        )
        items, _ = parse_codex_events(line)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "shell"
        assert json.loads(items[0].arguments)["command"] == "python3 -c 'print(6)'"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output

    def test_turn_completed_usage(self) -> None:
        line = _ev("turn.completed", usage={"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 5})
        _, usage = parse_codex_events(line)
        assert usage["input_tokens"] == 105
        assert usage["output_tokens"] == 20

    def test_non_completed_items_ignored(self) -> None:
        line = _ev("item.started", item={"id": "i1", "type": "agent_message", "text": "partial"})
        assert parse_codex_events(line)[0] == []

    def test_malformed_lines_skipped(self) -> None:
        line = "not-json\n" + _ev("item.completed", item={"id": "i1", "type": "agent_message", "text": "ok"})
        items, _ = parse_codex_events(line)
        assert len(items) == 1


class TestConfigToml:
    def test_writes_provider_block(self, tmp_path) -> None:
        agent = _make_agent(base_url="https://x/v1", model="m", model_provider="nvinf", api_key_env="NVIDIA_API_KEY")
        agent._write_config_toml(tmp_path)
        toml = (tmp_path / "config.toml").read_text()
        assert 'model_provider = "nvinf"' in toml
        assert "multi_agent = false" in toml
        assert "[model_providers.nvinf]" in toml
        assert 'base_url = "https://x/v1"' in toml
        assert 'wire_api = "responses"' in toml


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "codex_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        inner = data["codex_agent"]["responses_api_agents"]["codex_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["command"] == "codex"
        assert inner["wire_api"] == "responses"

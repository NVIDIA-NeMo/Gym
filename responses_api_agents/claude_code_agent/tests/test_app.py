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
from responses_api_agents.claude_code_agent.app import (
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    ResourcesServerRef,
    _extract_instruction,
    parse_stream_json,
)


def _config(**kwargs) -> ClaudeCodeAgentConfig:
    return ClaudeCodeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> ClaudeCodeAgent:
    with patch("responses_api_agents.claude_code_agent.app.ClaudeCodeAgent.model_post_init"):
        agent = ClaudeCodeAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _event(type_: str, **kwargs) -> str:
    return json.dumps({"type": type_, **kwargs})


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 32
        assert cfg.max_turns == 30
        assert cfg.timeout == 300
        assert cfg.model == "claude-sonnet-4-6"

    def test_semaphore_initialized(self) -> None:
        agent = _make_agent(concurrency=4)
        assert agent.sem._value == 4


class TestExtractInstruction:
    def test_user_only(self) -> None:
        items = [NeMoGymEasyInputMessage(role="user", content="hello")]
        user, system = _extract_instruction(items)
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


class TestParseStreamJson:
    def _assistant(self, content: list) -> str:
        return _event("assistant", message={"content": content, "usage": {"input_tokens": 10, "output_tokens": 5}})

    def _user_tool_result(self, tool_use_id: str, result: str) -> str:
        return _event(
            "user", message={"content": [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result}]}
        )

    def test_empty(self) -> None:
        items, usage = parse_stream_json("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_text_message(self) -> None:
        line = self._assistant([{"type": "text", "text": "hello"}])
        items, usage = parse_stream_json(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "hello"
        assert usage["input_tokens"] == 10
        assert usage["output_tokens"] == 5

    def test_thinking_prepended(self) -> None:
        line = self._assistant(
            [
                {"type": "thinking", "thinking": "let me reason"},
                {"type": "text", "text": "answer"},
            ]
        )
        items, _ = parse_stream_json(line)
        assert len(items) == 1
        text = items[0].content[0].text
        assert "<think>\nlet me reason\n</think>" in text
        assert "answer" in text

    def test_thinking_without_text_not_emitted(self) -> None:
        line = self._assistant([{"type": "thinking", "thinking": "just thinking"}])
        items, _ = parse_stream_json(line)
        assert items == []

    def test_thinking_cleared_after_message(self) -> None:
        l1 = self._assistant([{"type": "thinking", "thinking": "think"}, {"type": "text", "text": "msg1"}])
        l2 = self._assistant([{"type": "text", "text": "msg2"}])
        items, _ = parse_stream_json(f"{l1}\n{l2}")
        assert len(items) == 2
        assert "<think>" in items[0].content[0].text
        assert "<think>" not in items[1].content[0].text

    def test_tool_call_and_result(self) -> None:
        assistant_line = self._assistant(
            [
                {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}},
            ]
        )
        user_line = self._user_tool_result("t1", "file.txt\n")
        items, _ = parse_stream_json(f"{assistant_line}\n{user_line}")
        assert len(items) == 2
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "Bash"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert "file.txt" in items[1].output

    def test_text_then_tool_call(self) -> None:
        assistant_line = self._assistant(
            [
                {"type": "text", "text": "running bash"},
                {"type": "tool_use", "id": "t2", "name": "Bash", "input": {"command": "pwd"}},
            ]
        )
        user_line = self._user_tool_result("t2", "/home/user\n")
        items, _ = parse_stream_json(f"{assistant_line}\n{user_line}")
        assert len(items) == 3
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert isinstance(items[1], NeMoGymResponseFunctionToolCall)
        assert isinstance(items[2], NeMoGymFunctionCallOutput)

    def test_malformed_lines_skipped(self) -> None:
        good = self._assistant([{"type": "text", "text": "ok"}])
        items, _ = parse_stream_json(f"not-json\n{good}\n{{bad")
        assert len(items) == 1

    def test_result_event_accumulates_usage(self) -> None:
        result = _event("result", usage={"input_tokens": 100, "output_tokens": 50})
        _, usage = parse_stream_json(result)
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "claude_code_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "claude_code_agent" in data
        inner = data["claude_code_agent"]["responses_api_agents"]["claude_code_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 32
        assert inner["max_turns"] == 30

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

from nemo_gym.config_types import ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.kilocode_agent.app import (
    KiloCodeAgent,
    KiloCodeAgentConfig,
    _extract_instruction,
    parse_kilo_events,
)


def _config(**kwargs) -> KiloCodeAgentConfig:
    return KiloCodeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> KiloCodeAgent:
    with patch("responses_api_agents.kilocode_agent.app.KiloCodeAgent.model_post_init"):
        agent = KiloCodeAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _events(*objs) -> str:
    """Serialize a list of event dicts into the JSONL stream kilo run --format json emits."""
    return "\n".join(json.dumps(o) for o in objs)


def _text_event(text: str, pid: str = "prt_text") -> dict:
    return {"type": "text", "sessionID": "s", "part": {"id": pid, "type": "text", "text": text, "time": {"end": 1}}}


def _reasoning_event(text: str, pid: str = "prt_reason") -> dict:
    return {
        "type": "reasoning",
        "sessionID": "s",
        "part": {"id": pid, "type": "reasoning", "text": text, "time": {"end": 1}},
    }


def _tool_event(tool: str, call_id: str, state: dict, pid: str = "prt_tool") -> dict:
    return {
        "type": "tool_use",
        "sessionID": "s",
        "part": {"id": pid, "type": "tool", "tool": tool, "callID": call_id, "state": state},
    }


def _step_finish_event(tokens: dict, pid: str = "prt_step") -> dict:
    return {"type": "step_finish", "sessionID": "s", "part": {"id": pid, "type": "step-finish", "tokens": tokens}}


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 8
        assert cfg.command == "kilo"
        assert cfg.thinking is False
        assert cfg.command_parts == ["kilo"]

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


class TestParseKiloEvents:
    def test_empty_stream(self) -> None:
        items, usage = parse_kilo_events("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_malformed_lines_skipped(self) -> None:
        stream = "not json\n" + _events(_text_event("answer is 4")) + "\n{ broken"
        items, _ = parse_kilo_events(stream)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)

    def test_assistant_text(self) -> None:
        items, _ = parse_kilo_events(_events(_text_event("the answer is 4")))
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"

    def test_blank_text_ignored(self) -> None:
        items, _ = parse_kilo_events(_events(_text_event("   ")))
        assert items == []

    def test_tool_call_and_output(self) -> None:
        stream = _events(
            _tool_event("bash", "c1", {"status": "completed", "input": {"command": "echo 6"}, "output": "6\n"}),
            _text_event("answer is 6"),
        )
        items, _ = parse_kilo_events(stream)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "bash"
        assert json.loads(items[0].arguments)["command"] == "echo 6"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output
        assert isinstance(items[2], NeMoGymResponseOutputMessage)

    def test_tool_error_surfaces_error_text(self) -> None:
        stream = _events(_tool_event("bash", "c2", {"status": "error", "error": "boom"}))
        items, _ = parse_kilo_events(stream)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].output == "boom"

    def test_step_finish_usage(self) -> None:
        stream = _events(_step_finish_event({"input": 100, "output": 20, "cache": {"read": 5}}))
        _, usage = parse_kilo_events(stream)
        assert usage["input_tokens"] == 105
        assert usage["output_tokens"] == 20

    def test_reasoning_prepended_to_next_text(self) -> None:
        stream = _events(_reasoning_event("let me think"), _text_event("final answer"))
        items, _ = parse_kilo_events(stream)
        assert len(items) == 1
        text = items[0].content[0].text
        assert "<think>" in text
        assert "let me think" in text
        assert text.endswith("final answer")

    def test_trailing_reasoning_surfaced(self) -> None:
        items, _ = parse_kilo_events(_events(_reasoning_event("dangling thought")))
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert "dangling thought" in items[0].content[0].text

    def test_reasoning_ignored_when_no_thinking_events(self) -> None:
        # Without --thinking kilo emits no reasoning events; the parser must not fabricate any.
        items, _ = parse_kilo_events(_events(_text_event("plain")))
        assert len(items) == 1
        assert "<think>" not in items[0].content[0].text

    def test_duplicate_part_events_deduped(self) -> None:
        # kilo emits each part.updated event twice with the same part id; the parser must not
        # double the tool call, output, message, or token counts.
        tool = _tool_event("bash", "c1", {"status": "completed", "input": {"command": "echo 6"}, "output": "6\n"})
        text = _text_event("answer is 6")
        step = _step_finish_event({"input": 100, "output": 20, "cache": {"read": 5}})
        stream = _events(tool, tool, step, step, text, text)
        items, usage = parse_kilo_events(stream)
        assert [type(i).__name__ for i in items] == [
            "NeMoGymResponseFunctionToolCall",
            "NeMoGymFunctionCallOutput",
            "NeMoGymResponseOutputMessage",
        ]
        assert usage == {"input_tokens": 105, "output_tokens": 20}


class TestDeepMerge:
    def test_nested_merge(self) -> None:
        base = {"a": {"b": 1, "c": 2}}
        KiloCodeAgent._deep_merge(base, {"a": {"c": 3, "d": 4}})
        assert base == {"a": {"b": 1, "c": 3, "d": 4}}


class TestEnv:
    def test_env_passthrough_and_isolation(self) -> None:
        agent = _make_agent(openai_api_key="k", openai_base_url="https://x/v1", env={"FOO": "bar", "EMPTY": ""})
        env = agent._env("/tmp/data", "/tmp/config")
        assert env["OPENAI_API_KEY"] == "k"
        assert env["OPENAI_BASE_URL"] == "https://x/v1"
        assert env["XDG_DATA_HOME"] == "/tmp/data"
        assert env["XDG_CONFIG_HOME"] == "/tmp/config"
        assert env["KILO_NO_DAEMON"] == "1"
        assert env["KILO_DB"] == ":memory:"
        assert env["FOO"] == "bar"
        assert "EMPTY" not in env


class TestBuildCommand:
    def test_command_shape(self) -> None:
        agent = _make_agent(model="nvinf/some-model")
        cmd = agent._build_command(Path("/tmp/ws"), "solve it")
        assert cmd[:4] == ["kilo", "run", "--auto", "--pure"]
        assert "--format" in cmd and cmd[cmd.index("--format") + 1] == "json"
        assert cmd[cmd.index("-m") + 1] == "nvinf/some-model"
        assert cmd[cmd.index("--dir") + 1] == "/tmp/ws"
        # prompt is passed after `--` so a leading-dash prompt is safe
        assert cmd[-2:] == ["--", "solve it"]
        assert "--thinking" not in cmd

    def test_thinking_flag_gated(self) -> None:
        agent = _make_agent(thinking=True)
        cmd = agent._build_command(Path("/tmp/ws"), "hi")
        assert "--thinking" in cmd


class TestWriteConfig:
    def test_writes_kilo_json(self, tmp_path) -> None:
        agent = _make_agent(kilo_config={"permission": {"bash": "allow"}})
        agent._write_kilo_config(tmp_path)
        written = json.loads((tmp_path / "kilo.json").read_text())
        assert written["permission"]["bash"] == "allow"

    def test_no_config_no_file(self, tmp_path) -> None:
        agent = _make_agent()
        agent._write_kilo_config(tmp_path)
        assert not (tmp_path / "kilo.json").exists()


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "kilocode_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "kilocode_agent" in data
        inner = data["kilocode_agent"]["responses_api_agents"]["kilocode_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 8
        assert inner["command"] == "kilo"

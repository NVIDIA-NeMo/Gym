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

    def test_runtime_capability_defaults(self) -> None:
        cfg = _config()
        assert cfg.bare is True
        assert cfg.mcp_config is None
        assert cfg.settings is None

    def test_semaphore_initialized(self) -> None:
        agent = _make_agent(concurrency=4)
        assert agent.sem._value == 4


class TestBuildCommand:
    def test_default_passes_bare(self) -> None:
        agent = _make_agent()
        cmd = agent._build_command("claude-sonnet-4-6", "do the thing")
        assert "--bare" in cmd
        assert "--mcp-config" not in cmd
        # instruction is the final positional after the `--` separator
        assert cmd[-2:] == ["--", "do the thing"]
        assert cmd[:6] == [
            "claude",
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
        ]

    def test_bare_false_omits_flag(self) -> None:
        agent = _make_agent(bare=False)
        cmd = agent._build_command("m", "x")
        assert "--bare" not in cmd

    def test_mcp_config_passed_independently_of_bare(self) -> None:
        agent = _make_agent(mcp_config="/path/to/mcp.json")
        cmd = agent._build_command("m", "x")
        # --mcp-config is explicit, so it coexists with the default --bare
        assert "--bare" in cmd
        assert cmd[cmd.index("--mcp-config") + 1] == "/path/to/mcp.json"

    def test_optional_flags_threaded_through(self) -> None:
        agent = _make_agent(
            allowed_tools="Bash,Read",
            disallowed_tools="Write",
            thinking="enabled",
            max_thinking_tokens=1024,
            max_turns=7,
        )
        cmd = agent._build_command("m", "x", system_prompt="be terse")
        assert cmd[cmd.index("--allowedTools") + 1] == "Bash,Read"
        assert cmd[cmd.index("--disallowedTools") + 1] == "Write"
        assert cmd[cmd.index("--thinking") + 1] == "enabled"
        assert cmd[cmd.index("--max-thinking-tokens") + 1] == "1024"
        assert cmd[cmd.index("--max-turns") + 1] == "7"
        assert cmd[cmd.index("--append-system-prompt") + 1] == "be terse"


class TestBuildSettings:
    def test_default_disables_telemetry(self) -> None:
        agent = _make_agent()
        settings = agent._build_settings()
        assert settings["env"]["CLAUDE_CODE_ENABLE_TELEMETRY"] == "0"
        assert settings["env"]["CLAUDE_CODE_ATTRIBUTION_HEADER"] == "0"
        assert set(settings.keys()) == {"env"}

    def test_user_settings_merged_preserving_telemetry(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"env": {"FOO": "bar"}, "permissions": {"allow": ["Bash"]}}))
        agent = _make_agent(settings=str(settings_file))
        settings = agent._build_settings()
        # user env layered on top of telemetry defaults
        assert settings["env"]["FOO"] == "bar"
        assert settings["env"]["CLAUDE_CODE_ENABLE_TELEMETRY"] == "0"
        # non-env top-level keys passed through
        assert settings["permissions"] == {"allow": ["Bash"]}

    def test_user_settings_can_override_telemetry(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.json"
        settings_file.write_text(json.dumps({"env": {"CLAUDE_CODE_ENABLE_TELEMETRY": "1"}}))
        agent = _make_agent(settings=str(settings_file))
        settings = agent._build_settings()
        assert settings["env"]["CLAUDE_CODE_ENABLE_TELEMETRY"] == "1"


class TestSetupConfigDir:
    def test_creates_dir_with_settings(self, tmp_path: Path) -> None:
        agent = _make_agent()
        with patch("responses_api_agents.claude_code_agent.app.Path.home", return_value=tmp_path):
            config_dir = agent._setup_config_dir()
        try:
            settings_path = config_dir / "settings.json"
            assert settings_path.is_file()
            written = json.loads(settings_path.read_text())
            assert written["env"]["CLAUDE_CODE_ENABLE_TELEMETRY"] == "0"
        finally:
            import shutil as _shutil

            _shutil.rmtree(config_dir, ignore_errors=True)


class TestRunClaudeCode:
    def test_wires_command_env_and_cleans_up(self, tmp_path: Path) -> None:
        agent = _make_agent(mcp_config="/path/to/mcp.json")
        captured: dict = {}

        class FakeProc:
            returncode = 0

            async def communicate(self):
                return b'{"type":"result","usage":{"input_tokens":3,"output_tokens":4}}\n', b""

        async def fake_exec(*cmd, **kwargs):
            env = kwargs["env"]
            config_dir = env["CLAUDE_CONFIG_DIR"]
            captured["cmd"] = list(cmd)
            captured["config_dir"] = config_dir
            # the staged dir + settings must exist while the subprocess runs
            captured["dir_exists_during_run"] = (Path(config_dir) / "settings.json").is_file()
            captured["sandbox"] = env.get("IS_SANDBOX")
            return FakeProc()

        with (
            patch("responses_api_agents.claude_code_agent.app.Path.home", return_value=tmp_path),
            patch("responses_api_agents.claude_code_agent.app.asyncio.create_subprocess_exec", fake_exec),
        ):
            stdout, model = asyncio.run(agent._run_claude_code("hello", system_prompt="be terse"))

        assert "claude" in captured["cmd"][0]
        assert "--mcp-config" in captured["cmd"]
        assert "be terse" in captured["cmd"]
        assert captured["sandbox"] == "1"
        assert captured["dir_exists_during_run"] is True
        # config dir is removed after the run (no leakage between rollouts)
        assert not Path(captured["config_dir"]).exists()
        assert "result" in stdout
        assert model == "claude-sonnet-4-6"

    def test_timeout_returns_empty(self, tmp_path: Path) -> None:
        agent = _make_agent(timeout=1)
        killed = {"called": False}

        class SlowProc:
            returncode = None

            def kill(self):
                killed["called"] = True

            async def communicate(self):
                return b"", b""

        async def fake_exec(*cmd, **kwargs):
            return SlowProc()

        async def fake_wait_for(coro, timeout):
            coro.close()  # avoid un-awaited coroutine warning
            raise asyncio.TimeoutError

        with (
            patch("responses_api_agents.claude_code_agent.app.Path.home", return_value=tmp_path),
            patch("responses_api_agents.claude_code_agent.app.asyncio.create_subprocess_exec", fake_exec),
            patch("responses_api_agents.claude_code_agent.app.asyncio.wait_for", fake_wait_for),
        ):
            stdout, model = asyncio.run(agent._run_claude_code("hello"))

        assert stdout == ""
        assert killed["called"] is True
        assert model == "claude-sonnet-4-6"


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

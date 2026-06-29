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
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
import yaml
from fastapi import Request

from nemo_gym.global_config import SKILLS_REF_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.claude_code_agent.app import (
    ClaudeCodeAgent,
    ClaudeCodeAgentConfig,
    ClaudeCodeAgentRunRequest,
    ResourcesServerRef,
    _extract_instruction,
    parse_stream_json,
)


def _write_skill_dir(root: Path, name: str = "cot_enhanced") -> Path:
    skills_dir = root / "variant_a"
    skill = skills_dir / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: A skill.\n---\n# Body\n")
    return skills_dir


def _config(**kwargs) -> ClaudeCodeAgentConfig:
    kwargs.setdefault("resources_server", ResourcesServerRef(type="resources_servers", name=""))
    return ClaudeCodeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        **kwargs,
    )


def _make_agent(**kwargs) -> ClaudeCodeAgent:
    # Patch only the external side effect (claude-code install/version check) so the real
    # model_post_init still runs — it initializes the model's private attrs and the semaphore.
    with patch("responses_api_agents.claude_code_agent.app.ensure_claude_code"):
        return ClaudeCodeAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))


def _event(type_: str, **kwargs) -> str:
    return json.dumps({"type": type_, **kwargs})


class FakeAioHTTPResponse:
    ok = True

    def __init__(self, payload: dict, cookies: dict | None = None):
        self.payload = payload
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


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

    def test_dynamic_mcp_config_overrides_static_for_command(self) -> None:
        agent = _make_agent(mcp_config="/path/to/static.json")
        cmd = agent._build_command("m", "x", mcp_config="/tmp/dynamic.json")
        assert cmd[cmd.index("--mcp-config") + 1] == "/tmp/dynamic.json"

    def test_skills_active_forces_bare_off(self) -> None:
        # Default config has bare=True, but staged skills must be discoverable, so --bare is dropped.
        agent = _make_agent()
        cmd = agent._build_command("m", "x", skills_active=True)
        assert "--bare" not in cmd

    def test_skills_inactive_keeps_bare(self) -> None:
        agent = _make_agent()
        cmd = agent._build_command("m", "x", skills_active=False)
        assert "--bare" in cmd

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

    def test_stages_skills_into_config_dir(self, tmp_path: Path) -> None:
        skills_dir = _write_skill_dir(tmp_path)
        home = tmp_path / "home"
        home.mkdir()
        agent = _make_agent()
        with patch("responses_api_agents.claude_code_agent.app.Path.home", return_value=home):
            config_dir = agent._setup_config_dir(skills_path=str(skills_dir))
        try:
            assert (config_dir / "skills" / "cot_enhanced" / "SKILL.md").is_file()
        finally:
            import shutil as _shutil

            _shutil.rmtree(config_dir, ignore_errors=True)


class _FakeHttpResp:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.cookies: dict = {}
        self.ok = True

    async def read(self) -> bytes:
        return orjson.dumps(self._payload)


def _gym_response(text: str = "done") -> dict:
    return {
        "id": "resp_x",
        "created_at": 0.0,
        "model": "claude-sonnet-4-6",
        "object": "response",
        "output": [
            {
                "id": "msg_x",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 1,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 2,
        },
    }


class TestRunForwardsSkillsPath:
    """run() reads skills_ref off the request's model_extra (extra='allow') and forwards its path
    directly to _create_response/_run_claude_code."""

    def _seed_and_verify_post(self):
        async def _post(server_name, url_path, json=None, cookies=None, **kw):
            if url_path == "/verify":
                return _FakeHttpResp(
                    {"responses_create_params": {"input": []}, "response": _gym_response(), "reward": 1.0}
                )
            return _FakeHttpResp({})

        return AsyncMock(side_effect=_post)

    def _run(self, agent: ClaudeCodeAgent, body: ClaudeCodeAgentRunRequest, run_claude_code: AsyncMock):
        agent.server_client.post = self._seed_and_verify_post()
        req = MagicMock()
        req.cookies = {}
        # Stub the CLI invocation; _create_response still runs for real, so we exercise the full
        # run() -> _create_response -> _run_claude_code argument threading.
        with patch.object(
            ClaudeCodeAgent,
            "_run_claude_code",
            run_claude_code,
        ):
            return asyncio.run(agent.run(req, body))

    def test_skills_ref_path_forwarded(self) -> None:
        agent = _make_agent()
        run_claude_code = AsyncMock(return_value=("", "claude-sonnet-4-6"))
        body = ClaudeCodeAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": []},
                SKILLS_REF_KEY_NAME: {"path": "skills/variant_a/", "hash": "abc123", "skills": []},
            }
        )

        self._run(agent, body, run_claude_code)

        assert run_claude_code.call_args.kwargs["skills_path"] == "skills/variant_a/"

    def test_no_skills_ref_forwards_none(self) -> None:
        agent = _make_agent()
        run_claude_code = AsyncMock(return_value=("", "claude-sonnet-4-6"))
        body = ClaudeCodeAgentRunRequest.model_validate({"responses_create_params": {"input": []}})

        self._run(agent, body, run_claude_code)

        assert run_claude_code.call_args.kwargs["skills_path"] is None


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

    def test_skills_staged_and_bare_dropped(self, tmp_path: Path) -> None:
        skills_dir = _write_skill_dir(tmp_path)
        home = tmp_path / "home"
        home.mkdir()
        agent = _make_agent()  # bare defaults to True
        captured: dict = {}

        class FakeProc:
            returncode = 0

            async def communicate(self):
                return b'{"type":"result","usage":{"input_tokens":1,"output_tokens":1}}\n', b""

        async def fake_exec(*cmd, **kwargs):
            config_dir = Path(kwargs["env"]["CLAUDE_CONFIG_DIR"])
            captured["cmd"] = list(cmd)
            captured["skill_staged"] = (config_dir / "skills" / "cot_enhanced" / "SKILL.md").is_file()
            return FakeProc()

        with (
            patch("responses_api_agents.claude_code_agent.app.Path.home", return_value=home),
            patch("responses_api_agents.claude_code_agent.app.asyncio.create_subprocess_exec", fake_exec),
        ):
            asyncio.run(agent._run_claude_code("hello", skills_path=str(skills_dir)))

        assert captured["skill_staged"] is True
        # skills present => --bare must be dropped even though config.bare is True
        assert "--bare" not in captured["cmd"]

    def test_bad_skills_path_does_not_leak_config_dir(self, tmp_path: Path) -> None:
        # stage_skills raises for a missing skills dir; the partially-created config dir must
        # still be cleaned up (setup happens inside the try whose finally rmtree's it).
        home = tmp_path / "home"
        home.mkdir()
        agent = _make_agent()

        with patch("responses_api_agents.claude_code_agent.app.Path.home", return_value=home):
            with pytest.raises(ValueError):
                asyncio.run(agent._run_claude_code("hello", skills_path=str(tmp_path / "does_not_exist")))

        leaked = home / ".claude_code_agent"
        assert not leaked.exists() or not any(leaked.iterdir())

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


class TestRolloutMCPConfig:
    def test_no_metadata_preserves_static_config(self, tmp_path: Path) -> None:
        agent = _make_agent(mcp_config="/path/to/static.json")
        assert agent._write_rollout_mcp_config({}, tmp_path) is None

    def test_writes_rollout_mcp_config_with_session_header(self, tmp_path: Path) -> None:
        agent = _make_agent(resources_server=ResourcesServerRef(type="resources_servers", name="example_mcp_weather"))
        agent.server_client.global_config_dict = {
            "example_mcp_weather": {
                "resources_servers": {
                    "example_mcp_weather": {
                        "host": "127.0.0.1",
                        "port": 8123,
                    }
                }
            }
        }
        agent.server_client._build_server_base_url.side_effect = lambda cfg: f"http://{cfg['host']}:{cfg['port']}"

        config_path = agent._write_rollout_mcp_config(
            {
                "mcp": {
                    "server_name": "example_mcp_weather",
                    "url_path": "/mcp",
                    "headers": {"X-NeMo-Gym-Session-Token": "secret-token"},
                }
            },
            tmp_path,
        )

        assert config_path is not None
        config = json.loads(Path(config_path).read_text())
        server = config["mcpServers"]["example_mcp_weather"]
        assert server["type"] == "http"
        assert server["url"] == "http://127.0.0.1:8123/mcp"
        assert server["headers"]["X-NeMo-Gym-Session-Token"] == "secret-token"

    def test_merges_static_mcp_config_when_metadata_present(self, tmp_path: Path) -> None:
        static_config = tmp_path / "static_mcp.json"
        static_config.write_text(json.dumps({"mcpServers": {"static": {"type": "stdio", "command": "server"}}}))
        agent = _make_agent(
            mcp_config=str(static_config),
            resources_server=ResourcesServerRef(type="resources_servers", name="example_mcp_weather"),
        )
        agent.server_client.global_config_dict = {
            "example_mcp_weather": {
                "resources_servers": {
                    "example_mcp_weather": {
                        "host": "127.0.0.1",
                        "port": 8123,
                    }
                }
            }
        }
        agent.server_client._build_server_base_url.side_effect = lambda cfg: f"http://{cfg['host']}:{cfg['port']}"

        config_path = agent._write_rollout_mcp_config(
            {
                "mcp": {
                    "server_name": "dynamic",
                    "url_path": "/mcp",
                    "headers": {"X-NeMo-Gym-Session-Token": "tok"},
                }
            },
            tmp_path / "run",
        )

        config = json.loads(Path(config_path).read_text())
        assert "static" in config["mcpServers"]
        assert config["mcpServers"]["dynamic"]["headers"]["X-NeMo-Gym-Session-Token"] == "tok"

    def test_run_passes_generated_mcp_config(self, tmp_path: Path) -> None:
        agent = _make_agent(resources_server=ResourcesServerRef(type="resources_servers", name="example_mcp_weather"))
        agent.server_client.global_config_dict = {
            "example_mcp_weather": {
                "resources_servers": {
                    "example_mcp_weather": {
                        "host": "127.0.0.1",
                        "port": 8123,
                    }
                }
            }
        }
        agent.server_client._build_server_base_url.side_effect = lambda cfg: f"http://{cfg['host']}:{cfg['port']}"

        async def fake_post(server_name, url_path, json=None, cookies=None):
            if url_path == "/seed_session":
                return FakeAioHTTPResponse(
                    {
                        "mcp": {
                            "server_name": "example_mcp_weather",
                            "url_path": "/mcp",
                            "headers": {"X-NeMo-Gym-Session-Token": "tok"},
                        }
                    },
                    cookies={"session": "abc"},
                )
            if url_path == "/verify":
                return FakeAioHTTPResponse(json | {"reward": 1.0})
            raise AssertionError(f"unexpected post: {server_name} {url_path}")

        captured: dict = {}

        async def fake_run_claude_code(instruction, system_prompt=None, mcp_config=None, skills_path=None):
            captured["instruction"] = instruction
            captured["mcp_config"] = mcp_config
            captured["config_exists_during_run"] = Path(mcp_config).is_file()
            captured["config"] = json.loads(Path(mcp_config).read_text())
            return _event(
                "assistant",
                message={"content": [{"type": "text", "text": "The weather in Paris is sunny and 72 F."}]},
            ), "claude-sonnet-4-6"

        agent.server_client.post.side_effect = fake_post
        object.__setattr__(agent, "_run_claude_code", fake_run_claude_code)
        request = MagicMock(spec=Request)
        request.cookies = {}
        body = ClaudeCodeAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="use the weather tool"),
            expected_city="Paris",
        )

        result = asyncio.run(agent.run(request, body))

        assert result.reward == 1.0
        assert captured["instruction"] == "use the weather tool"
        assert captured["config_exists_during_run"] is True
        server = captured["config"]["mcpServers"]["example_mcp_weather"]
        assert server["url"] == "http://127.0.0.1:8123/mcp"
        assert server["headers"]["X-NeMo-Gym-Session-Token"] == "tok"
        assert not Path(captured["mcp_config"]).exists()

    def test_run_threads_session_cookie_seed_to_verify(self, tmp_path: Path) -> None:
        agent = _make_agent(resources_server=ResourcesServerRef(type="resources_servers", name="example_mcp_weather"))
        agent.server_client.global_config_dict = {
            "example_mcp_weather": {"resources_servers": {"example_mcp_weather": {"host": "127.0.0.1", "port": 8123}}}
        }
        agent.server_client._build_server_base_url.side_effect = lambda cfg: f"http://{cfg['host']}:{cfg['port']}"

        captured: dict = {}

        async def fake_post(server_name, url_path, json=None, cookies=None):
            if url_path == "/seed_session":
                # the resources server sets a session cookie on the seed response
                return FakeAioHTTPResponse(
                    {
                        "mcp": {
                            "server_name": "example_mcp_weather",
                            "url_path": "/mcp",
                            "headers": {"X-NeMo-Gym-Session-Token": "tok"},
                        }
                    },
                    cookies={"session": "sess-cookie"},
                )
            if url_path == "/verify":
                captured["verify_cookies"] = cookies
                return FakeAioHTTPResponse(json | {"reward": 1.0})
            raise AssertionError(f"unexpected post: {server_name} {url_path}")

        async def fake_run_claude_code(instruction, system_prompt=None, mcp_config=None, skills_path=None):
            captured["config_token"] = json.loads(Path(mcp_config).read_text())["mcpServers"]["example_mcp_weather"][
                "headers"
            ]["X-NeMo-Gym-Session-Token"]
            return _event("assistant", message={"content": [{"type": "text", "text": "ok"}]}), "claude-sonnet-4-6"

        agent.server_client.post.side_effect = fake_post
        object.__setattr__(agent, "_run_claude_code", fake_run_claude_code)
        request = MagicMock(spec=Request)
        request.cookies = {}
        body = ClaudeCodeAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="use the weather tool"),
            verifier_metadata={"expected_city": "Paris"},
        )

        asyncio.run(agent.run(request, body))

        # the cookie set on /seed_session is threaded into the /verify call (same rollout session),
        # and the per-rollout token from seed metadata reaches the generated MCP config.
        assert captured["verify_cookies"] == {"session": "sess-cookie"}
        assert captured["config_token"] == "tok"


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

    def test_result_event_exposes_num_turns(self) -> None:
        result = _event("result", num_turns=9, usage={"input_tokens": 1, "output_tokens": 1})
        _, usage = parse_stream_json(result)
        assert usage["num_turns"] == 9

    def test_num_turns_absent_when_no_result_event(self) -> None:
        assistant = self._assistant([{"type": "text", "text": "hi"}])
        _, usage = parse_stream_json(assistant)
        assert "num_turns" not in usage


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

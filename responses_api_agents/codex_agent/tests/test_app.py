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
import tomllib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import pytest
import yaml
from fastapi import Request
from pydantic import ValidationError

from nemo_gym.global_config import SKILLS_REF_KEY_NAME
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.codex_agent.app import (
    CodexAgent,
    CodexAgentConfig,
    CodexAgentRunRequest,
    ModelServerRef,
    ResourcesServerRef,
    _extract_instruction,
    parse_exec_jsonl,
    toml_dumps,
)


def _write_skill_dir(root: Path, name: str = "cot_enhanced") -> Path:
    skills_dir = root / "variant_a"
    skill = skills_dir / name
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\nname: {name}\ndescription: A skill.\n---\n# Body\n")
    return skills_dir


def _config(**kwargs) -> CodexAgentConfig:
    kwargs.setdefault("resources_server", ResourcesServerRef(type="resources_servers", name=""))
    kwargs.setdefault("codex_version", "0.144.4")
    return CodexAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        **kwargs,
    )


def _make_agent(**kwargs) -> CodexAgent:
    # Patch only the external side effect (codex install/version check) so the real
    # model_post_init still runs — it initializes the semaphore.
    with patch("responses_api_agents.codex_agent.app.ensure_codex"):
        return CodexAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))


def _event(type_: str, **kwargs) -> str:
    return json.dumps({"type": type_, **kwargs})


def _item_completed(item: dict) -> str:
    return _event("item.completed", item=item)


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
        assert cfg.timeout == 600
        assert cfg.model is None
        assert cfg.sandbox_mode == "danger-full-access"
        assert cfg.cwd is None
        assert cfg.extra_config == {}

    def test_semaphore_initialized(self) -> None:
        agent = _make_agent(concurrency=4)
        assert agent.sem._value == 4

    def test_codex_version_is_required(self) -> None:
        # Pinning is mandatory so auto-install cannot silently drift; omitting it is a config error.
        with pytest.raises(ValidationError):
            CodexAgentConfig(
                host="0.0.0.0",
                port=8080,
                entrypoint="",
                name="",
                resources_server=ResourcesServerRef(type="resources_servers", name=""),
            )


class TestTomlDumps:
    def test_round_trips_via_tomllib(self) -> None:
        data = {
            "model": "gpt-5",
            "check_for_update_on_startup": False,
            "analytics": {"enabled": False},
            "model_providers": {"gym": {"name": "gym", "base_url": "http://x/v1", "stream_idle_timeout_ms": 600000}},
            "mcp_servers": {
                "weather": {
                    "url": "http://h:1/mcp",
                    "http_headers": {"X-NeMo-Gym-Session-Token": "tok"},
                }
            },
        }
        parsed = tomllib.loads(toml_dumps(data))
        assert parsed == data

    def test_quotes_non_bare_keys_and_escapes_strings(self) -> None:
        parsed = tomllib.loads(toml_dumps({"a b": 'quo"te\nnl', "list": ["x", "y"]}))
        assert parsed == {"a b": 'quo"te\nnl', "list": ["x", "y"]}


class TestBuildCommand:
    def test_command_shape(self) -> None:
        agent = _make_agent()
        cmd = agent._build_command("do the thing", "/work/dir")
        assert cmd[:5] == ["codex", "exec", "--json", "--ephemeral", "--skip-git-repo-check"]
        assert cmd[cmd.index("--cd") + 1] == "/work/dir"
        # instruction is the final positional after the `--` separator
        assert cmd[-2:] == ["--", "do the thing"]


class TestBuildConfig:
    def test_base_config_isolated_and_pinned_to_gym_provider(self) -> None:
        agent = _make_agent(timeout=30)
        config = agent._build_config("http://model:9000/v1")
        assert config["model_provider"] == "gym"
        assert config["approval_policy"] == "never"
        assert config["sandbox_mode"] == "danger-full-access"
        assert config["web_search"] == "disabled"
        assert config["check_for_update_on_startup"] is False
        assert config["analytics"] == {"enabled": False}
        assert config["history"] == {"persistence": "none"}
        assert config["features"] == {"multi_agent": False, "code_mode": False}
        provider = config["model_providers"]["gym"]
        assert provider["base_url"] == "http://model:9000/v1"
        assert provider["env_key"] == "OPENAI_API_KEY"
        assert provider["wire_api"] == "responses"
        # idle budget defaults to the whole-run timeout (Gym servers stream only at completion)
        assert provider["stream_idle_timeout_ms"] == 30_000
        assert "model" not in config
        assert "developer_instructions" not in config

    def test_optional_knobs_threaded_through(self) -> None:
        agent = _make_agent(model="gpt-5-codex", reasoning_effort="high", stream_idle_timeout_ms=42)
        config = agent._build_config("http://x/v1", developer_instructions="be terse")
        assert config["model"] == "gpt-5-codex"
        assert config["model_reasoning_effort"] == "high"
        assert config["developer_instructions"] == "be terse"
        assert config["model_providers"]["gym"]["stream_idle_timeout_ms"] == 42

    def test_model_server_without_model_pins_placeholder(self) -> None:
        # With a model server and no explicit model, config pins a placeholder to avoid Codex's
        # model-family code-mode gating, and the effective name is reported consistently.
        agent = _make_agent(model_server=ModelServerRef(type="responses_api_models", name="policy_model"))
        config = agent._build_config("http://x/v1")
        assert config["model"] == "gym-policy-model"
        assert agent._effective_model() == "gym-policy-model"

    def test_direct_endpoint_without_model_omits_model(self) -> None:
        # A direct endpoint with no model lets Codex pick its own default: no model key in config.
        agent = _make_agent()
        config = agent._build_config("http://x/v1")
        assert "model" not in config
        assert agent._effective_model() is None

    def test_extra_config_deep_merged(self) -> None:
        agent = _make_agent(
            extra_config={
                "features": {"web_search": True},
                "model_verbosity": "low",
                "mcp_servers": {"static": {"command": "server"}},
            }
        )
        config = agent._build_config("http://x/v1")
        # deep merge preserves base keys next to user keys
        assert config["features"] == {"multi_agent": False, "code_mode": False, "web_search": True}
        assert config["model_verbosity"] == "low"
        assert config["mcp_servers"] == {"static": {"command": "server"}}

    def test_rollout_mcp_servers_win_name_collisions(self) -> None:
        agent = _make_agent(extra_config={"mcp_servers": {"weather": {"command": "stale"}}})
        config = agent._build_config("http://x/v1", mcp_servers={"weather": {"url": "http://h:1/mcp"}})
        assert config["mcp_servers"]["weather"] == {"url": "http://h:1/mcp"}

    def test_extra_config_not_mutated_across_calls(self) -> None:
        agent = _make_agent(extra_config={"mcp_servers": {"static": {"command": "server"}}})
        agent._build_config("http://x/v1", mcp_servers={"dynamic": {"url": "http://h:1/mcp"}})
        assert agent.config.extra_config == {"mcp_servers": {"static": {"command": "server"}}}


class TestSetupCodexHome:
    def test_creates_home_with_config_toml(self, tmp_path: Path) -> None:
        agent = _make_agent()
        with patch("responses_api_agents.codex_agent.app.Path.home", return_value=tmp_path):
            codex_home = agent._setup_codex_home(agent._build_config("http://x/v1"))
        try:
            parsed = tomllib.loads((codex_home / "config.toml").read_text())
            assert parsed["model_provider"] == "gym"
            assert parsed["history"]["persistence"] == "none"
        finally:
            import shutil as _shutil

            _shutil.rmtree(codex_home, ignore_errors=True)

    def test_stages_skills_into_home(self, tmp_path: Path) -> None:
        skills_dir = _write_skill_dir(tmp_path)
        home = tmp_path / "home"
        home.mkdir()
        agent = _make_agent()
        with patch("responses_api_agents.codex_agent.app.Path.home", return_value=home):
            codex_home = agent._setup_codex_home(agent._build_config("http://x/v1"), skills_path=str(skills_dir))
        try:
            assert (codex_home / "skills" / "cot_enhanced" / "SKILL.md").is_file()
        finally:
            import shutil as _shutil

            _shutil.rmtree(codex_home, ignore_errors=True)


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
        "model": "codex-default",
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
    directly to _create_response/_run_codex."""

    def _seed_and_verify_post(self):
        async def _post(server_name, url_path, json=None, cookies=None, **kw):
            if url_path == "/verify":
                return _FakeHttpResp(
                    {"responses_create_params": {"input": []}, "response": _gym_response(), "reward": 1.0}
                )
            return _FakeHttpResp({})

        return AsyncMock(side_effect=_post)

    def _run(self, agent: CodexAgent, body: CodexAgentRunRequest, run_codex: AsyncMock):
        agent.server_client.post = self._seed_and_verify_post()
        req = MagicMock()
        req.cookies = {}
        # Stub the CLI invocation; _create_response still runs for real, so we exercise the full
        # run() -> _create_response -> _run_codex argument threading.
        with patch.object(CodexAgent, "_run_codex", run_codex):
            return asyncio.run(agent.run(req, body))

    def test_skills_ref_path_forwarded(self) -> None:
        agent = _make_agent()
        run_codex = AsyncMock(return_value=("", "codex-default"))
        body = CodexAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": []},
                SKILLS_REF_KEY_NAME: {"path": "skills/variant_a/", "hash": "abc123", "skills": []},
            }
        )

        self._run(agent, body, run_codex)

        assert run_codex.call_args.kwargs["skills_path"] == "skills/variant_a/"

    def test_no_skills_ref_forwards_none(self) -> None:
        agent = _make_agent()
        run_codex = AsyncMock(return_value=("", "codex-default"))
        body = CodexAgentRunRequest.model_validate({"responses_create_params": {"input": []}})

        self._run(agent, body, run_codex)

        assert run_codex.call_args.kwargs["skills_path"] is None


class TestRunCodex:
    def test_wires_command_env_and_cleans_up(self, tmp_path: Path) -> None:
        agent = _make_agent(openai_api_key="sk-test", system_prompt=None)  # pragma: allowlist secret
        captured: dict = {}

        class FakeProc:
            returncode = 0

            async def communicate(self):
                return (
                    b'{"type":"turn.completed","usage":{"input_tokens":3,"output_tokens":4}}\n',
                    b"",
                )

        async def fake_exec(*cmd, **kwargs):
            env = kwargs["env"]
            codex_home = env["CODEX_HOME"]
            captured["cmd"] = list(cmd)
            captured["codex_home"] = codex_home
            captured["api_key"] = env["OPENAI_API_KEY"]
            captured["stdin"] = kwargs.get("stdin")
            captured["start_new_session"] = kwargs.get("start_new_session")
            # the staged home + config must exist while the subprocess runs
            captured["config_during_run"] = tomllib.loads((Path(codex_home) / "config.toml").read_text())
            captured["cwd_exists_during_run"] = Path(cmd[cmd.index("--cd") + 1]).is_dir()
            captured["scratch_cwd"] = cmd[cmd.index("--cd") + 1]
            return FakeProc()

        with (
            patch("responses_api_agents.codex_agent.app.Path.home", return_value=tmp_path),
            patch("responses_api_agents.codex_agent.app.asyncio.create_subprocess_exec", fake_exec),
        ):
            stdout, model = asyncio.run(agent._run_codex("hello", system_prompt="be terse"))

        assert captured["cmd"][0] == "codex"
        assert captured["cmd"][-1] == "hello"
        assert captured["api_key"] == "sk-test"  # pragma: allowlist secret
        assert captured["stdin"] == asyncio.subprocess.DEVNULL
        # own process group, so a timeout kill reaps the npm shim's vendored-binary child too
        assert captured["start_new_session"] is True
        assert captured["config_during_run"]["developer_instructions"] == "be terse"
        assert captured["cwd_exists_during_run"] is True
        # per-run home and scratch cwd are removed after the run (no leakage between rollouts)
        assert not Path(captured["codex_home"]).exists()
        assert not Path(captured["scratch_cwd"]).exists()
        assert "turn.completed" in stdout
        assert model == "codex-default"

    def test_explicit_cwd_is_used_and_kept(self, tmp_path: Path) -> None:
        workdir = tmp_path / "work"
        workdir.mkdir()
        agent = _make_agent(cwd=str(workdir))

        class FakeProc:
            returncode = 0

            async def communicate(self):
                return b"", b""

        captured: dict = {}

        async def fake_exec(*cmd, **kwargs):
            captured["cwd"] = cmd[cmd.index("--cd") + 1]
            return FakeProc()

        with (
            patch("responses_api_agents.codex_agent.app.Path.home", return_value=tmp_path),
            patch("responses_api_agents.codex_agent.app.asyncio.create_subprocess_exec", fake_exec),
        ):
            asyncio.run(agent._run_codex("hello"))

        assert captured["cwd"] == str(workdir)
        assert workdir.is_dir()  # a user-provided cwd is never removed

    def test_bad_skills_path_does_not_leak_codex_home(self, tmp_path: Path) -> None:
        # stage_skills raises for a missing skills dir; the partially-created home must
        # still be cleaned up (setup happens inside the try whose finally rmtree's it).
        home = tmp_path / "home"
        home.mkdir()
        agent = _make_agent()

        with patch("responses_api_agents.codex_agent.app.Path.home", return_value=home):
            with pytest.raises(ValueError):
                asyncio.run(agent._run_codex("hello", skills_path=str(tmp_path / "does_not_exist")))

        leaked = home / ".codex_agent"
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
            patch("responses_api_agents.codex_agent.app.Path.home", return_value=tmp_path),
            patch("responses_api_agents.codex_agent.app.asyncio.create_subprocess_exec", fake_exec),
            patch("responses_api_agents.codex_agent.app.asyncio.wait_for", fake_wait_for),
        ):
            stdout, model = asyncio.run(agent._run_codex("hello"))

        assert stdout == ""
        assert killed["called"] is True
        assert model == "codex-default"


class TestRolloutMCPServers:
    def _agent_with_resources_server(self, **kwargs) -> CodexAgent:
        agent = _make_agent(
            resources_server=ResourcesServerRef(type="resources_servers", name="example_mcp_weather"), **kwargs
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
        return agent

    def test_no_metadata_returns_none(self) -> None:
        agent = self._agent_with_resources_server()
        assert agent._rollout_mcp_servers({}) is None

    def test_builds_streamable_http_entry_with_session_header(self) -> None:
        agent = self._agent_with_resources_server()
        servers = agent._rollout_mcp_servers(
            {
                "mcp": {
                    "server_name": "example_mcp_weather",
                    "url_path": "/mcp",
                    "headers": {"X-NeMo-Gym-Session-Token": "secret-token"},
                }
            }
        )
        assert servers == {
            "example_mcp_weather": {
                "url": "http://127.0.0.1:8123/mcp",
                "http_headers": {"X-NeMo-Gym-Session-Token": "secret-token"},
            }
        }

    def test_run_writes_mcp_entry_into_config(self) -> None:
        agent = self._agent_with_resources_server()

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

        async def fake_run_codex(instruction, system_prompt=None, mcp_servers=None, **kwargs):
            captured["instruction"] = instruction
            captured["mcp_servers"] = mcp_servers
            captured["config"] = agent._build_config("http://x/v1", mcp_servers=mcp_servers)
            return _item_completed(
                {"id": "item_1", "type": "agent_message", "text": "The weather in Paris is sunny and 72 F."}
            ), "codex-default"

        agent.server_client.post.side_effect = fake_post
        object.__setattr__(agent, "_run_codex", fake_run_codex)
        request = MagicMock(spec=Request)
        request.cookies = {}
        body = CodexAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="use the weather tool"),
            expected_city="Paris",
        )

        result = asyncio.run(agent.run(request, body))

        assert result.reward == 1.0
        assert captured["instruction"] == "use the weather tool"
        server = captured["config"]["mcp_servers"]["example_mcp_weather"]
        assert server["url"] == "http://127.0.0.1:8123/mcp"
        assert server["http_headers"]["X-NeMo-Gym-Session-Token"] == "tok"

    def test_run_threads_session_cookie_seed_to_verify(self) -> None:
        agent = self._agent_with_resources_server()
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

        async def fake_run_codex(instruction, system_prompt=None, mcp_servers=None, **kwargs):
            captured["mcp_servers"] = mcp_servers
            return _item_completed({"id": "item_1", "type": "agent_message", "text": "ok"}), "codex-default"

        agent.server_client.post.side_effect = fake_post
        object.__setattr__(agent, "_run_codex", fake_run_codex)
        request = MagicMock(spec=Request)
        request.cookies = {}
        body = CodexAgentRunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="use the weather tool"),
            verifier_metadata={"expected_city": "Paris"},
        )

        asyncio.run(agent.run(request, body))

        # the cookie set on /seed_session is threaded into the /verify call (same rollout session),
        # and the per-rollout token from seed metadata reaches the generated MCP config.
        assert captured["verify_cookies"] == {"session": "sess-cookie"}
        assert captured["mcp_servers"]["example_mcp_weather"]["http_headers"]["X-NeMo-Gym-Session-Token"] == "tok"


class TestRolloutCorrelation:
    """The CLI streams /v1/responses, so correlation rides on the provider base_url path prefix."""

    def _fake_proc(self):
        class FakeProc:
            returncode = 0

            async def communicate(self):
                return b'{"type":"turn.completed","usage":{"input_tokens":1,"output_tokens":1}}\n', b""

        return FakeProc()

    def _run_and_capture_base_url(self, agent, tmp_path: Path, **run_kwargs) -> str:
        captured: dict = {}

        async def fake_exec(*cmd, **kwargs):
            config = tomllib.loads((Path(kwargs["env"]["CODEX_HOME"]) / "config.toml").read_text())
            captured["base_url"] = config["model_providers"]["gym"]["base_url"]
            return self._fake_proc()

        def fake_resolve(name, rollout_id=None):
            prefix = f"/ng-rollout/{rollout_id}" if rollout_id else ""
            return f"http://model-server:9000{prefix}/v1"

        with (
            patch("responses_api_agents.codex_agent.app.Path.home", return_value=tmp_path),
            patch.object(type(agent), "resolve_model_base_url", side_effect=fake_resolve),
            patch("responses_api_agents.codex_agent.app.asyncio.create_subprocess_exec", fake_exec),
        ):
            asyncio.run(agent._run_codex("hi", **run_kwargs))
        return captured["base_url"]

    def test_base_url_correlation(self, tmp_path: Path) -> None:
        agent = _make_agent(model_server=ModelServerRef(type="responses_api_models", name="policy_model"))
        base_url = self._run_and_capture_base_url(agent, tmp_path, rollout_id="task3-roll1")
        # Codex appends /responses -> server strips /ng-rollout/<id> and keys capture by it.
        assert base_url == "http://model-server:9000/ng-rollout/task3-roll1/v1"

        # Direct endpoint (no model server): never prefixed -- it has no stripping middleware,
        # so a prefix would 404 every /v1/responses call.
        direct = _make_agent(openai_base_url="https://api.openai.com/v1")
        assert direct._resolve_call_base_url("t3-r1") == "https://api.openai.com/v1"

    def test_defaults_to_openai_when_nothing_configured(self) -> None:
        agent = _make_agent()
        assert agent._resolve_call_base_url(None) == "https://api.openai.com/v1"


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


class TestParseExecJsonl:
    def test_empty(self) -> None:
        items, usage = parse_exec_jsonl("")
        assert items == []
        assert usage["input_tokens"] == 0
        assert usage["output_tokens"] == 0

    def test_agent_message(self) -> None:
        line = _item_completed({"id": "item_1", "type": "agent_message", "text": "hello"})
        items, _ = parse_exec_jsonl(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "hello"

    def test_reasoning_prepended_to_next_message(self) -> None:
        lines = "\n".join(
            [
                _item_completed({"id": "item_1", "type": "reasoning", "text": "let me reason"}),
                _item_completed({"id": "item_2", "type": "agent_message", "text": "answer"}),
            ]
        )
        items, _ = parse_exec_jsonl(lines)
        assert len(items) == 1
        text = items[0].content[0].text
        assert "<think>\nlet me reason\n</think>" in text
        assert "answer" in text

    def test_trailing_reasoning_surfaced_as_think_message(self) -> None:
        # Some backends route the final answer through the reasoning channel (vLLM reasoning
        # parsers); a run ending on reasoning must not lose it.
        line = _item_completed({"id": "item_1", "type": "reasoning", "text": "the answer is 391"})
        items, _ = parse_exec_jsonl(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "<think>\nthe answer is 391\n</think>"

    def test_reasoning_cleared_after_message(self) -> None:
        lines = "\n".join(
            [
                _item_completed({"id": "item_1", "type": "reasoning", "text": "think"}),
                _item_completed({"id": "item_2", "type": "agent_message", "text": "msg1"}),
                _item_completed({"id": "item_3", "type": "agent_message", "text": "msg2"}),
            ]
        )
        items, _ = parse_exec_jsonl(lines)
        assert len(items) == 2
        assert "<think>" in items[0].content[0].text
        assert "<think>" not in items[1].content[0].text

    def test_command_execution_maps_to_call_and_output(self) -> None:
        line = _item_completed(
            {
                "id": "item_1",
                "type": "command_execution",
                "command": "/bin/bash -lc ls",
                "aggregated_output": "file.txt\n",
                "exit_code": 0,
                "status": "completed",
            }
        )
        items, _ = parse_exec_jsonl(line)
        assert len(items) == 2
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "exec_command"
        assert json.loads(items[0].arguments) == {"cmd": "/bin/bash -lc ls"}
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert "file.txt" in items[1].output
        assert items[0].call_id == items[1].call_id == "item_1"

    def test_command_execution_nonzero_exit_annotated(self) -> None:
        line = _item_completed(
            {
                "id": "item_1",
                "type": "command_execution",
                "command": "false",
                "aggregated_output": "",
                "exit_code": 1,
                "status": "failed",
            }
        )
        items, _ = parse_exec_jsonl(line)
        assert "[exit code: 1]" in items[1].output

    def test_mcp_tool_call_maps_result_text(self) -> None:
        line = _item_completed(
            {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "gymweather",
                "tool": "get_weather",
                "arguments": {"city": "Paris"},
                "result": {"content": [{"type": "text", "text": "sunny, 72F"}]},
                "error": None,
                "status": "completed",
            }
        )
        items, _ = parse_exec_jsonl(line)
        assert items[0].name == "get_weather"
        assert json.loads(items[0].arguments) == {"city": "Paris"}
        assert items[1].output == "sunny, 72F"

    def test_mcp_tool_call_error_surfaced(self) -> None:
        line = _item_completed(
            {
                "id": "item_1",
                "type": "mcp_tool_call",
                "server": "s",
                "tool": "t",
                "arguments": {},
                "result": None,
                "error": "boom",
                "status": "failed",
            }
        )
        items, _ = parse_exec_jsonl(line)
        assert items[1].output == "error: boom"

    def test_message_then_command(self) -> None:
        lines = "\n".join(
            [
                _item_completed({"id": "item_1", "type": "agent_message", "text": "running ls"}),
                _item_completed(
                    {
                        "id": "item_2",
                        "type": "command_execution",
                        "command": "ls",
                        "aggregated_output": "x",
                        "exit_code": 0,
                        "status": "completed",
                    }
                ),
            ]
        )
        items, _ = parse_exec_jsonl(lines)
        assert [type(i).__name__ for i in items] == [
            "NeMoGymResponseOutputMessage",
            "NeMoGymResponseFunctionToolCall",
            "NeMoGymFunctionCallOutput",
        ]

    def test_malformed_lines_skipped(self) -> None:
        good = _item_completed({"id": "item_1", "type": "agent_message", "text": "ok"})
        items, _ = parse_exec_jsonl(f"not-json\n{good}\n{{bad")
        assert len(items) == 1

    def test_turn_completed_accumulates_usage(self) -> None:
        line = _event(
            "turn.completed",
            usage={"input_tokens": 100, "cached_input_tokens": 40, "output_tokens": 50, "reasoning_output_tokens": 5},
        )
        _, usage = parse_exec_jsonl(line)
        assert usage["input_tokens"] == 100
        assert usage["cached_input_tokens"] == 40
        assert usage["output_tokens"] == 50
        assert usage["reasoning_tokens"] == 5

    def test_errors_collected(self) -> None:
        lines = "\n".join(
            [
                _item_completed({"id": "item_0", "type": "error", "message": "model metadata missing"}),
                _event("turn.failed", error={"message": "stream disconnected"}),
            ]
        )
        items, usage = parse_exec_jsonl(lines)
        assert items == []
        assert usage["errors"] == ["model metadata missing", "stream disconnected"]

    def test_item_started_events_ignored(self) -> None:
        lines = "\n".join(
            [
                _event(
                    "item.started",
                    item={"id": "item_1", "type": "command_execution", "command": "ls", "status": "in_progress"},
                ),
                _item_completed(
                    {
                        "id": "item_1",
                        "type": "command_execution",
                        "command": "ls",
                        "aggregated_output": "x",
                        "exit_code": 0,
                        "status": "completed",
                    }
                ),
            ]
        )
        items, _ = parse_exec_jsonl(lines)
        assert len(items) == 2  # one call + one output, not doubled


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "codex_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "codex_agent" in data
        inner = data["codex_agent"]["responses_api_agents"]["codex_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 32
        assert inner["sandbox_mode"] == "danger-full-access"

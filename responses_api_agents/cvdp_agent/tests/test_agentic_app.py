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
import shlex
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import yaml

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.claude_code_agent.app import (
    ClaudeCodeAgentVerifyResponse,
    ResourcesServerRef,
)
from responses_api_agents.cvdp_agent import agentic_app
from responses_api_agents.cvdp_agent.agentic_app import (
    CvdpAgent,
    CvdpAgentConfig,
    _is_harness_path,
    _safe_workspace_path,
    _summarize_claude_failure,
)


def _config(**kwargs) -> CvdpAgentConfig:
    return CvdpAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="cvdp_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="cvdp"),
        **kwargs,
    )


def _make_agent(provider=None, **kwargs) -> CvdpAgent:
    with patch("responses_api_agents.cvdp_agent.agentic_app.CvdpAgent.model_post_init"):
        agent = CvdpAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    agent._provider = provider
    agent._node_bind_dir = "/node" if provider is not None else None
    agent._provider_guard = asyncio.Lock()
    agent._sif_locks = {}
    agent._sif_lock_guard = asyncio.Lock()
    agent._sif_cache_dir = "/tmp/sif"
    return agent


class FakeHandle:
    def __init__(self, staging_dir: Path) -> None:
        self.sandbox_id = "nemo-gym-test"
        self.provider_name = "apptainer"
        self.raw = SimpleNamespace(staging_dir=staging_dir, name="nemo-gym-test", mount_point="/code", image="x.sif")


class FakeProvider:
    def __init__(self, staging_dir: Path, exec_result: SandboxExecResult, on_exec=None) -> None:
        self._staging = staging_dir
        self._exec_result = exec_result
        self._on_exec = on_exec
        self.created = False
        self.closed = False
        self.exec_calls: list[dict] = []

    async def create(self, spec):
        self.created = True
        self.spec = spec
        return FakeHandle(self._staging)

    async def exec(self, handle, command, *, env=None, stdin=None, timeout_s=None, **kwargs):
        self.exec_calls.append({"command": command, "env": env, "stdin": stdin, "timeout_s": timeout_s})
        if self._on_exec is not None:
            self._on_exec(handle)
        return self._exec_result

    async def close(self, handle):
        self.closed = True


def _assistant_line(text: str) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "message": {
                "content": [{"type": "text", "text": text}],
                "usage": {"input_tokens": 7, "output_tokens": 3},
            },
        }
    )


def _result_line(num_turns: int) -> str:
    return json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "num_turns": num_turns,
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }
    )


class TestConfig:
    def test_sandbox_defaults(self) -> None:
        cfg = _config()
        assert cfg.sim_image == "nvidia/cvdp-sim:v1.0.0"
        assert cfg.container_workdir == "/code"
        assert cfg.sif_path is None
        assert cfg.max_context_tokens == 1_000_000

    def test_inherits_claude_defaults(self) -> None:
        cfg = _config()
        assert cfg.bare is True
        assert cfg.max_turns == 30


class TestBuildClaudeArgs:
    def test_no_positional_prompt_and_bare(self) -> None:
        agent = _make_agent()
        args = agent._build_claude_args("m", None)
        assert "--bare" in args
        assert "--" not in args  # prompt is fed via stdin, never as a positional
        assert args[args.index("--model") + 1] == "m"
        assert args[args.index("--max-turns") + 1] == "30"

    def test_optional_flags(self) -> None:
        agent = _make_agent(allowed_tools="Bash", thinking="enabled", max_thinking_tokens=64)
        args = agent._build_claude_args("m", "be terse")
        assert args[args.index("--append-system-prompt") + 1] == "be terse"
        assert args[args.index("--allowedTools") + 1] == "Bash"
        assert args[args.index("--thinking") + 1] == "enabled"
        assert args[args.index("--max-thinking-tokens") + 1] == "64"


class TestContainerEnv:
    def test_home_not_set_and_paths(self) -> None:
        agent = _make_agent()
        env = agent._container_env("m", "", "key")
        # HOME must NOT be set via env (apptainer rejects it; it's exported in-shell).
        assert "HOME" not in env
        assert env["CLAUDE_CONFIG_DIR"] == f"{agentic_app._CONTAINER_STATE_DIR}/.claude_config"
        assert env["PATH"].startswith("/opt/claude_node/bin:")
        assert env["IS_SANDBOX"] == "1"
        assert env["CLAUDE_CODE_MAX_CONTEXT_TOKENS"] == "1000000"
        assert env["DISABLE_AUTO_COMPACT"] == "1"

    def test_base_url_branch(self) -> None:
        agent = _make_agent()
        env = agent._container_env("m", "http://host:9000", "key")
        assert env["ANTHROPIC_BASE_URL"] == "http://host:9000"
        assert env["ANTHROPIC_AUTH_TOKEN"] == "key"

    def test_max_context_none_omitted(self) -> None:
        agent = _make_agent(max_context_tokens=None)
        env = agent._container_env("m", "", "key")
        assert "CLAUDE_CODE_MAX_CONTEXT_TOKENS" not in env
        assert "DISABLE_AUTO_COMPACT" not in env


class TestSeedWorkspace:
    def test_seeds_context_only(self, tmp_path: Path) -> None:
        agent = _make_agent()
        agent._seed_workspace(
            tmp_path,
            {"rtl/a.sv": "module a;endmodule", "docs/spec.md": "# spec"},
            harness_files=None,
        )
        assert (tmp_path / "rtl" / "a.sv").read_text() == "module a;endmodule"
        assert (tmp_path / "docs" / "spec.md").read_text() == "# spec"
        # standard layout dirs created
        for d in ("rtl", "verif", "docs", "src", "rundir"):
            assert (tmp_path / d).is_dir()

    def test_skips_harness_and_unsafe(self, tmp_path: Path) -> None:
        agent = _make_agent()
        agent._seed_workspace(
            tmp_path,
            {
                "src/test_a.py": "secret",
                "docker-compose.yml": "secret",
                "../escape.sv": "secret",
                "verif/declared.sv": "ok",
            },
            harness_files={"verif/declared.sv": "..."},
        )
        assert not (tmp_path / "src" / "test_a.py").exists()
        assert not (tmp_path / "docker-compose.yml").exists()
        assert not (tmp_path.parent / "escape.sv").exists()
        # declared as a harness file -> skipped even though it's an HDL path
        assert not (tmp_path / "verif" / "declared.sv").exists()


class TestCollectProducedFiles:
    def test_collects_new_and_modified(self, tmp_path: Path) -> None:
        agent = _make_agent()
        (tmp_path / "rtl").mkdir()
        (tmp_path / "verif").mkdir()
        (tmp_path / "rtl" / "new.sv").write_text("module new;endmodule")
        (tmp_path / "rtl" / "same.sv").write_text("unchanged")
        (tmp_path / "rtl" / "build.out").write_text("artifact")
        context = {"rtl/same.sv": "unchanged"}
        produced = agent._collect_produced_files(tmp_path, context, target_files=[])
        assert "rtl/new.sv" in produced
        assert "rtl/same.sv" not in produced  # unchanged context file
        assert "rtl/build.out" not in produced  # not an HDL extension

    def test_includes_declared_targets(self, tmp_path: Path) -> None:
        agent = _make_agent()
        (tmp_path / "rtl").mkdir()
        (tmp_path / "rtl" / "target.sv").write_text("module t;endmodule")
        produced = agent._collect_produced_files(tmp_path, {}, target_files=["rtl/target.sv"])
        assert produced["rtl/target.sv"] == "module t;endmodule"


class TestRunClaudeInSandbox:
    def test_command_exports_home_and_feeds_stdin(self, tmp_path: Path) -> None:
        result = SandboxExecResult(stdout=_assistant_line("done"), stderr=None, return_code=0)
        provider = FakeProvider(tmp_path, result)
        agent = _make_agent(provider=provider)
        handle = FakeHandle(tmp_path)

        stdout, model = asyncio.run(agent._run_claude_in_sandbox(handle, "the prompt", "be terse"))

        assert model == "claude-sonnet-4-6"
        assert "done" in stdout
        call = provider.exec_calls[0]
        # prompt is fed via stdin, not the command
        assert call["stdin"] == b"the prompt"
        # HOME points outside the task workspace so Claude's state doesn't pollute it.
        assert call["command"].startswith(f"export HOME={shlex.quote(agentic_app._CONTAINER_STATE_DIR)}")
        assert "cd /code" in call["command"]
        assert "/opt/claude_node/bin/claude" in call["command"]
        # settings are seeded in-shell into the config dir (outside the workspace),
        # not written into the bound workspace on the host.
        assert f"{agentic_app._CONTAINER_STATE_DIR}/.claude_config" in call["command"]
        assert "settings.json" in call["command"]
        assert not (tmp_path / ".claude_config").exists()

    def test_timeout_returns_empty(self, tmp_path: Path) -> None:
        result = SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout")
        provider = FakeProvider(tmp_path, result)
        agent = _make_agent(provider=provider)
        handle = FakeHandle(tmp_path)
        stdout, model = asyncio.run(agent._run_claude_in_sandbox(handle, "p", None))
        assert stdout == ""
        assert model == "claude-sonnet-4-6"

    def test_command_tees_stream_when_log_dir_given(self, tmp_path: Path) -> None:
        result = SandboxExecResult(stdout=_assistant_line("done"), stderr=None, return_code=0)
        provider = FakeProvider(tmp_path, result)
        agent = _make_agent(provider=provider)
        handle = FakeHandle(tmp_path)
        asyncio.run(agent._run_claude_in_sandbox(handle, "p", None, log_dir=tmp_path / "log"))
        cmd = provider.exec_calls[0]["command"]
        assert "tee" in cmd
        assert agentic_app._CONTAINER_LOG_DIR in cmd

    def test_timeout_recovers_partial_trajectory_from_log(self, tmp_path: Path) -> None:
        # The provider drops pipe output on timeout, but the tee'd file survives,
        # so the partial trajectory is recovered and returned as stdout.
        log_dir = tmp_path / "log"
        log_dir.mkdir()
        (log_dir / agentic_app._STREAM_LOG_NAME).write_text(_assistant_line("partial work before kill"))
        result = SandboxExecResult(stdout=None, stderr="timed out", return_code=125, error_type="timeout")
        provider = FakeProvider(tmp_path, result)
        agent = _make_agent(provider=provider)
        handle = FakeHandle(tmp_path)
        stdout, model = asyncio.run(agent._run_claude_in_sandbox(handle, "p", None, log_dir=log_dir))
        assert "partial work before kill" in stdout


class TestRunSandboxed:
    def test_full_flow_sends_rtl_files(self, tmp_path: Path) -> None:
        def writer(handle):
            (handle.raw.staging_dir / "rtl" / "target.sv").write_text("module target;endmodule")

        result = SandboxExecResult(stdout=_assistant_line("edited the file"), stderr=None, return_code=0)
        provider = FakeProvider(tmp_path, result, on_exec=writer)
        agent = _make_agent(provider=provider, sif_path="/cached/x.sif")

        body = MagicMock()
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input="solve it")
        body.model_dump.return_value = {"responses_create_params": {"input": "solve it"}}

        request = SimpleNamespace(cookies={})
        meta = {"target_files": ["rtl/target.sv"], "context_files": {}}

        captured = {}

        async def fake_post(*, server_name, url_path, json, cookies):
            captured["url_path"] = url_path
            captured["json"] = json
            return SimpleNamespace()

        agent.server_client.post = AsyncMock(side_effect=fake_post)

        with (
            patch("responses_api_agents.cvdp_agent.agentic_app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.cvdp_agent.agentic_app.get_response_json",
                AsyncMock(return_value={"reward": 1.0}),
            ),
            patch.object(ClaudeCodeAgentVerifyResponse, "model_validate", side_effect=lambda d: SimpleNamespace(**d)),
        ):
            resp = asyncio.run(agent._run_sandboxed(request, body, meta, ["rtl/target.sv"]))

        assert provider.created and provider.closed
        assert captured["url_path"] == "/verify"
        assert "rtl/target.sv" in captured["json"]["rtl_files"]
        assert resp.reward == 1.0
        assert resp.turns_used == 1
        assert resp.finished_naturally is True

    def test_turns_used_prefers_num_turns_from_result_event(self, tmp_path: Path) -> None:
        # One assistant text message but the result event reports 7 turns: the
        # authoritative num_turns must win over the message count (which is 1).
        def writer(handle):
            (handle.raw.staging_dir / "rtl" / "target.sv").write_text("module target;endmodule")

        stdout = _assistant_line("did several tool-only turns") + "\n" + _result_line(7)
        result = SandboxExecResult(stdout=stdout, stderr=None, return_code=0)
        provider = FakeProvider(tmp_path, result, on_exec=writer)
        agent = _make_agent(provider=provider, sif_path="/cached/x.sif")

        body = MagicMock()
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input="solve it")
        body.model_dump.return_value = {"responses_create_params": {"input": "solve it"}}
        request = SimpleNamespace(cookies={})
        meta = {"target_files": ["rtl/target.sv"], "context_files": {}}

        async def fake_post(*, server_name, url_path, json, cookies):
            return SimpleNamespace()

        agent.server_client.post = AsyncMock(side_effect=fake_post)

        with (
            patch("responses_api_agents.cvdp_agent.agentic_app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.cvdp_agent.agentic_app.get_response_json",
                AsyncMock(return_value={"reward": 1.0}),
            ),
            patch.object(ClaudeCodeAgentVerifyResponse, "model_validate", side_effect=lambda d: SimpleNamespace(**d)),
        ):
            resp = asyncio.run(agent._run_sandboxed(request, body, meta, ["rtl/target.sv"]))

        assert resp.turns_used == 7

    def test_binds_log_dir_and_cleans_up(self, tmp_path: Path) -> None:
        def writer(handle):
            (handle.raw.staging_dir / "rtl" / "target.sv").write_text("module target;endmodule")

        result = SandboxExecResult(stdout=_assistant_line("ok"), stderr=None, return_code=0)
        provider = FakeProvider(tmp_path, result, on_exec=writer)
        agent = _make_agent(provider=provider, sif_path="/cached/x.sif")

        body = MagicMock()
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input="solve it")
        body.model_dump.return_value = {"responses_create_params": {"input": "solve it"}}
        request = SimpleNamespace(cookies={})
        meta = {"target_files": ["rtl/target.sv"], "context_files": {}}

        async def fake_post(*, server_name, url_path, json, cookies):
            return SimpleNamespace()

        agent.server_client.post = AsyncMock(side_effect=fake_post)

        with (
            patch("responses_api_agents.cvdp_agent.agentic_app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.cvdp_agent.agentic_app.get_response_json",
                AsyncMock(return_value={"reward": 1.0}),
            ),
            patch.object(ClaudeCodeAgentVerifyResponse, "model_validate", side_effect=lambda d: SimpleNamespace(**d)),
        ):
            asyncio.run(agent._run_sandboxed(request, body, meta, ["rtl/target.sv"]))

        binds = provider.spec.provider_options["binds"]
        assert len(binds) == 1
        host_dir, container_dir = binds[0].split(":")
        assert container_dir == agentic_app._CONTAINER_LOG_DIR
        # temp log dir is removed once its contents have been folded into the response
        assert not Path(host_dir).exists()

    def test_no_produced_files_omits_rtl_files(self, tmp_path: Path) -> None:
        result = SandboxExecResult(stdout=_assistant_line("nothing written"), stderr=None, return_code=0)
        provider = FakeProvider(tmp_path, result)  # no writer -> no files produced
        agent = _make_agent(provider=provider, sif_path="/cached/x.sif")

        body = MagicMock()
        body.responses_create_params = NeMoGymResponseCreateParamsNonStreaming(input="solve it")
        body.model_dump.return_value = {"responses_create_params": {"input": "solve it"}}
        request = SimpleNamespace(cookies={})
        meta = {"target_files": ["rtl/target.sv"], "context_files": {}}

        captured = {}

        async def fake_post(*, server_name, url_path, json, cookies):
            captured["json"] = json
            return SimpleNamespace()

        agent.server_client.post = AsyncMock(side_effect=fake_post)

        with (
            patch("responses_api_agents.cvdp_agent.agentic_app.raise_for_status", AsyncMock()),
            patch(
                "responses_api_agents.cvdp_agent.agentic_app.get_response_json",
                AsyncMock(return_value={"reward": 0.0}),
            ),
            patch.object(ClaudeCodeAgentVerifyResponse, "model_validate", side_effect=lambda d: SimpleNamespace(**d)),
        ):
            resp = asyncio.run(agent._run_sandboxed(request, body, meta, ["rtl/target.sv"]))

        assert "rtl_files" not in captured["json"]
        assert resp.finished_naturally is False
        assert provider.closed


class TestRunDispatch:
    def test_no_target_files_uses_host_path(self) -> None:
        agent = _make_agent()
        body = MagicMock()
        body.model_extra = {"verifier_metadata": {}}
        request = SimpleNamespace(cookies={})

        host_run = AsyncMock(return_value="host-result")
        with patch("responses_api_agents.claude_code_agent.app.ClaudeCodeAgent.run", host_run):
            result = asyncio.run(agent.run(request, body))

        assert result == "host-result"
        host_run.assert_awaited_once()

    def test_target_files_uses_sandbox_path(self) -> None:
        agent = _make_agent()
        body = MagicMock()
        body.model_extra = {"verifier_metadata": {"target_files": ["rtl/a.sv"]}}
        request = SimpleNamespace(cookies={})

        agent._run_sandboxed = AsyncMock(return_value="sandbox-result")
        result = asyncio.run(agent.run(request, body))

        assert result == "sandbox-result"
        agent._run_sandboxed.assert_awaited_once()


class TestSummarizeFailure:
    def test_collects_retries_and_result(self) -> None:
        stdout = "\n".join(
            [
                json.dumps({"type": "system", "subtype": "api_retry", "error_status": 529}),
                json.dumps({"type": "system", "subtype": "api_retry", "error_status": 529}),
                json.dumps({"type": "result", "subtype": "error_max_turns", "is_error": True}),
            ]
        )
        summary = _summarize_claude_failure(stdout)
        assert "api_retry=529 x2" in summary
        assert "error_max_turns" in summary

    def test_empty(self) -> None:
        assert _summarize_claude_failure("not-json\n") == ""


class TestHelpers:
    def test_is_harness_path(self) -> None:
        assert _is_harness_path("src/test.py")
        assert _is_harness_path("docker-compose.yml")
        assert not _is_harness_path("rtl/a.sv")

    def test_safe_workspace_path_rejects_escape(self, tmp_path: Path) -> None:
        assert _safe_workspace_path(tmp_path, "../x") is None
        assert _safe_workspace_path(tmp_path, "") is None
        assert _safe_workspace_path(tmp_path, "rtl/a.sv") == (tmp_path / "rtl" / "a.sv").resolve()


class TestConfigYaml:
    def test_module_compiles(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "agentic_app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "cvdp_agent_agentic.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        # Top-level key is the instance name; nested key is the server *directory*
        # (responses_api_agents/cvdp_agent), which holds agentic_app.py.
        inner = data["cvdp_agent_agentic"]["responses_api_agents"]["cvdp_agent"]
        assert inner["entrypoint"] == "agentic_app.py"
        assert inner["container_workdir"] == "/code"

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
import signal
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
import yaml

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseReasoningItem,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.prime_agent.app import (
    PrimeAgent,
    PrimeAgentConfig,
    ResourcesServerRef,
    _extract_instruction,
    _kill_prime_processes,
    _process_groups_with_env,
    parse_prime_agent_events,
)
from responses_api_agents.prime_agent.setup_prime_agent import ensure_prime_agent


def _config(**kwargs) -> PrimeAgentConfig:
    return PrimeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> PrimeAgent:
    with patch("responses_api_agents.prime_agent.app.PrimeAgent.model_post_init"):
        agent = PrimeAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _message_end(role, content, **extra) -> str:
    return json.dumps({"type": "message_end", "message": {"role": role, "content": content, **extra}})


class TestSanity:
    def test_config_defaults(self) -> None:
        config = _config()
        assert config.concurrency == 8
        assert config.command == "prime-agent"
        assert config.command_parts == ["prime-agent"]
        assert config.kernel_venv == "outputs/prime_agent/kernel-venv"

    def test_semaphore_initialized(self) -> None:
        agent = _make_agent(concurrency=4)
        assert agent.sem._value == 4

    def test_model_post_init_checks_pinned_version(self) -> None:
        agent = _make_agent(prime_agent_version="0.7.0")
        with (
            patch("responses_api_agents.prime_agent.app.ensure_prime_agent") as ensure,
            patch("responses_api_agents.prime_agent.app.shutil.which", return_value="/bin/prime-agent"),
        ):
            agent.model_post_init(None)

        ensure.assert_called_once_with("0.7.0")


class TestExtractInstruction:
    def test_user_only(self) -> None:
        user, system = _extract_instruction([NeMoGymEasyInputMessage(role="user", content="hello")])
        assert user == "hello"
        assert system is None

    def test_system_developer_and_user(self) -> None:
        items = [
            NeMoGymEasyInputMessage(role="system", content="be concise"),
            NeMoGymEasyInputMessage(role="developer", content="use tools"),
            NeMoGymEasyInputMessage(role="user", content="hi"),
        ]
        user, system = _extract_instruction(items)
        assert user == "hi"
        assert system == "be concise\n\nuse tools"

    def test_empty(self) -> None:
        user, system = _extract_instruction([])
        assert user == ""
        assert system is None


class TestParsePrimeAgentEvents:
    def test_empty(self) -> None:
        items, usage = parse_prime_agent_events("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    def test_assistant_text_and_usage(self) -> None:
        line = _message_end(
            "assistant",
            [{"type": "text", "text": "the answer is 4"}],
            usage={"input": 100, "output": 20, "cacheRead": 5},
        )
        items, usage = parse_prime_agent_events(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"
        assert usage == {"input_tokens": 105, "output_tokens": 20, "cached_tokens": 5}

    def test_user_and_non_terminal_events_ignored(self) -> None:
        user = _message_end("user", [{"type": "text", "text": "hi"}])
        update = json.dumps({"type": "message_update", "message": {"role": "assistant", "content": []}})
        assert parse_prime_agent_events(f"{user}\n{update}")[0] == []

    def test_ipython_call_and_result(self) -> None:
        lines = "\n".join(
            [
                _message_end(
                    "assistant",
                    [{"type": "toolCall", "id": "c1", "name": "ipython", "arguments": {"code": "6 * 7"}}],
                ),
                _message_end("toolResult", [{"type": "text", "text": "42"}], toolCallId="c1", isError=True),
                _message_end("assistant", [{"type": "text", "text": "\\boxed{42}"}]),
            ]
        )
        items, _ = parse_prime_agent_events(lines)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "ipython"
        assert json.loads(items[0].arguments) == {"code": "6 * 7"}
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert items[1].output == "42"
        assert items[1].status == "incomplete"
        assert isinstance(items[2], NeMoGymResponseOutputMessage)

    def test_malformed_lines_skipped(self) -> None:
        line = "not-json\n" + _message_end("assistant", [{"type": "text", "text": "ok"}])
        items, _ = parse_prime_agent_events(line)
        assert len(items) == 1

    def test_agent_error_is_masked(self) -> None:
        line = _message_end("assistant", [], stopReason="error", errorMessage="model failed")
        items, usage = parse_prime_agent_events(line)
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    def test_reasoning_is_preserved_separately(self) -> None:
        line = _message_end(
            "assistant",
            [
                {"type": "thinking", "thinking": "check the arithmetic"},
                {"type": "text", "text": "the answer is 4"},
            ],
            usage={"input": 100, "output": 20},
        )
        items, _ = parse_prime_agent_events(line)
        assert len(items) == 2
        assert isinstance(items[0], NeMoGymResponseReasoningItem)
        assert items[0].summary[0].text == "check the arithmetic"
        assert isinstance(items[1], NeMoGymResponseOutputMessage)
        assert items[1].content[0].text == "the answer is 4"


class TestResponses:
    async def test_agent_failure_returns_empty_assistant(self) -> None:
        agent = _make_agent()
        request = MagicMock()
        request.path_params = {}
        result = ([], {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}, "test-model", False)
        with patch.object(PrimeAgent, "_run_prime_agent", new=AsyncMock(return_value=result)):
            response = await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="hello"))

        assert len(response.output) == 1
        assert response.output[0].content[0].text == ""

    async def test_timeout_preserves_partial_output_and_sets_metadata(self) -> None:
        agent = _make_agent()
        request = MagicMock()
        request.path_params = {}
        output = [
            NeMoGymResponseOutputMessage(
                id="partial",
                content=[],
                role="assistant",
                status="completed",
                type="message",
            )
        ]
        usage = {"input_tokens": 7, "output_tokens": 3, "cached_tokens": 2}
        result = (output, usage, "test-model", True)
        with patch.object(PrimeAgent, "_run_prime_agent", new=AsyncMock(return_value=result)):
            response = await agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="hello"))

        assert response.output == output
        assert response.metadata["prime_agent_timed_out"] == "true"
        assert response.usage.input_tokens == 7
        assert response.usage.input_tokens_details.cached_tokens == 2
        assert response.usage.output_tokens == 3


class TestRunPrimeAgent:
    def test_process_groups_with_env_finds_matching_processes(self, tmp_path: Path) -> None:
        for pid, environ in {
            "117": b"HOME=/tmp\0PRIME_AGENT_CODING_AGENT_DIR=/tmp/agent\0",  # pragma: allowlist secret
            "298": b"PRIME_AGENT_CODING_AGENT_DIR=/tmp/other\0",
        }.items():
            process_dir = tmp_path / pid
            process_dir.mkdir()
            (process_dir / "environ").write_bytes(environ)

        with patch("responses_api_agents.prime_agent.app.os.getpgid", return_value=117):
            groups = _process_groups_with_env("PRIME_AGENT_CODING_AGENT_DIR", "/tmp/agent", tmp_path)

        assert groups == [117]

    def test_kill_prime_processes_includes_daemonized_processes(self) -> None:
        with (
            patch("responses_api_agents.prime_agent.app._descendant_pids", return_value=[298, 117]),
            patch("responses_api_agents.prime_agent.app.os.getpgid", side_effect={298: 117, 117: 117}.get),
            patch("responses_api_agents.prime_agent.app._process_groups_with_env", return_value=[117, 412]),
            patch("responses_api_agents.prime_agent.app.os.killpg") as killpg,
        ):
            _kill_prime_processes(99, Path("/tmp/agent"))

        assert killpg.call_args_list == [
            call(117, signal.SIGKILL),
            call(412, signal.SIGKILL),
            call(99, signal.SIGKILL),
        ]

    async def test_timeout_does_not_cancel_stdout_collection(self, tmp_path: Path) -> None:
        agent = _make_agent(timeout=0)
        release = asyncio.Event()
        state = {"cancelled": False}
        stdout = _message_end(
            "assistant",
            [{"type": "text", "text": "partial answer"}],
            usage={"input": 7, "output": 3},
        ).encode()

        async def communicate() -> tuple[bytes, bytes]:
            try:
                await release.wait()
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
            return stdout, b""

        async def wait() -> int:
            await release.wait()
            return -signal.SIGKILL

        proc = MagicMock()
        proc.pid = 123
        proc.returncode = None
        proc.communicate = AsyncMock(side_effect=communicate)
        proc.wait = AsyncMock(side_effect=wait)

        with (
            patch.object(agent, "_workspace_root", return_value=tmp_path / "work"),
            patch(
                "responses_api_agents.prime_agent.app._kill_prime_processes",
                side_effect=lambda *_: release.set(),
            ) as kill_processes,
            patch(
                "responses_api_agents.prime_agent.app.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)
            ) as create_subprocess,
        ):
            output, usage, model, timed_out = await agent._run_prime_agent("solve it", None, None)

        assert state["cancelled"] is False
        assert proc.communicate.await_count == 1
        kill_processes.assert_called_once_with(proc.pid, tmp_path / "work/.prime-home/.prime/agent")
        assert create_subprocess.await_args.kwargs["start_new_session"] is True
        command = list(create_subprocess.await_args.args)
        socket = Path(command[command.index("--daemon-socket") + 1])
        assert socket.parent.parent == Path("/tmp")
        assert not socket.parent.exists()
        assert output[0].content[0].text == "partial answer"
        assert usage == {"input_tokens": 7, "output_tokens": 3, "cached_tokens": 0}
        assert model == agent.config.model
        assert timed_out is True

    async def test_client_exit_reaps_daemon_without_marking_timeout(self, tmp_path: Path) -> None:
        agent = _make_agent(timeout=60)
        release = asyncio.Event()
        stdout = _message_end("assistant", [{"type": "text", "text": "complete answer"}]).encode()

        async def communicate() -> tuple[bytes, bytes]:
            await release.wait()
            return stdout, b""

        async def wait() -> int:
            proc.returncode = 0
            return 0

        proc = MagicMock()
        proc.pid = 123
        proc.returncode = None
        proc.communicate = AsyncMock(side_effect=communicate)
        proc.wait = AsyncMock(side_effect=wait)

        with (
            patch.object(agent, "_workspace_root", return_value=tmp_path / "work"),
            patch(
                "responses_api_agents.prime_agent.app._kill_prime_processes",
                side_effect=lambda *_: release.set(),
            ) as kill_processes,
            patch(
                "responses_api_agents.prime_agent.app.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)
            ),
        ):
            output, _, _, timed_out = await agent._run_prime_agent("solve it", None, None)

        kill_processes.assert_called_once_with(proc.pid, tmp_path / "work/.prime-home/.prime/agent")
        assert output[0].content[0].text == "complete answer"
        assert timed_out is False

    async def test_external_cancellation_kills_process_and_drains_output(self, tmp_path: Path) -> None:
        agent = _make_agent(timeout=60)
        started = asyncio.Event()
        release = asyncio.Event()
        state = {"cancelled": False}

        async def communicate() -> tuple[bytes, bytes]:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                state["cancelled"] = True
                raise
            return b"", b""

        async def wait() -> int:
            await release.wait()
            return -signal.SIGKILL

        proc = MagicMock()
        proc.pid = 123
        proc.returncode = None
        proc.communicate = AsyncMock(side_effect=communicate)
        proc.wait = AsyncMock(side_effect=wait)

        with (
            patch.object(agent, "_workspace_root", return_value=tmp_path / "work"),
            patch(
                "responses_api_agents.prime_agent.app._kill_prime_processes",
                side_effect=lambda *_: release.set(),
            ) as kill_processes,
            patch(
                "responses_api_agents.prime_agent.app.asyncio.create_subprocess_exec", new=AsyncMock(return_value=proc)
            ) as create_subprocess,
        ):
            task = asyncio.create_task(agent._run_prime_agent("solve it", None, None))
            await started.wait()
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                raise AssertionError("expected Prime Agent run to be cancelled")

        assert state["cancelled"] is False
        assert proc.communicate.await_count == 1
        kill_processes.assert_called_once_with(proc.pid, tmp_path / "work/.prime-home/.prime/agent")
        assert create_subprocess.await_args.kwargs["start_new_session"] is True


class TestEnvironmentAndCommand:
    def test_env_isolates_config_and_shares_kernel(self, tmp_path: Path) -> None:
        kernel_venv = tmp_path / "kernel"
        agent = _make_agent(kernel_venv=str(kernel_venv), env={"NVIDIA_API_KEY": "k", "EMPTY": ""})
        home = tmp_path / "home"
        env = agent._env(home)
        assert env["HOME"] == str(home)
        assert env["PRIME_AGENT_CODING_AGENT_DIR"] == str(home / ".prime" / "agent")
        assert env["PRIME_AGENT_KERNEL_VENV"] == str(kernel_venv)
        assert "PRIME_AGENT_INTERNAL_LEGACY_OWNED_WORKER_FRONTEND" not in env
        assert env["PI_SKIP_VERSION_CHECK"] == "1"
        assert env["NVIDIA_API_KEY"] == "k"
        assert "EMPTY" not in env

    def test_command_uses_provider_model_system_prompt_and_private_daemon(self, tmp_path: Path) -> None:
        agent = _make_agent(
            model="policy/test-model",
            thinking="high",
            extra_args=["--no-skills"],
        )
        socket = tmp_path / "daemon.sock"
        command = agent._build_command("solve it", "be exact", socket)
        assert command[:6] == ["prime-agent", "--print", "--mode", "json", "--no-session", "--daemon-socket"]
        assert command[-3:] == ["be exact", "--no-skills", "solve it"]
        assert command[command.index("--daemon-socket") + 1] == str(socket)
        assert command[command.index("--model") + 1] == "test-model"
        assert command[command.index("--thinking") + 1] == "high"

    def test_command_accepts_unqualified_model(self) -> None:
        agent = _make_agent(model="test-model")
        command = agent._build_command("solve it", None)
        assert "--provider" not in command
        assert command[command.index("--model") + 1] == "test-model"


class TestModelServer:
    def test_builds_prime_agent_provider_config(self) -> None:
        agent = _make_agent(
            model="Qwen3.6-35B-A3B",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        with patch.object(PrimeAgent, "resolve_model_base_url", return_value="http://model/v1"):
            config = agent._build_models_config()

        provider = config["providers"]["nemo"]
        assert agent._effective_model() == "nemo/Qwen3.6-35B-A3B"
        assert provider["baseUrl"] == "http://model/v1"
        assert provider["compat"]["supportsReasoningEffort"] is True
        assert provider["models"][0]["id"] == "Qwen3.6-35B-A3B"
        assert provider["models"][0]["maxTokens"] == 131072

    def test_preserves_explicit_provider_without_model_server(self) -> None:
        config = {"providers": {"custom": {"baseUrl": "https://example.test"}}}
        agent = _make_agent(models_config=config)
        assert agent._effective_model() == agent.config.model
        assert agent._build_models_config() == config


class TestSetup:
    def test_existing_version_matches_pin(self) -> None:
        result = CompletedProcess(["prime-agent", "--version"], 0, stdout="0.7.0\n", stderr="")
        with (
            patch("responses_api_agents.prime_agent.setup_prime_agent.shutil.which", return_value="/bin/prime-agent"),
            patch("responses_api_agents.prime_agent.setup_prime_agent.subprocess.run", return_value=result) as run,
        ):
            ensure_prime_agent("0.7.0")

        run.assert_called_once_with(["/bin/prime-agent", "--version"], check=True, capture_output=True, text=True)

    def test_existing_version_must_match_pin(self) -> None:
        result = CompletedProcess(["prime-agent", "--version"], 0, stdout="0.8.0\n", stderr="")
        with (
            patch("responses_api_agents.prime_agent.setup_prime_agent.shutil.which", return_value="/bin/prime-agent"),
            patch("responses_api_agents.prime_agent.setup_prime_agent.subprocess.run", return_value=result),
            pytest.raises(RuntimeError, match="does not match configured version"),
        ):
            ensure_prime_agent("0.7.0")


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "prime_agent.yaml"
        data = yaml.safe_load(config_path.read_text())
        inner = data["prime_agent"]["responses_api_agents"]["prime_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 8
        assert inner["command"] == "prime-agent"
        assert inner["prime_agent_version"] == "0.7.0"
        assert inner["models_config"]["providers"]["policy"]["compat"]["supportsReasoningEffort"] is True

    def test_environment_bundles(self) -> None:
        root = Path(__file__).resolve().parents[3]
        expected = {
            "prime_agent_math": "prime_agent_math_agent",
            "prime_agent_reasoning_gym": "prime_agent_reasoning_gym_agent",
        }
        for environment, agent_name in expected.items():
            data = yaml.safe_load((root / "environments" / environment / "config.yaml").read_text())
            assert agent_name in data
            config = data[agent_name]["responses_api_agents"]["prime_agent"]
            assert config["model_server"]["name"] == "policy_model"

    def test_environment_rollouts(self) -> None:
        root = Path(__file__).resolve().parents[3]
        for environment in ["prime_agent_math", "prime_agent_reasoning_gym"]:
            path = root / "environments" / environment / "data" / "example_rollouts.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 5
            assert all(row["reward"] == 1.0 for row in rows)

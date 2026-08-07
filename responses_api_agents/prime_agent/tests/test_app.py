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

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.prime_agent.app import (
    PrimeAgent,
    PrimeAgentConfig,
    ResourcesServerRef,
    _extract_instruction,
    parse_prime_agent_events,
)


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


class TestParsePrimeAgentEvents:
    def test_empty(self) -> None:
        items, usage = parse_prime_agent_events("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

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
        assert usage == {"input_tokens": 105, "output_tokens": 20}

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
                _message_end("toolResult", [{"type": "text", "text": "42"}], toolCallId="c1"),
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
        assert isinstance(items[2], NeMoGymResponseOutputMessage)

    def test_malformed_lines_skipped(self) -> None:
        line = "not-json\n" + _message_end("assistant", [{"type": "text", "text": "ok"}])
        items, _ = parse_prime_agent_events(line)
        assert len(items) == 1


class TestEnvironmentAndCommand:
    def test_env_isolates_config_and_shares_kernel(self, tmp_path: Path) -> None:
        kernel_venv = tmp_path / "kernel"
        agent = _make_agent(kernel_venv=str(kernel_venv), env={"NVIDIA_API_KEY": "k", "EMPTY": ""})
        home = tmp_path / "home"
        env = agent._env(home)
        assert env["HOME"] == str(home)
        assert env["PRIME_AGENT_CODING_AGENT_DIR"] == str(home / ".prime" / "agent")
        assert env["PRIME_AGENT_KERNEL_VENV"] == str(kernel_venv)
        assert env["PRIME_AGENT_INTERNAL_LEGACY_OWNED_WORKER_FRONTEND"] == "1"
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
        with patch.object(agent, "_resolve_model_base_url", return_value="http://model/v1"):
            config = agent._build_models_config()

        provider = config["providers"]["nemo"]
        assert agent._effective_model() == "nemo/Qwen3.6-35B-A3B"
        assert provider["baseUrl"] == "http://model/v1"
        assert provider["models"][0]["id"] == "Qwen3.6-35B-A3B"
        assert provider["models"][0]["maxTokens"] == 131072

    def test_preserves_explicit_provider_without_model_server(self) -> None:
        config = {"providers": {"custom": {"baseUrl": "https://example.test"}}}
        agent = _make_agent(models_config=config)
        assert agent._effective_model() == agent.config.model
        assert agent._build_models_config() == config


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

    def test_environment_bundles(self) -> None:
        root = Path(__file__).resolve().parents[3]
        expected = {
            "prime_agent_math": "prime_agent_math_agent",
            "prime_agent_reasoning_gym": "prime_agent_reasoning_gym_agent",
        }
        for environment, agent_name in expected.items():
            data = yaml.safe_load((root / "environments" / environment / "config.yaml").read_text())
            assert agent_name in data
            assert "prime_agent" in data[agent_name]["responses_api_agents"]

    def test_environment_rollouts(self) -> None:
        root = Path(__file__).resolve().parents[3]
        for environment in ["prime_agent_math", "prime_agent_reasoning_gym"]:
            path = root / "environments" / environment / "data" / "example_rollouts.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(rows) == 5
            assert all(row["reward"] == 1.0 for row in rows)

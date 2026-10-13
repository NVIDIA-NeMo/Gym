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
import logging
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import yaml
from omegaconf import OmegaConf

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
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


def _make_model_server_agent(**kwargs) -> KiloCodeAgent:
    """An agent wired to a Gym model server, with just enough server client to resolve its URL.

    The mocked client carries no global config, so the model server entry and the real base-URL
    builder are attached to it; base-URL resolution itself runs unmocked.
    """
    agent = _make_agent(model_server=ModelServerRef(type="responses_api_models", name="policy_model"), **kwargs)
    agent.server_client.global_config_dict = OmegaConf.create(
        {"policy_model": {"responses_api_models": {"vllm_model": {"host": "model-host", "port": 9000}}}}
    )
    agent.server_client._build_server_base_url = partial(ServerClient._build_server_base_url, agent.server_client)
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

    def test_empty_length_stop_warns(self, caplog) -> None:
        # The signature of an output budget that does not fit the model server's context window.
        event = _step_finish_event({"input": 10368, "output": 0})
        event["part"]["reason"] = "length"
        with caplog.at_level(logging.WARNING):
            _, usage = parse_kilo_events(_events(event))
        assert usage["output_tokens"] == 0
        assert "max_output_tokens" in caplog.text

    def test_length_stop_with_output_does_not_warn(self, caplog) -> None:
        event = _step_finish_event({"input": 10368, "output": 8192})
        event["part"]["reason"] = "length"
        with caplog.at_level(logging.WARNING):
            parse_kilo_events(_events(event))
        assert "max_output_tokens" not in caplog.text

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
        agent = _make_agent(model="policy/some-model")
        cmd = agent._build_command(Path("/tmp/ws"), "solve it")
        assert cmd[:4] == ["kilo", "run", "--auto", "--pure"]
        assert "--format" in cmd and cmd[cmd.index("--format") + 1] == "json"
        assert cmd[cmd.index("-m") + 1] == "policy/some-model"
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

    def test_model_registered_under_its_provider(self, tmp_path) -> None:
        # Kilo fails with "Model not found" unless the name appears in the provider's models map, so
        # the model is registered from `model` rather than repeated in every config.
        agent = _make_agent(
            model="policy/nvidia/qwen/qwen3-next-80b-a3b-instruct",
            kilo_config={"provider": {"policy": {"npm": "@ai-sdk/openai-compatible"}}},
        )
        agent._write_kilo_config(tmp_path)
        written = json.loads((tmp_path / "kilo.json").read_text())
        assert written["provider"]["policy"]["models"] == {"nvidia/qwen/qwen3-next-80b-a3b-instruct": {}}

    def test_existing_model_entry_preserved(self, tmp_path) -> None:
        agent = _make_agent(
            model="policy/m",
            kilo_config={"provider": {"policy": {"models": {"m": {"name": "custom"}, "other": {}}}}},
        )
        agent._write_kilo_config(tmp_path)
        written = json.loads((tmp_path / "kilo.json").read_text())
        assert written["provider"]["policy"]["models"] == {"m": {"name": "custom"}, "other": {}}

    def test_unknown_provider_left_alone(self, tmp_path) -> None:
        # A model on a provider kilo resolves itself (e.g. the gateway) must not fabricate a provider.
        agent = _make_agent(model="anthropic/claude", kilo_config={"provider": {"policy": {}}})
        agent._write_kilo_config(tmp_path)
        written = json.loads((tmp_path / "kilo.json").read_text())
        assert written["provider"] == {"policy": {}}


class TestModelServer:
    def test_effective_model_prefixed_only_with_model_server(self) -> None:
        assert _make_agent(model="policy/m")._effective_model() == "policy/m"
        assert _make_model_server_agent(model="Qwen/Qwen3-8B")._effective_model() == "nemo/Qwen/Qwen3-8B"

    def test_base_url_resolution(self) -> None:
        assert _make_agent()._resolve_model_base_url() == ""
        agent = _make_model_server_agent()
        assert agent._resolve_model_base_url() == "http://model-host:9000/v1"
        assert agent._resolve_model_base_url("r1") == "http://model-host:9000/ng-rollout/r1/v1"

    def test_nemo_provider_written(self, tmp_path) -> None:
        # A slashed model name stays whole: kilo splits `-m` on the first `/` only, so the provider is
        # `nemo` and the model is `Qwen/Qwen3-8B`, which is the key it looks up in `models`.
        agent = _make_model_server_agent(model="Qwen/Qwen3-8B")
        agent._write_kilo_config(tmp_path, agent._resolve_model_base_url())
        provider = json.loads((tmp_path / "kilo.json").read_text())["provider"]["nemo"]
        assert provider["npm"] == "@ai-sdk/openai-compatible"
        assert provider["options"] == {
            "apiKey": "EMPTY",  # pragma: allowlist secret
            "baseURL": "http://model-host:9000/v1",
        }
        assert provider["models"] == {
            "Qwen/Qwen3-8B": {
                "name": "Qwen/Qwen3-8B",
                "limit": {"context": 32768, "output": 8192},
                "interleaved": {"field": "reasoning_content"},
            }
        }

    def test_model_server_config_written_without_kilo_config(self, tmp_path) -> None:
        # An empty kilo_config used to short-circuit before kilo.json was written, which would leave a
        # model-server-only run with no provider at all.
        agent = _make_model_server_agent(model="m")
        agent._write_kilo_config(tmp_path, agent._resolve_model_base_url())
        assert json.loads((tmp_path / "kilo.json").read_text())["provider"]["nemo"]["models"].keys() == {"m"}

    def test_kilo_config_merged_and_overridable(self, tmp_path) -> None:
        agent = _make_model_server_agent(
            model="m",
            kilo_config={"permission": {"bash": "allow"}, "provider": {"nemo": {"name": "mine"}}},
        )
        agent._write_kilo_config(tmp_path, agent._resolve_model_base_url())
        written = json.loads((tmp_path / "kilo.json").read_text())
        assert written["permission"]["bash"] == "allow"
        assert written["provider"]["nemo"]["name"] == "mine"

    def test_limits_and_reasoning_field_configurable(self, tmp_path) -> None:
        agent = _make_model_server_agent(
            model="m", context_window=262144, max_output_tokens=16384, reasoning_field=None
        )
        agent._write_kilo_config(tmp_path, agent._resolve_model_base_url())
        model = json.loads((tmp_path / "kilo.json").read_text())["provider"]["nemo"]["models"]["m"]
        assert model["limit"] == {"context": 262144, "output": 16384}
        assert "interleaved" not in model

    def test_rollout_prefix_applied_to_base_url(self, tmp_path) -> None:
        agent = _make_model_server_agent(model="m")
        agent._write_kilo_config(tmp_path, agent._resolve_model_base_url("task0-rollout1"))
        written = json.loads((tmp_path / "kilo.json").read_text())
        assert written["provider"]["nemo"]["options"]["baseURL"].endswith("/ng-rollout/task0-rollout1/v1")

    def test_command_uses_effective_model(self) -> None:
        agent = _make_model_server_agent(model="Qwen/Qwen3-8B")
        cmd = agent._build_command(Path("/tmp/ws"), "hi")
        assert cmd[cmd.index("-m") + 1] == "nemo/Qwen/Qwen3-8B"

    def test_run_kilo_threads_resolved_url_into_config_and_env(self, tmp_path) -> None:
        # The seam _run_kilo owns: resolve once, then hand the same URL to kilo.json and the env.
        # repo_dir is the project dir, and unlike the workspace it survives the run's cleanup.
        agent = _make_model_server_agent(model="m", repo_dir=str(tmp_path))
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        with patch("responses_api_agents.kilocode_agent.app.asyncio.create_subprocess_exec") as spawn:
            spawn.return_value = proc
            asyncio.run(agent._run_kilo("hi", None, "task0-rollout1"))

        expected = "http://model-host:9000/ng-rollout/task0-rollout1/v1"
        written = json.loads((tmp_path / "kilo.json").read_text())
        assert written["provider"]["nemo"]["options"]["baseURL"] == expected
        assert spawn.call_args.kwargs["env"]["OPENAI_BASE_URL"] == expected

    def test_env_prefers_model_server_over_openai_base_url(self) -> None:
        agent = _make_model_server_agent(openai_api_key="k", openai_base_url="https://api.openai.com/v1")
        env = agent._env("/tmp/data", "/tmp/config", agent._resolve_model_base_url("r1"))
        assert env["OPENAI_BASE_URL"] == "http://model-host:9000/ng-rollout/r1/v1"
        assert env["OPENAI_API_KEY"] == "EMPTY"  # pragma: allowlist secret


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
        # The shipped config routes model calls through a Gym model server, so `model` is the bare
        # name and the agent supplies the provider; kilo_config must not declare one to collide with.
        assert inner["model_server"] == {"type": "responses_api_models", "name": "policy_model"}
        assert "provider" not in inner["kilo_config"]

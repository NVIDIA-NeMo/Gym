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
from functools import partial
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
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
from responses_api_agents.cline_agent.app import (
    ClineAgent,
    ClineAgentConfig,
    _extract_instruction,
    parse_cline_events,
    quote_prompt,
)


def _config(**kwargs) -> ClineAgentConfig:
    return ClineAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> ClineAgent:
    with patch("responses_api_agents.cline_agent.app.ClineAgent.model_post_init"):
        agent = ClineAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _make_model_server_agent(**kwargs) -> ClineAgent:
    """An agent wired to a Gym model server, with just enough server client to resolve its URL.

    The mocked client carries no global config, so the model server entry and the real base-URL
    builder are attached to it; base-URL resolution itself runs unmocked.
    """
    kwargs.setdefault("model", "policy-model")
    agent = _make_agent(model_server=ModelServerRef(type="responses_api_models", name="policy_model"), **kwargs)
    agent.server_client.global_config_dict = OmegaConf.create(
        {"policy_model": {"responses_api_models": {"vllm_model": {"host": "model-host", "port": 9000}}}}
    )
    agent.server_client._build_server_base_url = partial(ServerClient._build_server_base_url, agent.server_client)
    return agent


def _lines(*records) -> str:
    """Serialize records into the JSONL stream `cline --json` writes to stdout."""
    return "\n".join(json.dumps(r) for r in records)


def _agent_event(event: dict) -> dict:
    return {"ts": "2026-01-01T00:00:00.000Z", "type": "agent_event", "event": event}


def _text_start(text: str) -> dict:
    return _agent_event({"type": "content_start", "contentType": "text", "text": text, "accumulated": text})


def _text_end(text: str) -> dict:
    return _agent_event({"type": "content_end", "contentType": "text", "text": text})


def _reasoning_start(text: str, redacted: bool = False) -> dict:
    return _agent_event({"type": "content_start", "contentType": "reasoning", "reasoning": text, "redacted": redacted})


def _reasoning_end(text: str) -> dict:
    return _agent_event({"type": "content_end", "contentType": "reasoning", "reasoning": text})


def _tool_start(name: str, call_id: str, tool_input) -> dict:
    return _agent_event(
        {
            "type": "content_start",
            "contentType": "tool",
            "toolName": name,
            "toolCallId": call_id,
            "input": tool_input,
        }
    )


def _tool_end(name: str, call_id: str, output=None, error: str | None = None) -> dict:
    event = {
        "type": "content_end",
        "contentType": "tool",
        "toolName": name,
        "toolCallId": call_id,
        "durationMs": 11,
    }
    if error is not None:
        event["error"] = error
    else:
        event["output"] = output
    return _agent_event(event)


def _usage(total_in: int, total_out: int) -> dict:
    return _agent_event(
        {
            "type": "usage",
            "inputTokens": total_in,
            "outputTokens": total_out,
            "totalInputTokens": total_in,
            "totalOutputTokens": total_out,
        }
    )


def _run_result(**kwargs) -> dict:
    record = {
        "ts": "2026-01-01T00:00:00.000Z",
        "type": "run_result",
        "finishReason": "completed",
        "iterations": 1,
        "usage": {"inputTokens": 10, "outputTokens": 2, "cacheReadTokens": 0},
        "aggregateUsage": {"inputTokens": 10, "outputTokens": 2, "cacheReadTokens": 0},
        "durationMs": 100,
        "text": "done",
        "model": {"id": "test-model", "provider": "openai-compatible"},
    }
    record.update(kwargs)
    return record


def _hook_event(name: str) -> dict:
    return {"ts": "2026-01-01T00:00:00.000Z", "type": "hook_event", "hookEventName": name, "agentId": "a"}


class TestExtractInstruction:
    def test_user_only(self) -> None:
        user, system = _extract_instruction([NeMoGymEasyInputMessage(role="user", content="what is 2+2")])
        assert user == "what is 2+2"
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


class TestQuotePrompt:
    def test_single_word_padded(self) -> None:
        # Cline rejects a whitespace-free prompt arg as "Unknown command or unquoted prompt".
        assert quote_prompt("hello") == "hello "

    def test_multiword_untouched(self) -> None:
        assert quote_prompt("fix the tests") == "fix the tests"

    def test_newline_counts_as_whitespace(self) -> None:
        assert quote_prompt("hello\nworld") == "hello\nworld"


class TestParseClineEvents:
    def test_empty_stream(self) -> None:
        items, metadata = parse_cline_events("")
        assert items == []
        assert metadata == {"input_tokens": 0, "output_tokens": 0}

    def test_malformed_lines_skipped(self) -> None:
        stream = "not json\n" + _lines(_text_end("answer is 4")) + "\n{ broken"
        items, _ = parse_cline_events(stream)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)

    def test_assistant_text_from_content_end(self) -> None:
        # content_end carries the turn's full text; the streamed chunks must not be doubled onto it.
        stream = _lines(_text_start("the answer "), _text_start("is 4"), _text_end("the answer is 4"))
        items, _ = parse_cline_events(stream)
        assert len(items) == 1
        assert items[0].content[0].text == "the answer is 4"

    def test_streamed_text_survives_missing_content_end(self) -> None:
        # A stream cut off mid-turn never emits content_end; the chunks are all there is.
        items, _ = parse_cline_events(_lines(_text_start("partial "), _text_start("answer")))
        assert len(items) == 1
        assert items[0].content[0].text == "partial answer"

    def test_blank_text_ignored(self) -> None:
        items, _ = parse_cline_events(_lines(_text_end("   ")))
        assert items == []

    def test_hook_events_ignored(self) -> None:
        # hook_event mirrors tool boundaries the agent events already carry; counting both would
        # duplicate every call.
        stream = _lines(
            _hook_event("agent_start"),
            _hook_event("tool_call"),
            _tool_start("run_commands", "c1", {"commands": ["echo 6"]}),
            _hook_event("tool_result"),
            _tool_end("run_commands", "c1", output="6\n"),
            _hook_event("agent_end"),
        )
        items, _ = parse_cline_events(stream)
        assert [type(i).__name__ for i in items] == [
            "NeMoGymResponseFunctionToolCall",
            "NeMoGymFunctionCallOutput",
        ]

    def test_tool_call_and_output(self) -> None:
        stream = _lines(
            _tool_start("run_commands", "c1", {"commands": ["echo 6"]}),
            _tool_end("run_commands", "c1", output=[{"result": "6"}]),
            _text_end("answer is 6"),
        )
        items, _ = parse_cline_events(stream)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "run_commands"
        assert json.loads(items[0].arguments)["commands"] == ["echo 6"]
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output
        assert isinstance(items[2], NeMoGymResponseOutputMessage)

    def test_tool_error_surfaces_error_text(self) -> None:
        stream = _lines(
            _tool_start("read_files", "c2", {"files": []}),
            _tool_end("read_files", "c2", error="boom"),
        )
        items, _ = parse_cline_events(stream)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[1].output == "boom"

    def test_orphan_tool_result_gets_a_call(self) -> None:
        # A malformed stream can carry a result with no start; the output must not be orphaned.
        items, _ = parse_cline_events(_lines(_tool_end("run_commands", "c3", output="ok")))
        assert [type(i).__name__ for i in items] == [
            "NeMoGymResponseFunctionToolCall",
            "NeMoGymFunctionCallOutput",
        ]
        assert items[0].call_id == "c3"

    def test_text_before_tool_call_is_ordered_first(self) -> None:
        stream = _lines(
            _text_start("let me look"),
            _tool_start("run_commands", "c1", {"commands": ["ls"]}),
            _tool_end("run_commands", "c1", output=""),
            _text_end("done"),
        )
        items, _ = parse_cline_events(stream)
        assert [type(i).__name__ for i in items] == [
            "NeMoGymResponseOutputMessage",
            "NeMoGymResponseFunctionToolCall",
            "NeMoGymFunctionCallOutput",
            "NeMoGymResponseOutputMessage",
        ]
        assert items[0].content[0].text == "let me look"

    def test_reasoning_attached_to_its_own_turn(self) -> None:
        # Cline closes reasoning *after* the text of the same turn, so a parser that waited for
        # content_end would attach the think block to the following turn instead.
        stream = _lines(
            _reasoning_start("first I check the file"),
            _text_start("looking now"),
            _text_end("looking now"),
            _reasoning_end("first I check the file"),
            _text_end("all done"),
        )
        items, _ = parse_cline_events(stream)
        assert len(items) == 2
        assert items[0].content[0].text == "<think>\nfirst I check the file\n</think>\n\nlooking now"
        assert items[1].content[0].text == "all done"

    def test_reasoning_only_turn_surfaced(self) -> None:
        # Some vLLM reasoning parsers route the whole answer through the reasoning channel; it is
        # surfaced on its own rather than dropped.
        items, _ = parse_cline_events(_lines(_reasoning_start("dangling"), _reasoning_end("dangling thought")))
        assert len(items) == 1
        assert "dangling thought" in items[0].content[0].text

    def test_redacted_reasoning_skipped(self) -> None:
        stream = _lines(
            _agent_event({"type": "content_start", "contentType": "reasoning", "redacted": True}),
            _text_end("answer"),
        )
        items, _ = parse_cline_events(stream)
        assert items[0].content[0].text == "answer"

    def test_no_reasoning_means_no_think_block(self) -> None:
        items, _ = parse_cline_events(_lines(_text_end("plain")))
        assert "<think>" not in items[0].content[0].text

    def test_usage_from_run_result_aggregate(self) -> None:
        stream = _lines(
            _usage(50, 10),
            _run_result(aggregateUsage={"inputTokens": 100, "outputTokens": 20, "cacheReadTokens": 5}),
        )
        _, metadata = parse_cline_events(stream)
        assert metadata["input_tokens"] == 105
        assert metadata["output_tokens"] == 20

    def test_usage_falls_back_to_events_without_run_result(self) -> None:
        # A killed run never prints run_result; the per-turn totals are what remains.
        _, metadata = parse_cline_events(_lines(_usage(50, 10)))
        assert metadata["input_tokens"] == 50
        assert metadata["output_tokens"] == 10

    def test_run_result_metadata(self) -> None:
        _, metadata = parse_cline_events(_lines(_run_result(finishReason="aborted", iterations=3)))
        assert metadata["finish_reason"] == "aborted"
        assert metadata["iterations"] == 3
        assert metadata["model"] == "test-model"

    def test_done_event_fills_finish_reason_without_run_result(self) -> None:
        stream = _lines(_agent_event({"type": "done", "reason": "completed", "text": "hi", "iterations": 2}))
        _, metadata = parse_cline_events(stream)
        assert metadata["finish_reason"] == "completed"
        assert metadata["iterations"] == 2

    def test_error_record_recorded(self) -> None:
        stream = _lines({"ts": "t", "type": "error", "message": "no API key"}, _text_end("hi"))
        items, metadata = parse_cline_events(stream)
        assert metadata["error"] == "no API key"
        assert len(items) == 1


class TestBuildCommand:
    def test_command_shape(self) -> None:
        agent = _make_agent(model="some-model", timeout=600)
        cmd = agent._build_command(Path("/tmp/proj"), Path("/tmp/data"), "solve it")
        assert cmd[0] == "cline"
        assert "--json" in cmd
        assert cmd[cmd.index("--auto-approve") + 1] == "true"
        assert cmd[cmd.index("--cwd") + 1] == "/tmp/proj"
        assert cmd[cmd.index("--data-dir") + 1] == "/tmp/data"
        assert cmd[cmd.index("--timeout") + 1] == "600"
        assert cmd[cmd.index("-m") + 1] == "some-model"
        assert cmd[cmd.index("-P") + 1] == "openai-compatible"
        # prompt is passed after `--` so a leading-dash prompt is not read as a flag
        assert cmd[-2:] == ["--", "solve it"]
        assert "--thinking" not in cmd
        assert "-s" not in cmd

    def test_single_word_prompt_is_quoted(self) -> None:
        agent = _make_agent()
        cmd = agent._build_command(Path("/tmp/proj"), Path("/tmp/data"), "hello")
        assert cmd[-1] == "hello "

    def test_model_omitted_when_unset(self) -> None:
        agent = _make_agent()
        cmd = agent._build_command(Path("/tmp/proj"), Path("/tmp/data"), "hi there")
        assert "-m" not in cmd

    def test_optional_flags_gated(self) -> None:
        agent = _make_agent(
            thinking="high",
            compaction="off",
            retries=3,
            system_prompt_override="be terse",
            extra_args=["--verbose"],
        )
        cmd = agent._build_command(Path("/tmp/proj"), Path("/tmp/data"), "hi there")
        assert cmd[cmd.index("--thinking") + 1] == "high"
        assert cmd[cmd.index("--compaction") + 1] == "off"
        assert cmd[cmd.index("--retries") + 1] == "3"
        assert cmd[cmd.index("-s") + 1] == "be terse"
        assert "--verbose" in cmd

    def test_multiword_command_split(self) -> None:
        agent = _make_agent(command="npx cline")
        cmd = agent._build_command(Path("/tmp/proj"), Path("/tmp/data"), "hi there")
        assert cmd[:2] == ["npx", "cline"]

    def test_provider_used_without_model_server(self) -> None:
        agent = _make_agent(provider="anthropic")
        cmd = agent._build_command(Path("/tmp/proj"), Path("/tmp/data"), "hi there")
        assert cmd[cmd.index("-P") + 1] == "anthropic"


class TestEnv:
    def test_env_passthrough_and_isolation(self, tmp_path) -> None:
        agent = _make_agent(openai_api_key="k", openai_base_url="https://x/v1", env={"FOO": "bar", "EMPTY": ""})
        env = agent._env(tmp_path)
        assert env["OPENAI_API_KEY"] == "k"
        assert env["OPENAI_BASE_URL"] == "https://x/v1"
        # Every state path is pinned inside the run dir: an ambient CLINE_DATA_DIR or
        # CLINE_PROVIDER_SETTINGS_PATH would otherwise share provider settings and the session db
        # across concurrent rollouts.
        assert env["CLINE_DATA_DIR"] == str(tmp_path)
        assert env["CLINE_PROVIDER_SETTINGS_PATH"] == str(tmp_path / "settings" / "providers.json")
        assert env["CLINE_SESSION_DATA_DIR"] == str(tmp_path / "sessions")
        assert env["CLINE_SESSION_BACKEND_MODE"] == "local"
        assert env["FOO"] == "bar"
        assert "EMPTY" not in env

    def test_ambient_cline_env_is_overridden(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("CLINE_DATA_DIR", "/home/someone/.cline")
        monkeypatch.setenv("CLINE_PROVIDER_SETTINGS_PATH", "/home/someone/.cline/settings/providers.json")
        env = _make_agent()._env(tmp_path)
        assert env["CLINE_DATA_DIR"] == str(tmp_path)
        assert env["CLINE_PROVIDER_SETTINGS_PATH"] == str(tmp_path / "settings" / "providers.json")

    def test_command_permissions_serialized(self, tmp_path) -> None:
        agent = _make_agent(command_permissions={"allow": ["python3 *"], "deny": ["sudo *"]})
        env = agent._env(tmp_path)
        assert json.loads(env["CLINE_COMMAND_PERMISSIONS"]) == {"allow": ["python3 *"], "deny": ["sudo *"]}

    def test_command_permissions_absent_when_empty(self, tmp_path) -> None:
        assert "CLINE_COMMAND_PERMISSIONS" not in _make_agent()._env(tmp_path)


class TestModelServer:
    def test_auth_command_carries_resolved_url(self, tmp_path) -> None:
        # Cline reads provider/key/model/base URL from its settings file, not from run flags, so
        # the model-server path has to write that file first.
        agent = _make_model_server_agent(model="Qwen/Qwen3-8B")
        cmd = agent._build_auth_command(tmp_path, agent._resolve_model_base_url())
        assert cmd[:3] == ["cline", "auth", "openai-compatible"]
        assert cmd[cmd.index("--modelid") + 1] == "Qwen/Qwen3-8B"
        assert cmd[cmd.index("--baseurl") + 1] == "http://model-host:9000/v1"
        assert cmd[cmd.index("--data-dir") + 1] == str(tmp_path)

    def test_no_auth_command_without_model_server(self, tmp_path) -> None:
        assert _make_agent()._build_auth_command(tmp_path, "") is None

    def test_auth_requires_model(self, tmp_path) -> None:
        agent = _make_model_server_agent()
        agent.config.model = None
        with pytest.raises(ValueError, match="model"):
            agent._build_auth_command(tmp_path, agent._resolve_model_base_url())

    def test_rollout_prefix_applied_to_base_url(self, tmp_path) -> None:
        agent = _make_model_server_agent()
        cmd = agent._build_auth_command(tmp_path, agent._resolve_model_base_url("task0-rollout1"))
        assert cmd[cmd.index("--baseurl") + 1].endswith("/ng-rollout/task0-rollout1/v1")

    def test_provider_forced_to_openai_compatible(self) -> None:
        # `cline auth` only accepts a base URL for the OpenAI/OpenAI-compatible providers, so a
        # config naming another one cannot be honoured on the model-server path.
        agent = _make_model_server_agent(provider="anthropic")
        assert agent._effective_provider() == "openai-compatible"

    def test_env_prefers_model_server_over_openai_base_url(self, tmp_path) -> None:
        agent = _make_model_server_agent(openai_api_key="k", openai_base_url="https://api.openai.com/v1")
        env = agent._env(tmp_path, agent._resolve_model_base_url("r1"))
        assert env["OPENAI_BASE_URL"] == "http://model-host:9000/ng-rollout/r1/v1"
        assert env["OPENAI_API_KEY"] == "EMPTY"  # pragma: allowlist secret

    def test_run_cline_authenticates_then_runs_with_same_url(self, tmp_path) -> None:
        # The seam _run_cline owns: resolve the URL once, then use it for both subprocesses.
        agent = _make_model_server_agent(model="m", repo_dir=str(tmp_path), workspace_root=str(tmp_path / "ws"))
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        with patch("responses_api_agents.cline_agent.app.asyncio.create_subprocess_exec") as spawn:
            spawn.return_value = proc
            asyncio.run(agent._run_cline("fix it", None, "task0-rollout1"))

        expected = "http://model-host:9000/ng-rollout/task0-rollout1/v1"
        auth_argv, run_argv = [call.args for call in spawn.call_args_list]
        assert auth_argv[1] == "auth"
        assert expected in auth_argv
        assert "--json" in run_argv
        assert spawn.call_args_list[1].kwargs["env"]["OPENAI_BASE_URL"] == expected

    def test_run_cline_skips_auth_without_model_server(self, tmp_path) -> None:
        agent = _make_agent(repo_dir=str(tmp_path), workspace_root=str(tmp_path / "ws"))
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        with patch("responses_api_agents.cline_agent.app.asyncio.create_subprocess_exec") as spawn:
            spawn.return_value = proc
            asyncio.run(agent._run_cline("fix it", None))
        assert spawn.call_count == 1
        assert "auth" not in spawn.call_args.args

    def test_run_cline_prepends_system_prompt(self, tmp_path) -> None:
        agent = _make_agent(repo_dir=str(tmp_path), workspace_root=str(tmp_path / "ws"))
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        with patch("responses_api_agents.cline_agent.app.asyncio.create_subprocess_exec") as spawn:
            spawn.return_value = proc
            asyncio.run(agent._run_cline("fix it", "be careful"))
        assert spawn.call_args.args[-1] == "be careful\n\nfix it"

    def test_run_cline_cleans_up_workspace(self, tmp_path) -> None:
        # The ephemeral data dir holds this run's provider settings; it must not outlive the run.
        workspace_root = tmp_path / "ws"
        agent = _make_agent(workspace_root=str(workspace_root))
        proc = MagicMock()
        proc.returncode = 0
        proc.communicate = AsyncMock(return_value=(b"", b""))
        with patch("responses_api_agents.cline_agent.app.asyncio.create_subprocess_exec") as spawn:
            spawn.return_value = proc
            asyncio.run(agent._run_cline("fix it", None))
        assert list(workspace_root.iterdir()) == []

    def test_timeout_kills_group_and_flags_metadata(self, tmp_path) -> None:
        agent = _make_agent(repo_dir=str(tmp_path), workspace_root=str(tmp_path / "ws"), timeout=1)
        proc = MagicMock()
        proc.pid = 4242
        proc.returncode = None
        proc.communicate = AsyncMock(side_effect=[asyncio.TimeoutError(), (b"", b"")])
        with (
            patch("responses_api_agents.cline_agent.app.asyncio.create_subprocess_exec") as spawn,
            patch("responses_api_agents.cline_agent.app.os.killpg") as killpg,
            patch("responses_api_agents.cline_agent.app.os.getpgid", return_value=4242),
        ):
            spawn.return_value = proc
            _, metadata, _ = asyncio.run(agent._run_cline("fix it", None))
        assert killpg.called
        assert metadata["timed_out"] is True


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "cline_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "cline_agent" in data
        inner = data["cline_agent"]["responses_api_agents"]["cline_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["command"] == "cline"
        assert inner["concurrency"] == 8
        # The shipped config routes model calls through a Gym model server, so `model` is the bare
        # name the server serves and the agent supplies the provider and base URL.
        assert inner["model_server"] == {"type": "responses_api_models", "name": "policy_model"}
        # The version is pinned: the event parser was validated against it.
        assert inner["cline_version"]

    def test_anyswe_config_parses(self) -> None:
        # The SWE-bench path: anyswe runs this agent inside the task image and grades the patch.
        cfg_path = Path(__file__).resolve().parents[2] / "anyswe_agent" / "configs" / "anyswe_cline.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        inner = data["anyswe_cline"]["responses_api_agents"]["anyswe_agent"]
        assert inner["agent_server_module"] == "responses_api_agents.cline_agent.app"
        assert inner["agent_server_class"] == "ClineAgent"
        assert inner["agent_config_class"] == "ClineAgentConfig"
        # The repo under test must be the project dir, or the agent's edits land outside the
        # checkout anyswe takes the patch from.
        assert inner["agent_kwargs"]["repo_dir"] == "/testbed"

    def test_anyswe_config_kwargs_are_valid_config_fields(self) -> None:
        # anyswe passes agent_kwargs straight into ClineAgentConfig, so an unknown key there only
        # fails once a sandbox is up.
        cfg_path = Path(__file__).resolve().parents[2] / "anyswe_agent" / "configs" / "anyswe_cline.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        kwargs = data["anyswe_cline"]["responses_api_agents"]["anyswe_agent"]["agent_kwargs"]
        assert set(kwargs) <= set(ClineAgentConfig.model_fields)

    def test_anyswe_deps_script_exists(self) -> None:
        scripts = Path(__file__).resolve().parents[2] / "anyswe_agent" / "setup_scripts"
        assert (scripts / "cline_agent_deps.sh").exists()

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
from responses_api_agents.codex_agent.app import (
    CodexAgent,
    CodexAgentConfig,
    ResourcesServerRef,
    _extract_instruction,
    _response_to_codex_sse,
    _sanitize_codex_responses_body,
    parse_codex_events,
)


class _FakeRequest:
    """Minimal stand-in for a FastAPI Request: just carries the run-token header."""

    def __init__(self, run_token=None) -> None:
        self.headers = {"x-nemo-gym-run-token": run_token} if run_token else {}


def _config(**kwargs) -> CodexAgentConfig:
    return CodexAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> CodexAgent:
    with patch("responses_api_agents.codex_agent.app.CodexAgent.model_post_init"):
        agent = CodexAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _ev(type_, **kwargs) -> str:
    return json.dumps({"type": type_, **kwargs})


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 8
        assert cfg.command == "codex"
        assert cfg.wire_api == "responses"
        assert cfg.command_parts == ["codex"]

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


class TestParseCodexEvents:
    def test_empty(self) -> None:
        items, usage = parse_codex_events("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_agent_message(self) -> None:
        line = _ev("item.completed", item={"id": "i1", "type": "agent_message", "text": "the answer is 4"})
        items, _ = parse_codex_events(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"

    def test_command_execution_becomes_tool_call(self) -> None:
        line = _ev(
            "item.completed",
            item={
                "id": "c1",
                "type": "command_execution",
                "command": "python3 -c 'print(6)'",
                "aggregated_output": "6\n",
            },
        )
        items, _ = parse_codex_events(line)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "shell"
        assert json.loads(items[0].arguments)["command"] == "python3 -c 'print(6)'"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output

    def test_turn_completed_usage(self) -> None:
        line = _ev("turn.completed", usage={"input_tokens": 100, "output_tokens": 20, "cached_input_tokens": 5})
        _, usage = parse_codex_events(line)
        assert usage["input_tokens"] == 105
        assert usage["output_tokens"] == 20

    def test_non_completed_items_ignored(self) -> None:
        line = _ev("item.started", item={"id": "i1", "type": "agent_message", "text": "partial"})
        assert parse_codex_events(line)[0] == []

    def test_malformed_lines_skipped(self) -> None:
        line = "not-json\n" + _ev("item.completed", item={"id": "i1", "type": "agent_message", "text": "ok"})
        items, _ = parse_codex_events(line)
        assert len(items) == 1


class TestConfigToml:
    def test_writes_provider_block_direct(self, tmp_path) -> None:
        # no model_server -> direct base_url is used as-is (non-buffered eval)
        agent = _make_agent(base_url="https://x/v1", model="m", model_provider="nvinf", api_key_env="NVIDIA_API_KEY")
        agent._write_config_toml(tmp_path, _FakeRequest())
        toml = (tmp_path / "config.toml").read_text()
        assert 'model = "m"' in toml
        assert 'model_provider = "nvinf"' in toml
        assert "multi_agent = false" in toml
        assert "[model_providers.nvinf]" in toml
        assert 'base_url = "https://x/v1"' in toml
        assert 'wire_api = "responses"' in toml

    def test_run_scoped_points_at_agent_shim(self, tmp_path) -> None:
        # with model_server + run token, codex is pointed at THIS agent's run-scoped /runs/<token>/v1
        # shim and the model id becomes the model server name (the model server overrides the model).
        from nemo_gym.config_types import ModelServerRef

        agent = _make_agent(
            base_url="https://x/v1",
            model_provider="nvinf",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        with patch.object(CodexAgent, "_own_base_url", return_value="http://10.0.0.5:9000"):
            agent._write_config_toml(tmp_path, _FakeRequest(run_token="abc123"))
        toml = (tmp_path / "config.toml").read_text()
        assert 'model = "policy_model"' in toml
        assert 'base_url = "http://10.0.0.5:9000/runs/abc123/v1"' in toml


class TestSanitizeCodexResponsesBody:
    def test_drops_codex_only_fields_and_folds_instructions(self) -> None:
        body = {
            "model": "policy_model",
            "stream": True,
            "store": False,
            "include": [],
            "reasoning": None,
            "prompt_cache_key": "k",
            "client_metadata": {"x": "y"},
            "instructions": "be a math agent",
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hi"}]}],
            "tools": [
                {"type": "function", "name": "shell", "description": "", "strict": True, "parameters": {}},
                {"type": "web_search", "external_web_access": True},
            ],
        }
        out = _sanitize_codex_responses_body(body)
        dropped_keys = ("stream", "store", "include", "reasoning", "prompt_cache_key", "client_metadata")
        for dropped in (*dropped_keys, "instructions"):
            assert dropped not in out
        # instructions folded into input as a leading system message
        assert out["input"][0]["role"] == "system"
        assert out["input"][0]["content"][0]["text"] == "be a math agent"
        assert out["input"][1]["role"] == "user"
        # only function tools kept
        assert [t["type"] for t in out["tools"]] == ["function"]
        assert out["tool_choice"] == "auto"
        assert out["model"] == "policy_model"

    def test_multi_turn_input_validates_against_model_server(self) -> None:
        # A representative codex FOLLOW-UP turn: codex replays the whole prior trajectory
        # (system instructions + user + assistant reasoning + function_call + function_call_output
        # + next user). Several of these item shapes are rejected by the model server's Responses
        # `input` union as codex sends them (reasoning without id, output_text assistant content
        # without id/annotations, list-shaped tool output), which previously produced a 422.
        # After sanitizing, the whole body must validate as NeMoGymResponseCreateParamsNonStreaming.
        from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming

        body = {
            "model": "policy_model",
            "stream": True,
            "instructions": "be a math agent",
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "input": [
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "compute 6*7"}]},
                # reasoning item WITHOUT an id (codex) -> previously broke the union
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "I'll run python"}]},
                # assistant message with output_text content and no id/annotations
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "let me compute that"}],
                },
                # codex function_call: call_id is a chatcmpl-tool id; arguments as a dict
                {
                    "type": "function_call",
                    "call_id": "chatcmpl-tool-abc123",
                    "name": "shell",
                    "arguments": {"command": "python3 -c 'print(6*7)'"},
                },
                # function_call_output with a LIST output (content blocks) -> must become a string
                {
                    "type": "function_call_output",
                    "call_id": "chatcmpl-tool-abc123",
                    "output": [{"type": "output_text", "text": "42\n"}],
                },
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "now multiply by 2"}]},
            ],
            "tools": [{"type": "function", "name": "shell", "description": "", "strict": True, "parameters": {}}],
        }
        out = _sanitize_codex_responses_body(body)

        # Must construct without ValidationError (this is what the model server does on the forward).
        validated = NeMoGymResponseCreateParamsNonStreaming(**out)
        names = [type(x).__name__ for x in validated.input]
        # leading folded system instructions + the (normalized) replayed trajectory, in order
        assert names == [
            "NeMoGymEasyInputMessage",  # folded instructions (system)
            "NeMoGymEasyInputMessage",  # user
            "NeMoGymResponseReasoningItem",  # reasoning (id synthesized)
            "NeMoGymEasyInputMessage",  # assistant (output_text -> str)
            "NeMoGymResponseFunctionToolCall",  # function_call
            "NeMoGymFunctionCallOutput",  # function_call_output
            "NeMoGymEasyInputMessage",  # next user
        ]
        # turn order preserved
        assert validated.input[1].role == "user" and validated.input[1].content == "compute 6*7"
        assert validated.input[3].role == "assistant" and validated.input[3].content == "let me compute that"
        # reasoning got a synthesized id and list summary
        assert validated.input[2].id and isinstance(validated.input[2].summary, list)
        # function_call: dict arguments json-encoded, chatcmpl call_id preserved
        fc = validated.input[4]
        assert fc.call_id == "chatcmpl-tool-abc123" and fc.name == "shell"
        assert json.loads(fc.arguments)["command"] == "python3 -c 'print(6*7)'"
        # function_call_output: list output flattened to a string, call_id matched
        fco = validated.input[5]
        assert fco.call_id == "chatcmpl-tool-abc123" and fco.output == "42\n"

    def test_no_function_tools_drops_tool_fields(self) -> None:
        body = {
            "input": [{"type": "message", "role": "user", "content": []}],
            "tools": [{"type": "web_search", "external_web_access": True}],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        out = _sanitize_codex_responses_body(body)
        assert "tools" not in out
        assert "tool_choice" not in out
        assert "parallel_tool_calls" not in out


class TestResponseToCodexSse:
    def test_emits_minimal_event_sequence(self) -> None:
        response = {
            "id": "resp_1",
            "model": "policy_model",
            "output": [
                {
                    "type": "message",
                    "id": "m1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "\\boxed{42}", "annotations": []}],
                },
            ],
            "usage": {"input_tokens": 1, "output_tokens": 2, "total_tokens": 3},
        }
        sse = _response_to_codex_sse(response)
        events = [json.loads(line[len("data: ") :]) for line in sse.strip().split("\n\n") if line.startswith("data: ")]
        types = [e["type"] for e in events]
        assert types == ["response.created", "response.output_item.done", "response.completed"]
        # created carries an in-progress response with empty output
        assert events[0]["response"]["status"] == "in_progress"
        assert events[0]["response"]["output"] == []
        # the output item is carried verbatim
        assert events[1]["item"]["content"][0]["text"] == "\\boxed{42}"
        assert events[1]["output_index"] == 0
        # completed carries the full response with output + usage
        assert events[2]["response"]["status"] == "completed"
        assert events[2]["response"]["output"] == response["output"]
        assert events[2]["response"]["usage"]["total_tokens"] == 3


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "codex_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        inner = data["codex_agent"]["responses_api_agents"]["codex_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["command"] == "codex"
        assert inner["wire_api"] == "responses"

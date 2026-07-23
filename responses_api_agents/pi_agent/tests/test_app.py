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

import yaml

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.rollout_observability import AgentInvocation, ContextCompactionObservation, ToolCallObservation
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pi_agent.app import (
    PiAgent,
    PiAgentConfig,
    PiAgentRunRequest,
    ResourcesServerRef,
    _build_pi_observations,
    _extract_instruction,
    _read_pi_stdout,
    parse_pi_events,
)


def _config(**kwargs) -> PiAgentConfig:
    return PiAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> PiAgent:
    with patch("responses_api_agents.pi_agent.app.PiAgent.model_post_init"):
        agent = PiAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _msg_end(role, content, **extra) -> str:
    return json.dumps({"type": "message_end", "message": {"role": role, "content": content, **extra}})


def _records(bundle, record_type):
    return [record for record in bundle.records if isinstance(record, record_type)]


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 8
        assert cfg.command == "pi"
        assert cfg.command_parts == ["pi"]

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


class TestParsePiEvents:
    def test_empty(self) -> None:
        items, usage = parse_pi_events("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_assistant_text_and_usage(self) -> None:
        line = _msg_end(
            "assistant",
            [{"type": "text", "text": "the answer is 4"}],
            usage={"input": 100, "output": 20, "cacheRead": 5},
        )
        items, usage = parse_pi_events(line)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"
        assert usage["input_tokens"] == 105
        assert usage["output_tokens"] == 20

    def test_user_messages_ignored(self) -> None:
        line = _msg_end("user", [{"type": "text", "text": "hi"}])
        assert parse_pi_events(line)[0] == []

    def test_non_message_end_events_ignored(self) -> None:
        line = json.dumps({"type": "message_update", "message": {"role": "assistant", "content": []}})
        assert parse_pi_events(line)[0] == []

    def test_tool_call_and_result(self) -> None:
        lines = "\n".join(
            [
                _msg_end(
                    "assistant", [{"type": "toolCall", "id": "c1", "name": "bash", "arguments": {"command": "echo 6"}}]
                ),
                _msg_end("toolResult", [{"type": "text", "text": "6\n"}], toolCallId="c1", toolName="bash"),
                _msg_end("assistant", [{"type": "text", "text": "answer is 6"}]),
            ]
        )
        items, _ = parse_pi_events(lines)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "bash"
        assert json.loads(items[0].arguments)["command"] == "echo 6"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output
        assert isinstance(items[2], NeMoGymResponseOutputMessage)

    def test_malformed_lines_skipped(self) -> None:
        line = "not-json\nnull\n[]\n" + _msg_end("assistant", [{"type": "text", "text": "ok"}])
        items, _ = parse_pi_events(line)
        assert len(items) == 1


class TestEnv:
    def test_env_passthrough(self) -> None:
        agent = _make_agent(env={"NVIDIA_API_KEY": "k", "EMPTY": ""})
        env = agent._env(Path("/tmp/h"))
        assert env["NVIDIA_API_KEY"] == "k"
        assert env["HOME"] == "/tmp/h"
        assert "EMPTY" not in env


class TestRolloutObservability:
    def test_routes_configured_provider_without_mutating_config(self) -> None:
        models_config = {"providers": {"nvinf": {"baseUrl": "https://inference.api.nvidia.com/v1"}}}
        agent = _make_agent(
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            models_config=models_config,
        )

        with patch.object(PiAgent, "resolve_model_base_url", return_value="http://policy/ng-rollout/1-2/v1"):
            routed = agent._models_config_for_run("1-2")

        assert routed["providers"]["nvinf"]["baseUrl"] == "http://policy/ng-rollout/1-2/v1"
        assert agent.config.models_config == models_config

    async def test_reads_and_timestamps_json_events(self) -> None:
        stream = asyncio.StreamReader()
        stream.feed_data(b'{"type":"tool_execution_start","toolCallId":"a"}\nnot-json\n')
        stream.feed_eof()

        with patch("responses_api_agents.pi_agent.app.time", side_effect=[10.0, 11.0]):
            stdout, events = await _read_pi_stdout(stream)

        assert stdout.endswith("not-json\n")
        assert events == [(10.0, {"type": "tool_execution_start", "toolCallId": "a"})]

    async def test_reads_json_events_larger_than_streamreader_line_limit(self) -> None:
        event = {"type": "message_end", "message": {"role": "assistant", "content": "x" * 70_000}}
        payload = (json.dumps(event) + "\n").encode()
        stream = asyncio.StreamReader()
        stream.feed_data(payload)
        stream.feed_eof()

        stdout, events = await _read_pi_stdout(stream)

        assert stdout.encode() == payload
        assert events[0][1] == event

    def test_exact_model_calls_parallel_tools_and_compaction(self) -> None:
        assistant = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "responseId": "resp-upstream-1",
                "content": [
                    {"type": "toolCall", "id": "a", "name": "read", "arguments": {}},
                    {"type": "toolCall", "id": "b", "name": "bash", "arguments": {}},
                ],
            },
        }
        final_assistant = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "responseId": "resp-upstream-2",
                "content": [{"type": "text", "text": "done"}],
            },
        }
        events = [
            (1.0, assistant),
            (2.0, {"type": "tool_execution_start", "toolCallId": "a", "toolName": "read"}),
            (3.0, {"type": "tool_execution_start", "toolCallId": "b", "toolName": "bash"}),
            (4.0, {"type": "tool_execution_end", "toolCallId": "b", "toolName": "bash", "isError": True}),
            (5.0, {"type": "tool_execution_end", "toolCallId": "a", "toolName": "read", "isError": False}),
            (6.0, {"type": "compaction_start", "reason": "threshold"}),
            (
                7.0,
                {
                    "type": "compaction_end",
                    "reason": "threshold",
                    "result": {
                        "summary": "condensed history",
                        "firstKeptEntryId": "entry-7",
                        "tokensBefore": 150_000,
                        "estimatedTokensAfter": 32_000,
                    },
                    "aborted": False,
                },
            ),
            (8.0, final_assistant),
        ]
        stdout = "\n".join(json.dumps(event) for _, event in events)
        items, _ = parse_pi_events(stdout)
        model_ref = ModelServerRef(type="responses_api_models", name="policy")

        bundle = _build_pi_observations(events, "rollout-1", model_ref, items)

        [invocation] = _records(bundle, AgentInvocation)
        assert invocation.model_calls[0].response_id == "resp-upstream-1"
        assert invocation.model_calls[0].model_ref == model_ref
        timings = {tool.tool_call_id: tool for tool in _records(bundle, ToolCallObservation)}
        assert (
            timings["a"].started_at < timings["b"].started_at < timings["b"].completed_at < timings["a"].completed_at
        )
        assert timings["a"].duration_ms == 3000
        assert timings["b"].duration_ms == 1000
        assert all(tool.timing_source == "harness" for tool in timings.values())
        assert timings["a"].status == "completed"
        assert timings["b"].status == "failed"
        assert timings["b"].error_type is None
        [compaction] = _records(bundle, ContextCompactionObservation)
        assert compaction.trigger == "threshold"
        assert compaction.tokens_before == 150_000
        assert compaction.tokens_after == 32_000
        assert compaction.outcome == "completed"
        assert compaction.summary == "condensed history"
        assert compaction.first_kept_item_id == "entry-7"
        assert compaction.before_model_call is not None
        assert compaction.after_model_call is not None
        assert compaction.before_model_call.response_id == "resp-upstream-1"
        assert compaction.after_model_call.response_id == "resp-upstream-2"
        assert {gap.code for gap in bundle.gaps} == {"subagent_hierarchy_unavailable"}

    def test_compaction_outcome_uses_native_status(self) -> None:
        events = [
            (1.0, {"type": "compaction_start", "reason": "manual"}),
            (2.0, {"type": "compaction_end", "reason": "manual", "result": None, "aborted": True}),
            (3.0, {"type": "compaction_start", "reason": "overflow"}),
            (
                4.0,
                {
                    "type": "compaction_end",
                    "reason": "overflow",
                    "result": None,
                    "aborted": False,
                    "errorMessage": "quota",
                },
            ),
        ]

        compactions = _records(
            _build_pi_observations(events, "rollout-1", None, []),
            ContextCompactionObservation,
        )

        assert [item.outcome for item in compactions] == ["aborted", "failed"]

    def test_compaction_join_does_not_skip_unjoinable_model_call(self) -> None:
        model_ref = ModelServerRef(type="responses_api_models", name="policy")
        events = [
            (1.0, {"type": "message_end", "message": {"role": "assistant", "responseId": "resp-1"}}),
            (2.0, {"type": "compaction_start", "reason": "threshold"}),
            (3.0, {"type": "compaction_end", "reason": "threshold", "result": None, "aborted": True}),
            (4.0, {"type": "message_end", "message": {"role": "assistant"}}),
            (5.0, {"type": "message_end", "message": {"role": "assistant", "responseId": "resp-3"}}),
        ]

        [compaction] = _records(
            _build_pi_observations(events, "rollout-1", model_ref, []),
            ContextCompactionObservation,
        )

        assert compaction.before_model_call is not None
        assert compaction.before_model_call.response_id == "resp-1"
        assert compaction.after_model_call is None

    def test_reports_only_missing_evidence(self) -> None:
        items, _ = parse_pi_events(_msg_end("assistant", [{"type": "toolCall", "id": "c1", "name": "bash"}]))
        bundle = _build_pi_observations([], "rollout-1", None, items)

        assert {gap.code for gap in bundle.gaps} == {
            "model_call_ownership_unavailable",
            "subagent_hierarchy_unavailable",
            "tool_timing_unavailable",
        }
        assert _records(bundle, ContextCompactionObservation) == []

    def test_episode_preserves_response_and_observations(self) -> None:
        event = {
            "type": "message_end",
            "message": {
                "role": "assistant",
                "responseId": "resp-upstream-1",
                "content": [{"type": "text", "text": "done"}],
            },
        }
        items, usage = parse_pi_events(json.dumps(event))
        agent = _make_agent(model_server=ModelServerRef(type="responses_api_models", name="policy"))
        agent._run_pi = AsyncMock(return_value=(items, usage, "model", [(1.0, event)]))

        episode = asyncio.run(
            agent._create_episode(NeMoGymResponseCreateParamsNonStreaming(input="solve"), rollout_id="1-2")
        )

        assert agent._run_pi.await_args.kwargs["rollout_id"] == "1-2"
        assert episode.response.output == items
        [invocation] = _records(episode.observations, AgentInvocation)
        assert invocation.conversation == [
            NeMoGymEasyInputMessage(role="user", content="solve"),
            *items,
        ]
        assert invocation.model_calls[0].response_id == "resp-upstream-1"

    def test_padding_is_not_reported_as_agent_evidence(self) -> None:
        agent = _make_agent()
        agent._run_pi = AsyncMock(return_value=([], {"input_tokens": 0, "output_tokens": 0}, "model", []))

        episode = asyncio.run(agent._create_episode(NeMoGymResponseCreateParamsNonStreaming(input="solve")))

        assert episode.response.output
        [invocation] = _records(episode.observations, AgentInvocation)
        assert invocation.conversation == [NeMoGymEasyInputMessage(role="user", content="solve")]
        assert "agent_transcript_unavailable" in {gap.code for gap in episode.observations.gaps}

    def test_partial_events_survive_empty_scoring_output(self) -> None:
        event = {"type": "tool_execution_start", "toolCallId": "call-1", "toolName": "bash"}
        agent = _make_agent()
        agent._run_pi = AsyncMock(return_value=([], {"input_tokens": 0, "output_tokens": 0}, "model", [(1.0, event)]))

        episode = asyncio.run(agent._create_episode(NeMoGymResponseCreateParamsNonStreaming(input="solve")))

        [tool] = _records(episode.observations, ToolCallObservation)
        assert tool.tool_call_id == "call-1"
        assert tool.status == "incomplete"

    def test_response_propagates_rollout(self) -> None:
        event = {
            "type": "message_end",
            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
        }
        items, usage = parse_pi_events(json.dumps(event))
        agent = _make_agent()
        agent._run_pi = AsyncMock(return_value=(items, usage, "model", [(1.0, event)]))
        request = MagicMock(path_params={"rollout_id": "1-2"})

        asyncio.run(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="solve")))

        assert agent._run_pi.await_args.kwargs["rollout_id"] == "1-2"

    def test_public_observation_boundary_collects_without_a_request(self) -> None:
        agent = _make_agent()
        response = NeMoGymResponseCreateParamsNonStreaming(input="solve")
        agent._run_pi = AsyncMock(return_value=([], {"input_tokens": 0, "output_tokens": 0}, "model", []))

        episode = asyncio.run(agent.responses_with_observations(None, response))

        assert episode.observations.source == "pi"
        assert agent._run_pi.await_args.kwargs["collect_observations"] is True
        assert episode.response.output

    def test_run_uses_prefixed_response_boundary(self) -> None:
        agent = _make_agent()
        agent.server_client.global_config_dict = {"observability_enabled": True}
        agent._run_pi = AsyncMock(return_value=([], {"input_tokens": 0, "output_tokens": 0}, "model", []))

        def response(payload):
            result = MagicMock(ok=True, cookies={})
            result.read = AsyncMock(return_value=json.dumps(payload).encode())
            return result

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path.endswith("/v1/responses"):
                agent_response = await agent.responses(MagicMock(path_params={"rollout_id": "1-2"}), json)
                return response(agent_response.model_dump(mode="json"))
            if url_path == "/verify":
                return response(json | {"reward": 1.0})
            return response({})

        agent.server_client.post = AsyncMock(side_effect=post)
        body = PiAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "solve"},
                "_ng_task_index": 1,
                "_ng_rollout_index": 2,
            }
        )

        result = asyncio.run(agent.run(MagicMock(cookies={}), body))

        assert result.ng_agent_observations is not None
        assert agent.server_client.post.await_args_list[1].kwargs["url_path"] == "/ng-rollout/1-2/v1/responses"
        assert agent.server_client.post.await_args_list[2].kwargs["json"]["rollout_id"] == "1-2"


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "pi_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "pi_agent" in data
        inner = data["pi_agent"]["responses_api_agents"]["pi_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 8
        assert inner["command"] == "pi"

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
import contextlib
import json
import os
import signal
import sys
from pathlib import Path
from typing import Optional
from unittest.mock import AsyncMock, MagicMock, patch

import psutil
import yaml

from nemo_gym.config_types import ModelServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.rollout_observability import AgentInvocation, ObservationGap
from nemo_gym.server_utils import ServerClient
from responses_api_agents.openclaw_agent.app import (
    OpenClawAgent,
    OpenClawAgentConfig,
    OpenClawAgentRunRequest,
    ResourcesServerRef,
    _decode_last_json_dict_suffix,
    _extract_instruction,
    _text_from_openclaw_payloads,
    openclaw_session_conversation,
    parse_openclaw_output,
    parse_openclaw_session,
    parse_openclaw_session_events,
    parse_openclaw_session_items,
)


class _FakeResponse:
    ok = True

    def __init__(self, payload: dict, cookies: dict | None = None) -> None:
        self.payload = payload
        self.cookies = cookies or {}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _invocations(bundle):
    return [record for record in bundle.records if isinstance(record, AgentInvocation)]


def _config(**kwargs) -> OpenClawAgentConfig:
    kwargs.setdefault("openclaw_version", "2026.6.11")
    return OpenClawAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _make_agent(**kwargs) -> OpenClawAgent:
    with patch("responses_api_agents.openclaw_agent.app.OpenClawAgent.model_post_init"):
        agent = OpenClawAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _envelope(payloads, usage=None, final_text=None) -> str:
    meta = {}
    if usage is not None:
        meta["agentMeta"] = {"usage": usage}
    if final_text is not None:
        meta["finalAssistantVisibleText"] = final_text
    return json.dumps({"payloads": payloads, "meta": meta})


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 32
        assert cfg.command == "openclaw"
        assert cfg.thinking == "off"
        assert cfg.command_parts == ["openclaw"]

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


class TestDecodeJsonSuffix:
    def test_plain_json(self) -> None:
        assert _decode_last_json_dict_suffix('{"a": 1}') == {"a": 1}

    def test_log_lines_before_json(self) -> None:
        raw = 'INFO booting\nWARN slow\n{"a": 1, "b": {"c": 2}}'
        assert _decode_last_json_dict_suffix(raw) == {"a": 1, "b": {"c": 2}}

    def test_empty_returns_none(self) -> None:
        assert _decode_last_json_dict_suffix("   ") is None

    def test_no_json_returns_none(self) -> None:
        assert _decode_last_json_dict_suffix("just logs, no json") is None


class TestTextFromPayloads:
    def test_plain_text(self) -> None:
        env = {"payloads": [{"text": "hello"}, {"text": "world"}]}
        assert _text_from_openclaw_payloads(env) == "hello\n\nworld"

    def test_falls_back_to_final_visible_text(self) -> None:
        env = {"payloads": [], "meta": {"finalAssistantVisibleText": "final"}}
        assert _text_from_openclaw_payloads(env) == "final"

    def test_no_payloads(self) -> None:
        assert _text_from_openclaw_payloads({}) == ""


class TestParseOpenclawOutput:
    def test_empty(self) -> None:
        items, usage = parse_openclaw_output("")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0}

    def test_text_message_and_usage(self) -> None:
        raw = _envelope(
            [{"text": "the answer is 4"}],
            usage={"input": 100, "output": 20, "cacheRead": 5},
        )
        items, usage = parse_openclaw_output(raw)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"
        assert usage["input_tokens"] == 105
        assert usage["output_tokens"] == 20
        assert usage["cached_tokens"] == 5

    def test_no_text_no_items(self) -> None:
        raw = _envelope([], usage={"input": 1, "output": 0})
        items, usage = parse_openclaw_output(raw)
        assert items == []
        assert usage["input_tokens"] == 1

    def test_log_prefix_tolerated(self) -> None:
        raw = "INFO starting agent\n" + _envelope([{"text": "ok"}])
        items, _ = parse_openclaw_output(raw)
        assert len(items) == 1
        assert items[0].content[0].text == "ok"


class TestParseOpenclawSession:
    def _msg(self, role, content, **extra) -> str:
        return json.dumps({"type": "message", "message": {"role": role, "content": content, **extra}})

    def test_tool_call_and_result_and_final_text(self) -> None:
        lines = "\n".join(
            [
                json.dumps({"type": "session", "id": "s1"}),  # non-message, ignored
                self._msg("user", [{"type": "text", "text": "compute 2*3"}]),
                self._msg(
                    "assistant",
                    [
                        {
                            "type": "toolCall",
                            "id": "c1",
                            "name": "exec",
                            "arguments": {"command": "python3 -c 'print(6)'"},
                        }
                    ],
                ),
                self._msg("toolResult", [{"type": "text", "text": "6\n"}], toolCallId="c1", toolName="exec"),
                self._msg("assistant", [{"type": "text", "text": "The answer is 6."}]),
            ]
        )
        items = parse_openclaw_session(lines)
        assert len(items) == 3
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "exec"
        assert json.loads(items[0].arguments)["command"] == "python3 -c 'print(6)'"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output
        assert isinstance(items[2], NeMoGymResponseOutputMessage)
        assert items[2].content[0].text == "The answer is 6."

    def test_user_messages_ignored(self) -> None:
        line = self._msg("user", [{"type": "text", "text": "hi"}])
        assert parse_openclaw_session(line) == []

    def test_malformed_lines_skipped(self) -> None:
        line = "not-json\nnull\n[]\n" + self._msg("assistant", [{"type": "text", "text": "ok"}])
        items = parse_openclaw_session(line)
        assert len(items) == 1

    def test_preserves_string_input_and_reasoning(self) -> None:
        events = [
            {"type": "message", "message": {"role": "user", "content": "solve this"}},
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "text": "need a tool"},
                        {"type": "toolCall", "id": "c1", "name": "search", "arguments": {}},
                    ],
                },
            },
        ]

        items = parse_openclaw_session_items(events, include_input=True)

        assert [item.type for item in items] == ["message", "reasoning", "function_call"]
        assert items[0].content == "solve this"
        assert items[1].summary[0].text == "need a tool"

    def test_input_only_transcript_uses_fallback_output(self) -> None:
        fallback = NeMoGymResponseOutputMessage(
            id="fallback",
            content=[{"type": "output_text", "text": "done", "annotations": []}],
        )

        conversation = openclaw_session_conversation(
            [{"type": "message", "message": {"role": "user", "content": "retained input"}}],
            input_items=[NeMoGymEasyInputMessage(role="user", content="request input")],
            fallback_output=[fallback],
        )

        assert conversation == [NeMoGymEasyInputMessage(role="user", content="retained input"), fallback]

    def test_preserves_conflicting_duplicate_event_ids(self) -> None:
        events = [
            {"id": "same", "type": "message", "message": {"role": "assistant", "content": "one"}},
            {"id": "same", "type": "message", "message": {"role": "assistant", "content": "two"}},
        ]

        items = parse_openclaw_session_items(events, include_input=True)

        assert [item.content[0].text for item in items] == ["one", "two"]

    def test_restores_missing_known_system_input(self) -> None:
        system = NeMoGymEasyInputMessage(role="system", content="system rules")
        user = NeMoGymEasyInputMessage(role="user", content="solve this")

        conversation = openclaw_session_conversation(
            [{"type": "message", "message": {"role": "user", "content": "solve this"}}],
            input_items=[system, user],
        )

        assert conversation == [system, user]


class TestBuildOpenclawConfig:
    def test_headless_message_tool_denied(self) -> None:
        agent = _make_agent()
        cfg = agent._build_openclaw_config({})
        assert "message" in cfg["tools"]["deny"]

    def test_preserves_setup_config(self) -> None:
        agent = _make_agent()
        base = {"gateway": {"mode": "local", "auth": {"token": "abc"}}, "agents": {"defaults": {"workspace": "/w"}}}
        cfg = agent._build_openclaw_config(base)
        assert cfg["gateway"]["auth"]["token"] == "abc"
        assert cfg["agents"]["defaults"]["workspace"] == "/w"

    def test_existing_denies_preserved_and_deduped(self) -> None:
        agent = _make_agent()
        cfg = agent._build_openclaw_config({"tools": {"deny": ["message", "gateway"]}})
        assert cfg["tools"]["deny"] == ["message", "gateway"]

    def test_user_openclaw_config_merged(self) -> None:
        agent = _make_agent(
            openclaw_config={"models": {"providers": {"nvinf": {"baseUrl": "https://x/v1"}}}, "extra": {"k": "v"}}
        )
        cfg = agent._build_openclaw_config({"gateway": {"mode": "local"}})
        assert cfg["models"]["providers"]["nvinf"]["baseUrl"] == "https://x/v1"
        assert cfg["extra"] == {"k": "v"}
        assert cfg["gateway"]["mode"] == "local"

    def test_model_server_replaces_provider_base_url_for_rollout(self) -> None:
        agent = _make_agent(
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
        )
        with patch.object(
            OpenClawAgent,
            "resolve_model_base_url",
            return_value="http://policy/ng-rollout/7-2/v1",
        ):
            cfg = agent._build_openclaw_config({}, "7-2")

        assert cfg["models"]["providers"]["nemo"]["baseUrl"] == "http://policy/ng-rollout/7-2/v1"

    def test_responses_propagates_rollout_path(self) -> None:
        agent = _make_agent()

        async def run_openclaw(*args, **kwargs):
            assert kwargs["rollout_id"] == "7-2"
            return [], {"input_tokens": 0, "output_tokens": 0}, "model"

        request = MagicMock(path_params={"rollout_id": "7-2"})
        with patch.object(agent, "_run_openclaw", run_openclaw):
            asyncio.run(agent.responses(request, NeMoGymResponseCreateParamsNonStreaming(input="solve")))

    def test_user_deny_cannot_drop_headless_deny(self) -> None:
        agent = _make_agent(openclaw_config={"tools": {"deny": ["custom"]}})
        cfg = agent._build_openclaw_config({})
        assert "message" in cfg["tools"]["deny"]
        assert "custom" in cfg["tools"]["deny"]

    def test_model_server_builds_local_provider(self) -> None:
        agent = _make_agent(
            model="Qwen3.6-35B-A3B",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        with patch.object(agent, "_resolve_model_base_url", return_value="http://model/v1"):
            cfg = agent._build_openclaw_config({})

        provider = cfg["models"]["providers"]["nemo"]
        assert agent._effective_model() == "nemo/Qwen3.6-35B-A3B"
        assert provider["baseUrl"] == "http://model/v1"
        assert provider["models"][0]["id"] == "Qwen3.6-35B-A3B"

    def test_context_window_and_max_tokens_omitted_by_default(self) -> None:
        # Regression: a static default here previously caused every request to fail unconditionally
        # once it didn't match the deployed model's real context window. None (the default) must omit
        # both keys so openclaw falls back to its own auto-discovery / sane self-hosted defaults.
        agent = _make_agent(model_server=ModelServerRef(type="responses_api_models", name="policy_model"))
        with patch.object(agent, "_resolve_model_base_url", return_value="http://model/v1"):
            cfg = agent._build_openclaw_config({})

        model_entry = cfg["models"]["providers"]["nemo"]["models"][0]
        assert "contextWindow" not in model_entry
        assert "maxTokens" not in model_entry

    def test_context_window_and_max_tokens_included_when_set(self) -> None:
        agent = _make_agent(
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            context_window=131072,
            max_output_tokens=8192,
        )
        with patch.object(agent, "_resolve_model_base_url", return_value="http://model/v1"):
            cfg = agent._build_openclaw_config({})

        model_entry = cfg["models"]["providers"]["nemo"]["models"][0]
        assert model_entry["contextWindow"] == 131072
        assert model_entry["maxTokens"] == 8192

    def test_timeout_pads_empty_output(self) -> None:
        agent = _make_agent()

        async def _boom(*args, **kwargs):
            raise TimeoutError("openclaw timed out")

        body = NeMoGymResponseCreateParamsNonStreaming(input="solve it")
        with patch.object(agent, "_run_openclaw", _boom):
            resp = asyncio.run(agent.responses(MagicMock(), body))
        assert len(resp.output) == 1
        assert resp.output[0].content[0].text == ""
        assert resp.usage.total_tokens == 0

    def test_env_passthrough(self) -> None:
        agent = _make_agent(env={"NVIDIA_API_KEY": "k", "EMPTY": ""})
        env = agent._env(Path("/tmp/h"))
        assert env["NVIDIA_API_KEY"] == "k"
        assert env["HOME"] == "/tmp/h"
        assert "EMPTY" not in env


class TestObservability:
    def test_collects_session_artifact_before_workspace_cleanup(self, tmp_path: Path) -> None:
        agent = _make_agent(workspace_root=str(tmp_path))
        work_dir = tmp_path / "run"
        config_path = work_dir / ".openclaw-home" / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{}")
        sessions_dir = config_path.parent / "agents" / "main" / "sessions"
        sessions_dir.mkdir(parents=True)
        session_path = sessions_dir / "session-1.jsonl"
        session_path.write_text(
            "\n".join(
                [
                    json.dumps({"type": "session", "id": "session-1"}),
                    json.dumps(
                        {
                            "type": "message",
                            "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                        }
                    ),
                ]
            )
        )
        child_path = sessions_dir / "child-transcript.jsonl"
        child_path.write_text(
            "\n".join(
                [
                    json.dumps({"type": "session", "id": "child-transcript"}),
                    json.dumps(
                        {
                            "type": "message",
                            "message": {
                                "role": "assistant",
                                "content": [{"type": "text", "text": "child done"}],
                            },
                        }
                    ),
                ]
            )
        )
        (sessions_dir / "sessions.json").write_text(
            json.dumps(
                {
                    "agent:main:main": {"sessionId": "session-1", "sessionFile": str(session_path)},
                    "agent:main:subagent:child": {
                        "sessionId": "child-transcript",
                        "spawnedBy": "agent:main:main",
                    },
                }
            )
        )
        stdout = json.dumps({"payloads": [], "meta": {"agentMeta": {"sessionFile": str(session_path)}}})
        collector = MagicMock()

        with (
            patch.object(agent, "_workspace_root", return_value=work_dir),
            patch.object(agent, "_run_exec", AsyncMock(side_effect=[(0, "", ""), (0, stdout, "")])),
        ):
            output, _, _ = asyncio.run(agent._run_openclaw("solve", None, observation_collector=collector))

        assert output[0].content[0].text == "done"
        assert collector.call_args.args[0] == "session-1"
        assert collector.call_args.args[1][1]["type"] == "message"
        assert [(item[0], item[1]) for item in collector.call_args.args[2]] == [
            ("agent:main:main", None),
            ("agent:main:subagent:child", "agent:main:main"),
        ]
        assert not work_dir.exists()

    def test_scoring_padding_is_not_reported_as_agent_output(self) -> None:
        agent = _make_agent()

        async def run_openclaw(*args, observation_collector=None, **kwargs):
            observation_collector(
                "session-without-output",
                [{"type": "session", "id": "session-without-output"}],
                [],
                [],
            )
            return [], {"input_tokens": 0, "output_tokens": 0}, "model"

        body = NeMoGymResponseCreateParamsNonStreaming(input="solve")
        with patch.object(agent, "_run_openclaw", run_openclaw):
            episode = asyncio.run(agent._create_episode(body, rollout_id="1-2"))

        assert episode.response.output[0].content[0].text == ""
        [invocation] = _invocations(episode.observations)
        assert invocation.invocation_id == "session-without-output"
        assert invocation.conversation == [NeMoGymEasyInputMessage(role="user", content="solve")]
        assert "agent_transcript_unavailable" in {gap.code for gap in episode.observations.gaps}

    def test_fallback_output_is_preserved_without_a_session_transcript(self) -> None:
        agent = _make_agent()
        output = NeMoGymResponseOutputMessage(
            id="fallback",
            content=[{"type": "output_text", "text": "done", "annotations": []}],
        )

        async def run_openclaw(*args, **kwargs):
            return [output], {"input_tokens": 1, "output_tokens": 1}, "model"

        body = NeMoGymResponseCreateParamsNonStreaming(input="solve")
        with patch.object(agent, "_run_openclaw", run_openclaw):
            episode = asyncio.run(agent._create_episode(body, rollout_id="1-2"))

        [invocation] = _invocations(episode.observations)
        assert invocation.conversation == [
            NeMoGymEasyInputMessage(role="user", content="solve"),
            output,
        ]
        assert "agent_transcript_unavailable" in {gap.code for gap in episode.observations.gaps}

    def test_hierarchy_discovery_gap_replaces_generic_fallback(self) -> None:
        agent = _make_agent()

        async def run_openclaw(*args, observation_collector=None, **kwargs):
            observation_collector(
                "session-1",
                [],
                [],
                [ObservationGap(code="subagent_hierarchy_unavailable", detail="root_session_not_found")],
            )
            return [], {"input_tokens": 0, "output_tokens": 0}, "model"

        with patch.object(agent, "_run_openclaw", run_openclaw):
            episode = asyncio.run(
                agent._create_episode(
                    NeMoGymResponseCreateParamsNonStreaming(input="solve"),
                    rollout_id="1-2",
                )
            )

        hierarchy_gaps = [gap for gap in episode.observations.gaps if gap.code == "subagent_hierarchy_unavailable"]
        assert [(gap.code, gap.detail) for gap in hierarchy_gaps] == [
            ("subagent_hierarchy_unavailable", "root_session_not_found")
        ]

    def test_root_observation_uses_retained_transcript(self) -> None:
        agent = _make_agent()
        scored_output = NeMoGymResponseOutputMessage(
            id="scored",
            content=[{"type": "output_text", "text": "scoring view", "annotations": []}],
        )
        events = [
            {"type": "session", "id": "session-1"},
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "retained input"}],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "from transcript"}],
                },
            },
        ]

        async def run_openclaw(*args, observation_collector=None, **kwargs):
            observation_collector("session-1", events, [("root", None, events)], [])
            return [scored_output], {"input_tokens": 1, "output_tokens": 1}, "model"

        with patch.object(agent, "_run_openclaw", run_openclaw):
            episode = asyncio.run(
                agent._create_episode(
                    NeMoGymResponseCreateParamsNonStreaming(input="solve"),
                    rollout_id="1-2",
                )
            )

        [invocation] = _invocations(episode.observations)
        conversation = invocation.conversation
        assert conversation[0] == NeMoGymEasyInputMessage(role="user", content="retained input")
        assert conversation[1].content[0].text == "from transcript"
        assert "scoring view" not in str(conversation)

    def test_run_returns_observations_when_enabled(self) -> None:
        agent = _make_agent()
        agent.server_client.global_config_dict = {"observability_enabled": True}
        session = "\n".join(
            [
                json.dumps({"type": "session", "id": "session-1"}),
                json.dumps(
                    {
                        "type": "message",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": "done"}]},
                    }
                ),
            ]
        )

        async def run_openclaw(*args, observation_collector=None, **kwargs):
            observation_collector("session-1", parse_openclaw_session_events(session), [], [])
            return parse_openclaw_session(session), {"input_tokens": 1, "output_tokens": 1}, "model"

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path == "/seed_session":
                return _FakeResponse({}, {"session": "1"})
            if url_path.endswith("/v1/responses"):
                response = await agent.responses(MagicMock(path_params={"rollout_id": "1-2"}), json)
                return _FakeResponse(response.model_dump(mode="json"), cookies)
            return _FakeResponse(json | {"reward": 1.0})

        agent.server_client.post = AsyncMock(side_effect=post)
        request = MagicMock()
        request.cookies = {}
        body = OpenClawAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "solve"},
                "_ng_task_index": 1,
                "_ng_rollout_index": 2,
            }
        )
        with patch.object(agent, "_run_openclaw", run_openclaw):
            result = asyncio.run(agent.run(request, body))

        observations = result.ng_agent_observations
        assert observations is not None
        [invocation] = _invocations(observations)
        assert invocation.invocation_id == "session-1"
        assert invocation.conversation
        verify_json = agent.server_client.post.await_args_list[-1].kwargs["json"]
        assert "_ng_agent_observations" not in verify_json["response"]

    def test_observation_failure_does_not_change_response(self) -> None:
        agent = _make_agent()

        async def run_openclaw(*args, observation_collector=None, **kwargs):
            if observation_collector is not None:
                observation_collector("session-1", [], [], [])
            return (
                [
                    NeMoGymResponseOutputMessage(
                        id="msg-1",
                        content=[{"type": "output_text", "text": "done", "annotations": []}],
                    )
                ],
                {"input_tokens": 1, "output_tokens": 1},
                "model",
            )

        body = NeMoGymResponseCreateParamsNonStreaming(input="solve")
        with patch.object(agent, "_run_openclaw", run_openclaw):
            baseline = asyncio.run(agent.responses(MagicMock(), body))
        with (
            patch.object(agent, "_run_openclaw", run_openclaw),
            patch(
                "responses_api_agents.openclaw_agent.app.build_openclaw_observations",
                side_effect=RuntimeError("observer failed"),
            ),
        ):
            episode = asyncio.run(agent._create_episode(body, rollout_id="1-2"))

        assert episode.response.output == baseline.output
        assert episode.response.usage == baseline.usage
        assert [gap.code for gap in episode.observations.gaps] == ["observation_capture_failed"]


def _seed_session(home: Path, session_id: str, text: str, mtime: Optional[float] = None) -> Path:
    session_path = home / ".openclaw" / "agents" / "main" / "sessions" / f"{session_id}.jsonl"
    session_path.parent.mkdir(parents=True, exist_ok=True)
    session_path.write_text(
        "\n".join(
            [
                json.dumps({"type": "session", "id": session_id}),
                json.dumps(
                    {
                        "type": "message",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
                    }
                ),
            ]
        )
    )
    if mtime is not None:
        os.utime(session_path, (mtime, mtime))
    return session_path


class TestFindPartialSession:
    def test_returns_most_recently_written_parseable_session(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        older = _seed_session(home, "older", "old", mtime=1)
        newer = _seed_session(home, "newer", "new", mtime=2)

        assert OpenClawAgent._find_partial_session(home) == newer
        assert older != newer

    def test_skips_unparseable_files_and_returns_none_when_none_parse(self, tmp_path: Path) -> None:
        home = tmp_path / "home"
        home.mkdir()
        (home / "empty.jsonl").write_text("")
        (home / "garbage.jsonl").write_text("not json\n")

        assert OpenClawAgent._find_partial_session(home) is None

    def test_returns_none_when_home_missing(self, tmp_path: Path) -> None:
        assert OpenClawAgent._find_partial_session(tmp_path / "missing") is None


class TestTimeoutSalvage:
    def test_config_timeout_returns_partial_session_from_disk(self, tmp_path: Path) -> None:
        agent = _make_agent()
        work_dir = tmp_path / "run"
        home = work_dir / ".openclaw-home"
        config_path = home / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{}")
        _seed_session(home, "session-1", "partial")

        async def _run_exec_stub(cmd, *, cwd, env, timeout):
            if "onboard" in cmd:
                return 0, "", ""
            raise TimeoutError("openclaw timed out")

        with (
            patch.object(agent, "_workspace_root", return_value=work_dir),
            patch.object(agent, "_run_exec", _run_exec_stub),
        ):
            output, usage, _ = asyncio.run(agent._run_openclaw("solve", None))

        assert output
        assert output[0].content[0].text == "partial"
        assert not work_dir.exists()  # workdir cleanup still runs after salvage


class TestRunExecCancellation:
    def test_cancellation_kills_child_process(self) -> None:
        agent = _make_agent()
        captured: dict = {}
        orig_create = asyncio.create_subprocess_exec

        async def _capturing_create(*args, **kwargs):
            proc = await orig_create(*args, **kwargs)
            captured["proc"] = proc
            return proc

        async def _main():
            with patch("asyncio.create_subprocess_exec", _capturing_create):
                task = asyncio.ensure_future(
                    agent._run_exec(["sleep", "30"], cwd=None, env=os.environ.copy(), timeout=100)
                )
                for _ in range(200):
                    if "proc" in captured:
                        break
                    await asyncio.sleep(0.01)
                assert "proc" in captured, "subprocess never started"
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                await captured["proc"].wait()

        asyncio.run(_main())

        assert captured["proc"].returncode is not None  # cancellation killed it, not left running

    def test_cancellation_kills_descendant_processes(self) -> None:
        agent = _make_agent()
        captured: dict = {}
        orig_create = asyncio.create_subprocess_exec

        async def _capturing_create(*args, **kwargs):
            proc = await orig_create(*args, **kwargs)
            captured["proc"] = proc
            return proc

        async def _main():
            with patch("asyncio.create_subprocess_exec", _capturing_create):
                task = asyncio.ensure_future(
                    agent._run_exec(
                        [
                            sys.executable,
                            "-c",
                            "import subprocess, sys, time; "
                            "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
                            "time.sleep(30)",
                        ],
                        cwd=None,
                        env=os.environ.copy(),
                        timeout=100,
                    )
                )
                for _ in range(200):
                    proc = captured.get("proc")
                    children = psutil.Process(proc.pid).children(recursive=True) if proc else []
                    if children:
                        break
                    await asyncio.sleep(0.01)
                assert children, "descendant process never started"
                child_pids = [child.pid for child in children]
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
                return child_pids

        child_pids = asyncio.run(_main())

        assert captured["proc"].returncode is not None
        assert all(not psutil.pid_exists(pid) for pid in child_pids)


class TestSigtermSalvage:
    def test_sigterm_returns_partial_session_from_disk(self, tmp_path: Path) -> None:
        agent = _make_agent()
        work_dir = tmp_path / "run"
        home = work_dir / ".openclaw-home"
        config_path = home / ".openclaw" / "openclaw.json"
        config_path.parent.mkdir(parents=True)
        config_path.write_text("{}")
        _seed_session(home, "session-1", "partial")

        registered: dict = {}

        async def _run_exec_stub(cmd, *, cwd, env, timeout):
            if "onboard" in cmd:
                return 0, "", ""
            await asyncio.Event().wait()  # hangs until the SIGTERM path cancels it

        async def _main():
            task = asyncio.ensure_future(agent._run_openclaw("solve", None))
            for _ in range(100):
                if signal.SIGTERM in registered:
                    break
                await asyncio.sleep(0)
            assert signal.SIGTERM in registered, "SIGTERM handler was never installed"
            registered[signal.SIGTERM](signal.SIGTERM, None)
            return await task

        with (
            patch.object(agent, "_workspace_root", return_value=work_dir),
            patch.object(agent, "_run_exec", _run_exec_stub),
            patch("signal.getsignal", return_value=signal.SIG_DFL),
            patch("signal.signal", side_effect=lambda sig, cb: registered.__setitem__(sig, cb)),
        ):
            output, usage, _ = asyncio.run(_main())

        assert output
        assert output[0].content[0].text == "partial"
        assert not work_dir.exists()

    def test_handler_installed_once_and_chains_to_previous_across_concurrent_runs(self, tmp_path: Path) -> None:
        agent = _make_agent()
        work_dir_a, work_dir_b = tmp_path / "run_a", tmp_path / "run_b"
        for work_dir in (work_dir_a, work_dir_b):
            home = work_dir / ".openclaw-home"
            config_path = home / ".openclaw" / "openclaw.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("{}")
            _seed_session(home, "session-1", "partial")

        async def _run_exec_stub(cmd, *, cwd, env, timeout):
            if "onboard" in cmd:
                return 0, "", ""
            await asyncio.Event().wait()  # hangs until the SIGTERM path cancels it

        install_calls = 0
        previous_calls = 0

        def _fake_previous() -> None:
            nonlocal previous_calls
            previous_calls += 1

        async def _main():
            loop = asyncio.get_running_loop()
            loop.add_signal_handler(signal.SIGTERM, _fake_previous)
            process_handler = signal.getsignal(signal.SIGTERM)
            real_signal = signal.signal

            def _install_signal(sig, handler):
                nonlocal install_calls
                install_calls += 1
                return real_signal(sig, handler)

            try:
                work_dirs = iter([work_dir_a, work_dir_b])
                with (
                    patch.object(agent, "_workspace_root", side_effect=lambda: next(work_dirs)),
                    patch.object(agent, "_run_exec", _run_exec_stub),
                    patch("signal.signal", side_effect=_install_signal),
                ):
                    task_a = asyncio.ensure_future(agent._run_openclaw("solve", None))
                    task_b = asyncio.ensure_future(agent._run_openclaw("solve", None))
                    for _ in range(100):
                        if len(agent.sigterm_events) == 2:
                            break
                        await asyncio.sleep(0)
                    assert len(agent.sigterm_events) == 2
                    os.kill(os.getpid(), signal.SIGTERM)
                    return await asyncio.gather(task_a, task_b)
            finally:
                real_signal(signal.SIGTERM, process_handler)
                loop.remove_signal_handler(signal.SIGTERM)

        results = asyncio.run(_main())

        assert install_calls == 1  # installed once, not once per run
        assert previous_calls == 1  # the previously-installed handler (uvicorn's) still fires
        for output, _usage, _model in results:
            assert output[0].content[0].text == "partial"


class TestDeepMerge:
    def test_nested_merge(self) -> None:
        base = {"a": {"b": 1, "c": 2}}
        OpenClawAgent._deep_merge(base, {"a": {"c": 3, "d": 4}})
        assert base == {"a": {"b": 1, "c": 3, "d": 4}}


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "openclaw_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "openclaw_agent" in data
        inner = data["openclaw_agent"]["responses_api_agents"]["openclaw_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 32
        assert inner["command"] == "openclaw"

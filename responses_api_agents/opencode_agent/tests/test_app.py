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

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.rollout_observability import (
    AgentInvocation,
    AgentObservationBundle,
    ContextCompactionObservation,
    ToolCallObservation,
)
from nemo_gym.server_utils import ServerClient
from responses_api_agents.opencode_agent.app import (
    OpenCodeAgent,
    OpenCodeAgentConfig,
    OpenCodeAgentRunRequest,
    _extract_instruction,
    _parse_opencode_session,
    parse_opencode_session,
)


def _config(**kwargs) -> OpenCodeAgentConfig:
    return OpenCodeAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        **kwargs,
    )


def _invocations(bundle: AgentObservationBundle) -> list[AgentInvocation]:
    return [record for record in bundle.records if isinstance(record, AgentInvocation)]


def _tool_calls(bundle: AgentObservationBundle) -> list[ToolCallObservation]:
    return [record for record in bundle.records if isinstance(record, ToolCallObservation)]


def _compactions(bundle: AgentObservationBundle) -> list[ContextCompactionObservation]:
    return [record for record in bundle.records if isinstance(record, ContextCompactionObservation)]


def _make_agent(**kwargs) -> OpenCodeAgent:
    with patch("responses_api_agents.opencode_agent.app.OpenCodeAgent.model_post_init"):
        agent = OpenCodeAgent(config=_config(**kwargs), server_client=MagicMock(spec=ServerClient))
    agent.sem = asyncio.Semaphore(agent.config.concurrency)
    return agent


def _session_db(tmp_path, messages, sessions=None) -> Path:
    """Build the subset of OpenCode's v1.17.11 artifact used by the adapter."""
    import sqlite3

    db = tmp_path / "opencode.db"
    con = sqlite3.connect(db)
    sessions = sessions or [("root", None)]
    con.execute("create table session (id text, parent_id text, time_created integer)")
    con.execute("create table message (id text, session_id text, data text, time_created integer)")
    con.execute("create table part (id text, message_id text, session_id text, data text, time_created integer)")
    for index, (session_id, parent_id) in enumerate(sessions):
        con.execute("insert into session values (?,?,?)", (session_id, parent_id, index))
    t = 0
    for mi, entry in enumerate(messages):
        session_id, role_or_message, parts = ("root", *entry) if len(entry) == 2 else entry
        mid = f"m{mi}"
        message = (
            {"role": role_or_message, "time": {"created": mi, "completed": mi + 1}}
            if isinstance(role_or_message, str)
            else role_or_message
        )
        con.execute("insert into message values (?,?,?,?)", (mid, session_id, json.dumps(message), mi))
        for p in parts:
            con.execute("insert into part values (?,?,?,?,?)", (f"p{t}", mid, session_id, json.dumps(p), t))
            t += 1
    con.commit()
    con.close()
    return db


class TestSanity:
    def test_config_defaults(self) -> None:
        cfg = _config()
        assert cfg.concurrency == 8
        assert cfg.command == "opencode"
        assert cfg.thinking is True
        assert cfg.command_parts == ["opencode"]

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


class TestParseOpencodeSession:
    def test_missing_db(self, tmp_path) -> None:
        items, usage = parse_opencode_session(tmp_path / "nope.db")
        assert items == []
        assert usage == {"input_tokens": 0, "output_tokens": 0}

    def test_assistant_text(self, tmp_path) -> None:
        db = _session_db(tmp_path, [("assistant", [{"type": "text", "text": "the answer is 4"}])])
        items, _ = parse_opencode_session(db)
        assert len(items) == 1
        assert isinstance(items[0], NeMoGymResponseOutputMessage)
        assert items[0].content[0].text == "the answer is 4"

    def test_user_parts_ignored(self, tmp_path) -> None:
        db = _session_db(tmp_path, [("user", [{"type": "text", "text": "hi"}])])
        items, _ = parse_opencode_session(db)
        assert items == []

    def test_tool_call_and_output(self, tmp_path) -> None:
        db = _session_db(
            tmp_path,
            [
                (
                    "assistant",
                    [
                        {
                            "type": "tool",
                            "callID": "c1",
                            "tool": "bash",
                            "state": {
                                "status": "completed",
                                "input": {"command": "echo 6"},
                                "output": "6\n",
                                "time": {"start": 1000, "end": 1200},
                            },
                        },
                        {"type": "text", "text": "answer is 6"},
                    ],
                )
            ],
        )
        items, _ = parse_opencode_session(db)
        assert isinstance(items[0], NeMoGymResponseFunctionToolCall)
        assert items[0].name == "bash"
        assert json.loads(items[0].arguments)["command"] == "echo 6"
        assert isinstance(items[1], NeMoGymFunctionCallOutput)
        assert items[1].call_id == "c1"
        assert "6" in items[1].output
        assert isinstance(items[2], NeMoGymResponseOutputMessage)

    def test_step_finish_usage(self, tmp_path) -> None:
        db = _session_db(
            tmp_path,
            [("assistant", [{"type": "step-finish", "tokens": {"input": 100, "output": 20, "cache": {"read": 5}}}])],
        )
        _, usage = parse_opencode_session(db)
        assert usage["input_tokens"] == 105
        assert usage["output_tokens"] == 20

    def test_preserves_tree_parallel_tools_compaction_and_reasoning(self, tmp_path) -> None:
        db = _session_db(
            tmp_path,
            [
                (
                    "root",
                    "user",
                    [
                        {"type": "text", "text": "solve"},
                        {"type": "text", "text": "not replayed", "ignored": True},
                    ],
                ),
                (
                    "root",
                    "assistant",
                    [
                        {
                            "type": "tool",
                            "callID": "task-1",
                            "tool": "task",
                            "state": {
                                "status": "completed",
                                "input": {"prompt": "inspect"},
                                "output": "done",
                                "metadata": {"sessionId": "child"},
                                "time": {"start": 1785813051824, "end": 1785813053824},
                            },
                        },
                        {
                            "type": "tool",
                            "callID": "bash-1",
                            "tool": "bash",
                            "state": {
                                "status": "completed",
                                "input": {"command": "pwd"},
                                "output": "/workspace",
                                "time": {
                                    "start": 1785813052000,
                                    "end": 1785813052400,
                                    "compacted": 1785813052500,
                                },
                            },
                        },
                    ],
                ),
                ("child", "user", [{"type": "text", "text": "inspect"}]),
                ("child", "assistant", [{"type": "reasoning", "text": "checking files"}]),
                (
                    "child",
                    "user",
                    [{"type": "compaction", "auto": True, "overflow": True, "tail_start_id": "m3"}],
                ),
                (
                    "child",
                    {
                        "role": "assistant",
                        "summary": True,
                        "parentID": "m4",
                        "time": {"created": 5, "completed": 6},
                    },
                    [{"type": "text", "text": "condensed context"}],
                ),
            ],
            sessions=[("root", None), ("child", "root")],
        )

        bundle = _parse_opencode_session(db, "fallback")

        root, child = _invocations(bundle)
        assert child.parent_invocation_id == root.invocation_id
        assert child.spawned_by_tool_call_id == "task-1"
        assert any(item.type == "reasoning" for item in child.conversation)
        assert all(getattr(item, "content", None) != "not replayed" for item in root.conversation)
        assert (
            next(
                item.output
                for item in root.conversation
                if isinstance(item, NeMoGymFunctionCallOutput) and item.call_id == "bash-1"
            )
            == "[Old tool result content cleared]"
        )
        tools = {tool.tool_call_id: tool for tool in _tool_calls(bundle)}
        assert {tool_id: tool.duration_ms for tool_id, tool in tools.items()} == {
            "task-1": 2000,
            "bash-1": 400,
        }
        assert tools["task-1"].started_at == 1785813051.824
        assert tools["task-1"].completed_at == 1785813053.824
        compaction = _compactions(bundle)[0]
        assert compaction.trigger == "overflow"
        assert compaction.summary == "condensed context"
        assert compaction.first_kept_item_id == "p5"
        assert "compaction_model_call_boundary_unavailable" in {gap.code for gap in bundle.gaps}

    def test_reports_unaddressable_compaction_boundary(self, tmp_path) -> None:
        db = _session_db(
            tmp_path,
            [
                ("user", [{"type": "text", "text": "keep this"}]),
                ("user", [{"type": "compaction", "tail_start_id": "m0"}]),
            ],
        )

        bundle = _parse_opencode_session(db, "fallback")

        assert _compactions(bundle)[0].first_kept_item_id is None
        assert any(gap.code == "compaction_first_kept_item_unavailable" and gap.detail == "m0" for gap in bundle.gaps)

    def test_preserves_parts_owned_by_session_when_message_is_unowned(self, tmp_path) -> None:
        import sqlite3

        db = _session_db(tmp_path, [("assistant", [{"type": "text", "text": "kept"}])])
        con = sqlite3.connect(db)
        con.execute("update message set session_id = NULL where id = 'm0'")
        con.commit()
        con.close()

        bundle = _parse_opencode_session(db, "fallback")

        root = _invocations(bundle)[0]
        assert root.status == "unknown"
        assert root.conversation[0].content[0].text == "kept"
        assert any(gap.code == "agent_artifact_record_unowned" and gap.detail == "m0" for gap in bundle.gaps)

    def test_reports_invalid_timing_and_unresolved_tree_edges(self, tmp_path) -> None:
        db = _session_db(
            tmp_path,
            [
                (
                    "root",
                    "assistant",
                    [
                        {
                            "type": "tool",
                            "callID": call_id,
                            "tool": "task",
                            "state": {
                                "status": "completed",
                                "metadata": {"sessionId": "child"},
                                "time": {"start": 2000, "end": 1000},
                            },
                        }
                        for call_id in ("task-1", "task-2")
                    ],
                )
            ],
            sessions=[("root", None), ("child", "missing-parent")],
        )

        bundle = _parse_opencode_session(db, "fallback")

        assert all(tool.started_at is None and tool.completed_at is None for tool in _tool_calls(bundle))
        assert {gap.code for gap in bundle.gaps} >= {
            "tool_timing_unavailable",
            "subagent_parent_unavailable",
            "subagent_spawn_ambiguous",
        }


class TestDeepMerge:
    def test_nested_merge(self) -> None:
        base = {"a": {"b": 1, "c": 2}}
        OpenCodeAgent._deep_merge(base, {"a": {"c": 3, "d": 4}})
        assert base == {"a": {"b": 1, "c": 3, "d": 4}}


class TestEnv:
    def test_env_passthrough(self) -> None:
        agent = _make_agent(openai_api_key="k", openai_base_url="https://x/v1", env={"FOO": "bar", "EMPTY": ""})
        env = agent._env("/tmp/data")
        assert env["OPENAI_API_KEY"] == "k"
        assert env["OPENAI_BASE_URL"] == "https://x/v1"
        assert env["XDG_DATA_HOME"] == "/tmp/data"
        assert env["FOO"] == "bar"
        assert "EMPTY" not in env

    def test_model_server_builds_local_provider(self) -> None:
        agent = _make_agent(
            model="Qwen3.6-35B-A3B",
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        )
        with patch.object(agent, "_resolve_model_base_url", return_value="http://model/v1"):
            env = agent._env("/tmp/data")
            config = agent._build_opencode_config()

        provider = config["provider"]["nemo"]
        assert agent._effective_model() == "nemo/Qwen3.6-35B-A3B"
        assert env["OPENAI_BASE_URL"] == "http://model/v1"
        assert provider["options"]["baseURL"] == "http://model/v1"
        assert provider["models"]["Qwen3.6-35B-A3B"]["limit"]["output"] == 131072


class TestRolloutObservability:
    def test_routes_model_server_without_mutating_config(self, tmp_path: Path) -> None:
        opencode_config = {"provider": {"openai": {"options": {"baseURL": "https://api.openai.com/v1"}}}}
        agent = _make_agent(
            model_server=ModelServerRef(type="responses_api_models", name="policy"),
            opencode_config=opencode_config,
            env={"OPENAI_BASE_URL": "https://wrong.invalid/v1"},
        )

        with patch.object(OpenCodeAgent, "resolve_model_base_url", return_value="http://policy/ng-rollout/1-2/v1"):
            agent._write_opencode_config(tmp_path, "1-2")
            env = agent._env(str(tmp_path), "1-2")

        written = json.loads((tmp_path / "opencode.json").read_text())
        assert written["provider"]["nemo"]["options"]["baseURL"] == "http://policy/ng-rollout/1-2/v1"
        assert written["provider"]["openai"]["options"]["baseURL"] == "https://api.openai.com/v1"
        assert env["OPENAI_BASE_URL"] == "http://policy/ng-rollout/1-2/v1"
        assert env["OPENAI_API_KEY"] == "EMPTY"  # pragma: allowlist secret
        assert agent.config.opencode_config == opencode_config

    def test_padding_is_not_reported_as_artifact_evidence(self, tmp_path: Path) -> None:
        _, usage = parse_opencode_session(tmp_path / "missing.db")
        observations = _parse_opencode_session(tmp_path / "missing.db", "1-2")
        agent = _make_agent(system_prompt="configured system")
        agent._run_opencode = AsyncMock(return_value=([], usage, "model", observations))
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                NeMoGymEasyInputMessage(role="system", content="request system"),
                NeMoGymEasyInputMessage(role="user", content="old question"),
                NeMoGymEasyInputMessage(role="assistant", content="old answer"),
                NeMoGymEasyInputMessage(role="user", content="solve"),
            ]
        )

        episode = asyncio.run(agent._create_episode(body, rollout_id="1-2"))

        assert episode.response.output
        assert agent._run_opencode.await_args.args == ("solve", "configured system\n\nrequest system")
        assert _invocations(episode.observations)[0].conversation == [
            NeMoGymEasyInputMessage(role="user", content="configured system\n\nrequest system\n\nsolve")
        ]
        assert "agent_transcript_unavailable" in {gap.code for gap in episode.observations.gaps}

    def test_run_attaches_artifact_observations_when_enabled(self, tmp_path: Path) -> None:
        db = _session_db(tmp_path, [("assistant", [{"type": "text", "text": "done"}])])
        items, usage = parse_opencode_session(db)
        observations = _parse_opencode_session(db, "1-2")
        agent = _make_agent()
        agent.server_client.global_config_dict = {"observability_enabled": True}
        agent._run_opencode = AsyncMock(return_value=(items, usage, "model", observations))

        class Response:
            ok = True
            cookies = {}

            def __init__(self, payload):
                self.payload = payload

            async def read(self):
                return json.dumps(self.payload).encode()

        async def post(server_name, url_path, json=None, cookies=None, **kwargs):
            if url_path.endswith("/v1/responses"):
                response = await agent.responses(MagicMock(path_params={"rollout_id": "1-2"}), json)
                return Response(response.model_dump(mode="json"))
            return Response(json | {"reward": 1.0}) if url_path == "/verify" else Response({})

        agent.server_client.post = AsyncMock(side_effect=post)
        request = MagicMock(cookies={})
        body = OpenCodeAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "solve"},
                "_ng_task_index": 1,
                "_ng_rollout_index": 2,
            }
        )

        result = asyncio.run(agent.run(request, body))

        assert result.ng_agent_observations is not None
        assert _invocations(result.ng_agent_observations)[0].conversation
        assert agent._run_opencode.await_args.kwargs["rollout_id"] == "1-2"
        assert agent.server_client.post.await_args_list[1].kwargs["url_path"] == "/ng-rollout/1-2/v1/responses"
        verify_json = agent.server_client.post.await_args_list[2].kwargs["json"]
        assert "_ng_agent_observations" not in verify_json["response"]


class TestRepoDir:
    def test_creates_configured_repo_dir(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "nested" / "repo"
        agent = _make_agent(repo_dir=str(repo_dir))

        assert agent._repo_dir(tmp_path / "fallback") == repo_dir
        assert repo_dir.is_dir()

    async def test_preserves_configured_repo_and_cleans_workspace(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        repo_dir = tmp_path / "repo"
        process = MagicMock(returncode=0)
        process.communicate = AsyncMock(return_value=(b"", b""))
        agent = _make_agent(repo_dir=str(repo_dir))
        scored = [NeMoGymResponseOutputMessage(id="scored", content=[])]

        with (
            patch.object(agent, "_workspace_root", return_value=workspace),
            patch(
                "responses_api_agents.opencode_agent.app.asyncio.create_subprocess_exec",
                AsyncMock(return_value=process),
            ),
            patch(
                "responses_api_agents.opencode_agent.app.parse_opencode_session",
                return_value=(scored, {"input_tokens": 1, "output_tokens": 2}),
            ),
            patch(
                "responses_api_agents.opencode_agent.app._parse_opencode_session",
                side_effect=ValueError("invalid observation artifact"),
            ),
        ):
            output, usage, _, observations = await agent._run_opencode(
                "fix the issue", None, collect_observations=True
            )

        assert output == scored
        assert usage == {"input_tokens": 1, "output_tokens": 2}
        assert "agent_artifact_unavailable" in {gap.code for gap in observations.gaps}
        assert repo_dir.is_dir()
        assert not workspace.exists()


class TestConfigYaml:
    def test_module_parses(self) -> None:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        compile(app_path.read_text(), str(app_path), "exec")

    def test_config_yaml_parses(self) -> None:
        cfg_path = Path(__file__).resolve().parent.parent / "configs" / "opencode_agent.yaml"
        data = yaml.safe_load(cfg_path.read_text())
        assert "opencode_agent" in data
        inner = data["opencode_agent"]["responses_api_agents"]["opencode_agent"]
        assert inner["entrypoint"] == "app.py"
        assert inner["concurrency"] == 8
        assert inner["command"] == "opencode"

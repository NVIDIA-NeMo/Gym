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

import io
import json
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

from nemo_gym.server_utils import ServerClient
from responses_api_agents.qwen_code_agent import app
from responses_api_agents.qwen_code_agent.app import (
    QwenCodeAgent,
    QwenCodeAgentConfig,
    QwenCodeAgentRunRequest,
    parse_stream,
    summarize_subagents,
)


ROOT_EVENTS = [
    {"type": "system", "subtype": "init", "session_id": "s1"},
    {
        "type": "assistant",
        "parent_tool_use_id": None,
        "message": {"content": [{"type": "thinking", "thinking": "plan"}]},
    },
    {
        "type": "assistant",
        "parent_tool_use_id": None,
        "message": {
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "tool_call",
                    "input": {"name": "agent", "arguments": {"prompt": "x"}},
                }
            ]
        },
    },
    # Child events stay out of the root trajectory.
    {
        "type": "assistant",
        "parent_tool_use_id": "call-1",
        "message": {"content": [{"type": "text", "text": "child work"}]},
    },
    {
        "type": "user",
        "parent_tool_use_id": None,
        "message": {"content": [{"type": "tool_result", "tool_use_id": "call-1", "content": [{"text": "42"}]}]},
    },
    {
        "type": "assistant",
        "parent_tool_use_id": None,
        "message": {"content": [{"type": "text", "text": "<think>check</think>answer 42"}]},
    },
    {"type": "result", "is_error": False, "result": "answer 42", "usage": {"input_tokens": 100, "output_tokens": 7}},
]


def subagent_archive(path, metas):
    """Write a sessions archive shaped like Qwen Code's ~/.qwen/projects export."""
    with tarfile.open(path, "w:gz") as archive:
        for name, data in [
            (f"projects/p/subagents/s1/agent-{i}.meta.json", json.dumps(meta)) for i, meta in enumerate(metas)
        ] + [
            (
                "projects/p/subagents/s1/agent-0.jsonl",
                json.dumps({"usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 4}}),
            ),
            ("projects/p/chats/s1.jsonl", json.dumps({"usageMetadata": {"promptTokenCount": 999}})),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(data.encode())
            archive.addfile(info, io.BytesIO(data.encode()))


def make_config(**overrides):
    return QwenCodeAgentConfig(
        name="qwen",
        host="127.0.0.1",
        port=9000,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "grader"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model="root-model",
        sandbox_provider="sandbox",
        sandbox_config={"image": "python:3.12-slim"},
        **overrides,
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "sandbox_server_url", lambda name, **kwargs: f"http://{name}.example:8000")
    client = MagicMock(spec=ServerClient)

    async def post(**kwargs):
        if kwargs["url_path"] != "/verify":
            return SimpleNamespace(cookies={}, ok=True, read=AsyncMock(return_value=b"{}"))
        reward = float(any(item["type"] == "message" for item in kwargs["json"]["response"]["output"]))
        body = json.dumps(kwargs["json"] | {"reward": reward})
        return SimpleNamespace(cookies={}, ok=True, read=AsyncMock(return_value=body))

    client.post = AsyncMock(side_effect=post)
    server = QwenCodeAgent(
        config=make_config(
            artifacts_dir=str(tmp_path / "artifacts"),
            subagent_model="worker",
            subagent_model_server={"type": "responses_api_models", "name": "worker_model"},
            max_subagent_depth=2,
            execution_failure_reward_zero=True,
        ),
        server_client=client,
    )
    monkeypatch.setattr(server, "_capture_correlation_enabled", lambda: True)

    staged = {}

    async def upload(local, remote):
        staged[remote] = local.read_text()

    async def exec_(command, **kwargs):
        if "--version" in command:
            return SimpleNamespace(return_code=0, error_type=None, stdout="tar: warning\n0.25.1\n", stderr="")
        return SimpleNamespace(return_code=0, error_type=None, stdout="", stderr="")

    async def download(remote, local):
        if remote.endswith("stdout.jsonl"):
            local.write_text("\n".join(json.dumps(event) for event in ROOT_EVENTS))
        elif remote.endswith("sessions.tar.gz"):
            subagent_archive(local, [{"depth": 0, "isBackgrounded": True}, {"depth": 1}])
        else:
            local.write_text("")

    sandbox = SimpleNamespace(
        upload=AsyncMock(side_effect=upload),
        exec=AsyncMock(side_effect=exec_),
        download=AsyncMock(side_effect=download),
        stop=AsyncMock(),
    )
    server._start_sandbox = AsyncMock(return_value=sandbox)
    return server, sandbox, staged


def request_body():
    return QwenCodeAgentRunRequest.model_validate(
        {
            "_ng_rollout_id": "r1",
            "responses_create_params": {
                "input": [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Compute 6*7"}]
            },
        }
    )


def test_parse_stream_keeps_root_trajectory_and_result():
    output, result = parse_stream("\n".join(json.dumps(e) for e in ROOT_EVENTS) + "\nnot json")
    types = [item.type for item in output]
    assert types == ["reasoning", "function_call", "function_call_output", "reasoning", "message"]
    assert output[0].summary[0].text == "plan" and output[3].summary[0].text == "check"
    assert output[1].name == "agent" and json.loads(output[1].arguments) == {"prompt": "x"}
    assert output[2].call_id == "call-1" and output[2].output == "42"
    assert output[4].content[0].text == "answer 42"
    assert result["usage"]["input_tokens"] == 100


def test_summarize_subagents_counts_children_depth_and_tokens(tmp_path):
    subagent_archive(tmp_path / "s.tgz", [{"depth": 0, "isBackgrounded": True}, {"depth": 2}, {"depth": 0}])
    assert summarize_subagents(tmp_path / "s.tgz") == {
        "qwen_subagents": 3,
        "qwen_subagents_background": 1,
        "qwen_subagent_max_depth": 3,
        "qwen_subagent_input_tokens": 30,
        "qwen_subagent_output_tokens": 4,
    }


def test_summarize_subagents_skips_truncated_and_corrupt_files(tmp_path):
    # A child killed mid-write leaves a partial last transcript line or a cut meta file.
    good = json.dumps({"usageMetadata": {"promptTokenCount": 30, "candidatesTokenCount": 4}})
    files = {
        "projects/p/subagents/s1/agent-0.meta.json": json.dumps({"depth": 0}),
        "projects/p/subagents/s1/agent-1.meta.json": '{"depth": 1, "isBackgr',
        "projects/p/subagents/s1/agent-0.jsonl": good + "\n" + '{"usageMetadata": {"promptTok' + "\n[1, 2]",
    }
    with tarfile.open(tmp_path / "s.tgz", "w:gz") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data.encode())
            archive.addfile(info, io.BytesIO(data.encode()))
    assert summarize_subagents(tmp_path / "s.tgz") == {
        "qwen_subagents": 1,
        "qwen_subagents_background": 0,
        "qwen_subagent_max_depth": 1,
        "qwen_subagent_input_tokens": 30,
        "qwen_subagent_output_tokens": 4,
    }


async def test_run_configures_routes_subagents_and_verifies(agent):
    server, sandbox, staged = agent
    result = await server.run(SimpleNamespace(cookies={}), request_body())

    assert result.reward == 1 and not result.qwen_failed
    assert result.qwen_subagents == 2 and result.qwen_subagent_max_depth == 2
    assert result.qwen_subagents_background == 1
    assert result.response.usage.input_tokens == 100

    settings = next(json.loads(v) for p, v in staged.items() if p.endswith("settings.json"))
    assert settings["modelProviders"]["openai"] == [
        {
            "id": "root-model",
            "baseUrl": "http://policy_model.example:8000/ng-rollout/r1/v1",
            "envKey": app.API_KEY_ENV,
        },
        {"id": "worker", "baseUrl": "http://worker_model.example:8000/ng-rollout/r1/v1", "envKey": app.API_KEY_ENV},
    ]
    assert settings["memory"] == {"enableManagedAutoMemory": False}
    override = next(v for p, v in staged.items() if p.endswith("agents/general-purpose.md"))
    assert "\nmodel: worker\n" in override
    assert next(v for p, v in staged.items() if p.endswith("prompt.txt")) == "Compute 6*7"

    command = next(c.kwargs["command"] for c in sandbox.exec.await_args_list if "stream-json" in c.kwargs["command"])
    assert "--max-subagent-depth 2" in command and "--exclude-tools" not in command
    assert "--append-system-prompt 'Be terse.'" in command and "prompt.txt > " in command
    sandbox.stop.assert_awaited_once()
    generation = json.loads((app.Path(result.qwen_results_dir) / "generation.json").read_text())
    assert generation["qwen_subagents"] == 2 and generation["response"]["output"]


async def test_failed_run_is_scored_as_empty_response(agent):
    server, sandbox, _ = agent
    run_exec = sandbox.exec.side_effect

    async def failing(command, **kwargs):
        if "stream-json" in command:
            return SimpleNamespace(return_code=55, error_type=None, stdout="", stderr="")
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = failing
    result = await server.run(SimpleNamespace(cookies={}), request_body())
    assert result.qwen_failed and result.qwen_exit_code == 55 and result.reward == 0
    assert server.server_client.post.await_args.kwargs["json"]["response"]["output"] == []
    assert result.response.output  # The generation is retained for inspection.


async def test_version_mismatch_is_a_setup_error(agent):
    server, sandbox, _ = agent
    sandbox.exec.side_effect = AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None, stdout="0.1\n"))
    with pytest.raises(RuntimeError, match="does not match qwen_version"):
        await server.run(SimpleNamespace(cookies={}), request_body())
    sandbox.stop.assert_awaited_once()


def test_solo_and_team_settings():
    solo = QwenCodeAgent(
        config=make_config(subagents=False, max_session_turns=30), server_client=MagicMock(spec=ServerClient)
    )
    command = solo._build_command(None)
    assert command[command.index("--max-session-turns") + 1] == "30"
    assert [command[i + 1] for i, arg in enumerate(command) if arg == "--exclude-tools"] == app.SUBAGENT_TOOLS

    team = QwenCodeAgent(
        config=make_config(agent_team=True, max_parallel_agents=64, settings={"model": {"maxSessionTurns": 9}}),
        server_client=MagicMock(spec=ServerClient),
    )
    team._model_url = lambda server, body: "http://m/v1"
    mcp = {"tools": {"url": "http://t/mcp", "headers": {"X-Token": "t"}, "enabled": True, "timeout": 5}}
    settings = team._build_settings(request_body(), mcp)
    assert settings["experimental"] == {"agentTeam": True} and settings["agents"] == {"maxParallelAgents": 64}
    assert settings["model"] == {"name": "root-model", "maxSessionTurns": 9}
    assert settings["mcpServers"] == {
        "tools": {"httpUrl": "http://t/mcp", "headers": {"X-Token": "t"}, "timeout": 5, "trust": True}
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"subagents": False, "subagent_model": "w"},
        {"subagents": False, "agent_team": True},
        {"subagent_model": "root-model"},
    ],
)
def test_invalid_subagent_settings(overrides):
    with pytest.raises(ValueError):
        make_config(**overrides)


def test_request_controls_reach_qwen():
    agent = QwenCodeAgent(config=make_config(max_tool_calls=50), server_client=MagicMock(spec=ServerClient))
    agent._model_url = lambda server, body: "http://m/v1"
    body = request_body()
    body.responses_create_params.temperature = 0.3
    body.responses_create_params.top_p = 0.9
    body.responses_create_params.max_output_tokens = 1000
    settings = agent._build_settings(body, {})
    assert settings["model"]["generationConfig"] == {
        "samplingParams": {"temperature": 0.3, "top_p": 0.9, "max_tokens": 1000}
    }
    command = agent._build_command(None, max_tool_calls=7)
    assert command[command.index("--max-tool-calls") + 1] == "7"
    command = agent._build_command(None)
    assert command[command.index("--max-tool-calls") + 1] == "50"


@pytest.mark.parametrize(
    "item, message",
    [
        ({"role": "assistant", "content": "earlier answer"}, "not conversation history"),
        ({"type": "function_call_output", "call_id": "c1", "output": "x"}, "not conversation history"),
        ({"role": "user", "content": [{"type": "input_image", "image_url": "data:,"}]}, "only supports text input"),
    ],
)
def test_history_and_non_text_input_are_rejected(item, message):
    params = request_body().responses_create_params.model_copy(
        update={"input": [{"role": "user", "content": "Go"}, item]}
    )
    with pytest.raises(HTTPException, match=message) as error:
        QwenCodeAgent._instruction(params)
    assert error.value.status_code == 422

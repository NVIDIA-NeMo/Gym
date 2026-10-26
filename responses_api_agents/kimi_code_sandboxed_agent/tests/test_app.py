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
import io
import json
import tarfile
import tomllib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.rollout_observability import AgentInvocation
from nemo_gym.server_utils import ServerClient
from responses_api_agents.kimi_code_sandboxed_agent import app
from responses_api_agents.kimi_code_sandboxed_agent.app import (
    KimiCodeSandboxedAgent,
    KimiCodeSandboxedAgentConfig,
    KimiCodeSandboxedAgentRunRequest,
    max_concurrency,
    parse_session,
    to_toml,
)


STDOUT = [
    {"role": "meta", "type": "system.version", "version": "2.1.1"},
    {
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": "call-1",
                "function": {"name": "AgentSwarm", "arguments": '{"items":["3","5"]}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call-1", "content": "<agent_swarm_result>9, 25</agent_swarm_result>"},
    {"role": "assistant", "content": "<think>combine</think>9 and 25"},
    {"role": "meta", "type": "session.resume_hint", "session_id": "session_1"},
]


def usage(agent: str, tokens: int) -> dict:
    return {
        "type": "usage.record",
        "agentId": agent,
        "usage": {"inputOther": tokens, "output": 1, "inputCacheRead": 2},
    }


WIRES = {
    "main": [
        usage("main", 100),
        *(
            {
                "type": "subagent.spawned",
                "subagentId": f"agent-{i}",
                "subagentName": "coder",
                "parentAgentId": "main",
                "parentToolCallId": "call-1",
                "swarmIndex": i + 1,
                "model": "subagent",
            }
            for i in range(2)
        ),
        {"type": "subagent.started", "subagentId": "agent-0", "time": 1000},
        {"type": "subagent.started", "subagentId": "agent-1", "time": 1500},
        {"type": "subagent.completed", "subagentId": "agent-0", "time": 3000},
        {"type": "subagent.failed", "subagentId": "agent-1", "time": 2000},
        usage("main", 200),
    ],
    "agent-0": [usage("agent-0", 10)],
    "agent-1": [usage("agent-1", 20)],
}


def sessions_archive() -> bytes:
    buffer = io.BytesIO()
    # session_2 stands for a kimi process an agent started itself; it must not be counted.
    wires = {("session_1", agent): events for agent, events in WIRES.items()} | {
        ("session_2", "main"): [usage("x", 999)]
    }
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for (session, agent), events in wires.items():
            data = "\n".join(json.dumps(e) for e in events).encode()
            info = tarfile.TarInfo(f"sessions/wd_workspace_0/{session}/agents/{agent}/wire.jsonl")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def make_agent(tmp_path, monkeypatch, **overrides) -> tuple[KimiCodeSandboxedAgent, SimpleNamespace, dict]:
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "sandbox_server_url", lambda name, **kwargs: f"http://{name}.example:8000")
    client = MagicMock(spec=ServerClient)

    async def post(**kwargs):
        body = {}
        if kwargs["url_path"] == "/verify":
            body = kwargs["json"] | {"reward": float(bool(kwargs["json"]["response"]["output"]))}
        return SimpleNamespace(cookies={}, ok=True, read=AsyncMock(return_value=json.dumps(body).encode()))

    client.post = AsyncMock(side_effect=post)
    config = KimiCodeSandboxedAgentConfig(
        name="kimi",
        host="127.0.0.1",
        port=9000,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "grader"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        sandbox_provider="sandbox",
        artifacts_dir=str(tmp_path / "artifacts"),
        **overrides,
    )
    server = KimiCodeSandboxedAgent(config=config, server_client=client)
    monkeypatch.setattr(server, "_capture_correlation_enabled", lambda: True)
    staged = {}

    async def upload(local, remote):
        staged[remote] = Path(local).read_bytes()

    async def download(remote, local):
        if remote.endswith("stdout.jsonl"):
            local.write_text("\n".join(json.dumps(line) for line in STDOUT))
        elif remote.endswith("sessions.tgz"):
            local.write_bytes(sessions_archive())
        else:
            local.write_text("")

    sandbox = SimpleNamespace(
        _handle=SimpleNamespace(sandbox_id="box", provider_name="opensandbox"),
        upload=AsyncMock(side_effect=upload),
        download=AsyncMock(side_effect=download),
        stop=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None, stdout="2.1.1\n", stderr="")),
    )
    server._start_sandbox = AsyncMock(return_value=sandbox)
    return server, sandbox, staged


def request_body(input_items=None) -> KimiCodeSandboxedAgentRunRequest:
    return KimiCodeSandboxedAgentRunRequest.model_validate(
        {
            "_ng_rollout_id": "rollout-1",
            "responses_create_params": {
                "input": input_items or [{"role": "system", "content": "Be terse."}, {"role": "user", "content": "Go"}]
            },
        }
    )


def staged_config(staged: dict) -> dict:
    return tomllib.loads(next(v for k, v in staged.items() if k.endswith("/home/config.toml")).decode())


async def test_swarm_run_counts_children_and_routes_through_gym(tmp_path, monkeypatch):
    server, sandbox, staged = make_agent(
        tmp_path,
        monkeypatch,
        subagent_model_server={"type": "responses_api_models", "name": "lightning"},
        subagent_model_force=True,
        swarm_max_concurrency=8,
        thinking_effort="high",
    )
    result = await server.run(SimpleNamespace(cookies={}), request_body())

    assert result.reward == 1 and not result.kimi_failed
    assert [item.type for item in result.response.output] == [
        "function_call",
        "function_call_output",
        "reasoning",
        "message",
    ]
    assert result.response.output[-1].content[0].text == "9 and 25"
    # Root and both children: input = inputOther + cache reads; output = 1 per record.
    assert result.response.usage.input_tokens == 100 + 200 + 10 + 20 + 4 * 2
    assert result.response.usage.output_tokens == 4
    assert result.kimi_num_subagents == 2 and result.kimi_max_concurrent_subagents == 2
    assert {c["id"]: c["status"] for c in result.kimi_subagents} == {"agent-0": "completed", "agent-1": "failed"}

    config = staged_config(staged)
    assert config["providers"]["policy"]["base_url"] == "http://policy_model.example:8000/ng-rollout/rollout-1/v1"
    assert config["providers"]["subagent"]["base_url"] == "http://lightning.example:8000/ng-rollout/rollout-1/v1"
    assert config["secondary_model"] == {"default_model": "subagent", "force": True}
    assert config["thinking"] == {"effort": "high"}
    assert "AgentSwarm" not in config["tools"]["disabled"]
    assert next(v for k, v in staged.items() if k.endswith("/home/AGENTS.md")) == b"Be terse."

    run = sandbox.exec.await_args_list[1].kwargs
    assert run["env"]["KIMI_CODE_AGENT_SWARM_MAX_CONCURRENCY"] == "8"
    assert "-p Go --output-format stream-json" in run["command"]
    children = [r for r in result.ng_agent_observations.records if isinstance(r, AgentInvocation)]
    assert {(r.invocation_id, r.parent_invocation_id) for r in children} == {
        ("rollout-1", None),
        ("rollout-1/agent-0", "rollout-1"),
        ("rollout-1/agent-1", "rollout-1"),
    }
    sandbox.stop.assert_awaited_once()


async def test_solo_mode_hides_subagent_tools(tmp_path, monkeypatch):
    server, _, staged = make_agent(tmp_path, monkeypatch, subagents=False, max_steps_per_turn=50)
    await server.run(SimpleNamespace(cookies={}), request_body([{"role": "user", "content": "Go"}]))
    config = staged_config(staged)
    assert {"Agent", "AgentSwarm", "CronCreate"} <= set(config["tools"]["disabled"])
    assert config["loop_control"] == {"max_steps_per_turn": 50}
    assert "secondary_model" not in config
    assert not any(k.endswith("AGENTS.md") for k in staged)


async def test_install_version_mismatch_is_an_infrastructure_error(tmp_path, monkeypatch):
    server, sandbox, _ = make_agent(tmp_path, monkeypatch)
    sandbox.exec.return_value = SimpleNamespace(return_code=0, error_type=None, stdout="2.0.0\n", stderr="")
    with pytest.raises(RuntimeError, match="install failed"):
        await server.run(SimpleNamespace(cookies={}), request_body())
    sandbox.stop.assert_awaited_once()


async def test_failed_execution_scores_zero(tmp_path, monkeypatch):
    server, sandbox, _ = make_agent(tmp_path, monkeypatch, execution_failure_reward_zero=True)
    ok = SimpleNamespace(return_code=0, error_type=None, stdout="2.1.1\n", stderr="")
    sandbox.exec.side_effect = [ok, SimpleNamespace(return_code=1, error_type=None), ok]
    verify = server.server_client.post.side_effect

    async def post(**kwargs):
        if kwargs["url_path"] == "/verify":
            assert kwargs["json"]["response"]["output"] == []
        return await verify(**kwargs)

    server.server_client.post.side_effect = post
    result = await server.run(SimpleNamespace(cookies={}), request_body())
    assert result.kimi_failed and result.kimi_exit_code == 1 and result.reward == 0
    # The failed generation is kept for inspection.
    assert result.response.output[-1].content[0].text == "9 and 25"


async def test_sandbox_stays_up_until_verification_returns(tmp_path, monkeypatch):
    server, sandbox, _ = make_agent(tmp_path, monkeypatch)
    verify = server.server_client.post.side_effect

    async def post(**kwargs):
        if kwargs["url_path"] == "/verify":
            # A resource-owned sandbox is graded in place.
            sandbox.stop.assert_not_awaited()
        return await verify(**kwargs)

    server.server_client.post.side_effect = post
    result = await server.run(SimpleNamespace(cookies={}), request_body())
    assert result.reward == 1
    sandbox.stop.assert_awaited_once()


@pytest.mark.parametrize("error", [RuntimeError("verifier down"), asyncio.CancelledError()])
async def test_verify_failure_or_cancellation_still_stops_the_sandbox(tmp_path, monkeypatch, error):
    server, sandbox, _ = make_agent(tmp_path, monkeypatch)
    verify = server.server_client.post.side_effect

    async def post(**kwargs):
        if kwargs["url_path"] == "/verify":
            raise error
        return await verify(**kwargs)

    server.server_client.post.side_effect = post
    with pytest.raises(type(error)):
        await server.run(SimpleNamespace(cookies={}), request_body())
    sandbox.stop.assert_awaited_once()


def test_to_toml_keeps_non_bmp_unicode_and_escapes_del():
    data = {
        "permission": {"rules": [{"decision": "deny", "pattern": "Bash(echo \U0001f600)"}]},
        "\U0001f600 key": ["\U0001f600", "a\x7fb", "literal \\u0041"],
    }
    assert tomllib.loads(to_toml(data)) == data


def test_to_toml_round_trips_nested_tables_and_rule_arrays():
    data = {
        "default_model": "policy",
        "models": {"a.b": {"capabilities": ["tool_use"], "max_context_size": 8}},
        "permission": {"rules": [{"decision": "deny", "pattern": "Bash(rm -rf*)"}]},
        "loop_control": {"max_steps_per_turn": None},
        "note": 'quote " and é',
    }
    assert tomllib.loads(to_toml(data)) == data | {"loop_control": {}}


def test_max_concurrency_counts_overlapping_children():
    children = [{"started": 0, "ended": 10}, {"started": 5, "ended": 6}, {"started": 6, "ended": 7}, {}]
    assert max_concurrency(children) == 2


def test_to_toml_rejects_null_in_arrays():
    with pytest.raises(ValueError, match="disabled holds None"):
        to_toml({"tools": {"disabled": ["Shell", None]}})


def test_unknown_subagent_events_do_not_end_a_child(tmp_path):
    wire = tmp_path / "agents" / "main" / "wire.jsonl"
    wire.parent.mkdir(parents=True)
    events = [
        {"type": "subagent.spawned", "subagentId": "agent-0", "parentAgentId": "main"},
        {"type": "subagent.started", "subagentId": "agent-0", "time": 1000},
        {"type": "subagent.progress", "subagentId": "agent-0", "time": 1500},
    ]
    wire.write_text("\n".join(json.dumps(e) for e in events))
    _, children = parse_session(tmp_path)
    assert children[0]["status"] == "incomplete" and "ended" not in children[0]


def test_restricted_network_needs_a_local_binary(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="needs kimi_binary_path"):
        make_agent(tmp_path, monkeypatch, network_access="model_only")


async def test_local_binary_is_uploaded_and_installed(tmp_path, monkeypatch):
    binary = tmp_path / "kimi"
    binary.write_bytes(b"\x7fELF")
    server, sandbox, staged = make_agent(
        tmp_path, monkeypatch, kimi_binary_path=str(binary), network_access="model_only"
    )
    await server.run(SimpleNamespace(cookies={}), request_body())
    assert next(v for k, v in staged.items() if k.endswith("/upload/kimi")) == b"\x7fELF"
    install = sandbox.exec.await_args_list[0].kwargs["command"]
    assert "install -m 0755" in install and "curl" not in install

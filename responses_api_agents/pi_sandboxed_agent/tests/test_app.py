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
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.rollout_observability import AgentInvocation, SandboxObservation, ToolCallObservation
from nemo_gym.sandbox.agent_tools import restricted_network_policy
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pi_agent.app import PiAgentRunRequest, PiMCPServerConfig
from responses_api_agents.pi_sandboxed_agent import app
from responses_api_agents.pi_sandboxed_agent.app import _RUN, PiSandboxedAgent, PiSandboxedAgentConfig


def response(value):
    return SimpleNamespace(
        cookies={}, ok=True, read=AsyncMock(return_value=json.dumps(value).encode()), raise_for_status=lambda: None
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "sandbox_server_url", lambda _, **kwargs: "http://model.example:8000")
    client = MagicMock(spec=ServerClient)

    async def post(**kwargs):
        return response(
            kwargs["json"]
            | {
                "reward": float(bool(kwargs["json"]["response"]["output"])),
                "library_reward": float(bool(kwargs["json"]["response"]["output"])),
            }
            if kwargs["url_path"] == "/verify"
            else {}
        )

    client.post = AsyncMock(side_effect=post)
    config = PiSandboxedAgentConfig(
        name="pi",
        host="127.0.0.1",
        port=9000,
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "grader"},
        model_server={"type": "responses_api_models", "name": "model"},
        sandbox_provider="sandbox",
        sandbox_config={"image": "offline-pi"},
        artifacts_dir=str(tmp_path),
        timeout=600,
        bash_timeout=120,
        auto_compaction=False,
        output_token_policy="remaining_context",
        execution_failure_reward_zero=True,
    )
    server = PiSandboxedAgent(config=config, server_client=client)
    monkeypatch.setattr(server, "_capture_correlation_enabled", lambda: True)
    sandbox = SimpleNamespace(
        _handle=SimpleNamespace(sandbox_id="box", provider_name="opensandbox"),
        upload=AsyncMock(),
        stop=AsyncMock(),
        exec=AsyncMock(return_value=SimpleNamespace(return_code=0, error_type=None)),
    )
    events = [
        (
            10.0,
            {
                "type": "tool_execution_start",
                "toolCallId": "call-1",
                "toolName": "bash",
                "args": {"command": "python3 -c 'print(4)'"},
            },
        ),
        (
            11.0,
            {
                "type": "tool_execution_end",
                "toolCallId": "call-1",
                "toolName": "bash",
                "result": {"content": [{"type": "text", "text": "4"}]},
            },
        ),
        (
            12.0,
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "4\u2028done"}],
                    "usage": {"input": 2, "output": 3},
                },
            },
        ),
        (13.0, {"type": "agent_end", "messages": [{"role": "assistant", "stopReason": "stop"}]}),
    ]

    async def download(remote, local):
        if remote.endswith("events.jsonl"):
            local.write_text("\n".join(json.dumps(event) for event in events))
        elif remote.endswith("stdout.jsonl"):
            local.write_text("\n".join(json.dumps(event, ensure_ascii=False) for _, event in events))
        else:
            local.write_text("")

    sandbox.download = AsyncMock(side_effect=download)
    server._start_sandbox = AsyncMock(return_value=sandbox)
    return server, sandbox


def request_body():
    return PiAgentRunRequest.model_validate(
        {"_ng_rollout_id": "pi-rollout", "responses_create_params": {"input": "Compute 2+2"}}
    )


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_native_execution_preserves_settings_timing_and_verification(agent, cleanup_fails, caplog):
    server, sandbox = agent
    if cleanup_fails:
        sandbox.stop.side_effect = RuntimeError("cleanup unavailable")
    staged = {}

    async def upload(local, remote):
        staged[remote] = local.read_text()

    sandbox.upload.side_effect = upload
    result = await server.run(SimpleNamespace(cookies={}), request_body())
    assert result.reward == 1 and not result.pi_failed
    assert result.response.output[0].content[0].text == "4\u2028done"
    assert next(json.loads(v) for p, v in staged.items() if p.endswith("settings.json")) == {
        "compaction": {"enabled": False}
    }
    models = next(json.loads(v) for p, v in staged.items() if p.endswith("models.json"))
    assert models["providers"]["nemo"]["baseUrl"] == "http://model.example:8000/ng-rollout/pi-rollout/v1"
    execution = sandbox.exec.await_args.kwargs
    assert execution["timeout_s"] == 600 and execution["env"]["NEMO_GYM_PI_BASH_TIMEOUT"] == "120"
    assert "remaining-context.mjs" in execution["command"] and "bash-timeout.mjs" in execution["command"]
    tool = next(r for r in result.ng_agent_observations.records if isinstance(r, ToolCallObservation))
    assert tool.started_at == 10 and tool.completed_at == 11 and tool.sandbox_id == "box"
    assert not any(g.code == "no_sandbox_runtime" for g in result.ng_agent_observations.gaps)
    sandbox.stop.assert_awaited_once()
    assert _RUN.get() is None
    if cleanup_fails:
        assert "Failed to stop Pi sandbox" in caplog.text


@pytest.mark.parametrize(
    "failure,tail,recover",
    [
        ("exit", '[14.0, {"type":', True),
        ("timeout", '[14.0, {"type":', True),
        ("exit", '[14.0, {"type":\n', False),
        ("exit", '[14.0, {"type":\n[15.0, {}]\n', False),
        (None, '[14.0, {"type":', False),
    ],
)
async def test_partial_event_tail_only_recovers_failed_execution(agent, failure, tail, recover, caplog):
    server, sandbox = agent
    download = sandbox.download.side_effect

    async def download_with_partial_tail(remote, local):
        await download(remote, local)
        if remote.endswith("events.jsonl"):
            with local.open("a") as stream:
                stream.write("\n" + tail)

    sandbox.download.side_effect = download_with_partial_tail
    if failure:
        sandbox.exec.side_effect = [
            SimpleNamespace(return_code=0, error_type=None),
            TimeoutError() if failure == "timeout" else SimpleNamespace(return_code=137, error_type=None),
        ]
    if recover:
        result = await server.run(SimpleNamespace(cookies={}), request_body())
        assert result.reward == 0 and result.pi_failed
        assert result.pi_exit_code == (137 if failure == "exit" else None)
        assert result.pi_error_type == ("TimeoutError" if failure == "timeout" else None)
        tool = next(r for r in result.ng_agent_observations.records if isinstance(r, ToolCallObservation))
        assert tool.started_at == 10 and tool.completed_at == 11
        assert (Path(result.pi_results_dir) / "events.jsonl").read_text().endswith(tail)
        assert "Ignoring incomplete trailing Pi event" in caplog.text
    else:
        with pytest.raises(json.JSONDecodeError):
            await server.run(SimpleNamespace(cookies={}), request_body())
    assert server.server_client.post.await_count == (2 if recover else 1)
    if recover:
        assert server.server_client.post.await_args.kwargs["json"]["response"]["output"] == []
        assert result.response.output  # Retain the original generation for inspection.
    sandbox.stop.assert_awaited_once()
    assert _RUN.get() is None


@pytest.mark.parametrize("failure", ["exit", "timeout", "timeout_result", "timeout_exit", "export", "cancel", "judge"])
@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_failures_preserve_cleanup_and_zero_reward_boundary(agent, failure, cleanup_fails):
    server, sandbox = agent
    if cleanup_fails:
        sandbox.stop.side_effect = RuntimeError("cleanup unavailable")
    if failure == "exit":
        sandbox.exec.side_effect = [
            SimpleNamespace(return_code=0, error_type=None),
            SimpleNamespace(return_code=137, error_type=None),
        ]
    elif failure == "timeout":
        sandbox.exec.side_effect = [SimpleNamespace(return_code=0, error_type=None), TimeoutError()]
    elif failure in {"timeout_result", "timeout_exit"}:
        sandbox.exec.side_effect = [
            SimpleNamespace(return_code=0, error_type=None),
            SimpleNamespace(return_code=124, error_type="timeout" if failure == "timeout_result" else None),
        ]
    elif failure == "export":
        sandbox.download.side_effect = OSError("export unavailable")
    elif failure == "cancel":
        sandbox.exec.side_effect = asyncio.CancelledError()
    else:
        server.server_client.post.side_effect = [response({}), RuntimeError("judge unavailable")]
    if failure in {"exit", "timeout", "timeout_result", "timeout_exit"}:
        result = await server.run(SimpleNamespace(cookies={}), request_body())
        assert result.reward == 0 and result.pi_failed
        observation = next(r for r in result.ng_agent_observations.records if isinstance(r, SandboxObservation))
        invocation = next(r for r in result.ng_agent_observations.records if isinstance(r, AgentInvocation))
        assert observation.outcome == ("failed" if failure == "exit" else "timeout")
        assert invocation.status == ("failed" if failure == "exit" else "incomplete")
        assert server.server_client.post.await_count == 2
        assert server.server_client.post.await_args.kwargs["json"]["response"]["output"] == []
    else:
        with pytest.raises((OSError, RuntimeError, asyncio.CancelledError)):
            await server.run(SimpleNamespace(cookies={}), request_body())
    sandbox.stop.assert_awaited_once()
    assert _RUN.get() is None


async def test_mcp_discovery_config_is_per_run_and_not_in_receipt(agent, monkeypatch):
    server, sandbox = agent
    monkeypatch.setattr(
        app,
        "seed_mcp_servers",
        AsyncMock(
            return_value={
                "tavily": {
                    "url": "http://tools/mcp",
                    "headers": {"X-NeMo-Gym-Session-Token": "scoped"},
                    "timeout": 600000,
                }
            }
        ),
    )
    staged = {}

    async def upload(local, remote):
        staged[remote] = local.read_text()

    sandbox.upload.side_effect = upload
    result = await server.run(SimpleNamespace(cookies={}), request_body())
    assert any("gym_mcp.mjs" in path for path in staged)
    assert json.loads(next(v for p, v in staged.items() if p.endswith("mcp.json")))["tavily"]["headers"] == {
        "X-NeMo-Gym-Session-Token": "scoped"
    }
    assert "scoped" not in (Path(result.pi_results_dir) / "generation.json").read_text()
    assert server.config.mcp_servers == {}


@pytest.mark.parametrize("host", ["localhost", "127.0.0.2", "0.0.0.0", "[::1]", "[::]"])
def test_network_policy_rejects_unreachable_hosts(host):
    with pytest.raises(ValueError, match="reachable"):
        restricted_network_policy("opensandbox", ["http://" + host + ":8000"])


def test_network_policy_fails_closed_for_other_providers():
    with pytest.raises(ValueError, match="OpenSandbox"):
        restricted_network_policy("docker", ["http://model.example"])
    assert restricted_network_policy("opensandbox", ["http://model.example", "http://tools.example"]) == {
        "defaultAction": "deny",
        "egress": [{"action": "allow", "target": "model.example"}, {"action": "allow", "target": "tools.example"}],
    }


def test_capture_preserves_multiline_unicode_and_process_exit(tmp_path):
    capture = Path(app.__file__).with_name("capture.py")
    output = tmp_path / "events.jsonl"
    payload = {"type": "message_end", "text": "line\u2028paragraph\u2029"}
    code = f"import sys; print({json.dumps(payload, ensure_ascii=False)!r});sys.exit(7)"
    result = subprocess.run(
        [sys.executable, str(capture), str(output), sys.executable, "-c", code], capture_output=True
    )
    assert result.returncode == 7
    assert json.loads(result.stdout) == payload
    observed, event = json.loads(output.read_text())
    assert observed > 0 and event == payload


async def test_mcp_initialization_failure_is_a_request_failure(agent):
    server, sandbox = agent
    server.config.mcp_servers = {"search": PiMCPServerConfig(url="http://tools/mcp")}
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        SimpleNamespace(return_code=78, error_type=None),
    ]
    with pytest.raises(RuntimeError, match="MCP tools could not be initialized"):
        await server.run(SimpleNamespace(cookies={}), request_body())
    assert server.server_client.post.await_count == 1
    sandbox.stop.assert_awaited_once()
    assert _RUN.get() is None


@pytest.mark.parametrize("stop_reason", ["length", "stop", "aborted", "error"])
@pytest.mark.parametrize("collect_observations", [False, True])
@pytest.mark.parametrize("force_zero", [False, True])
async def test_terminal_stop_controls_scoring_without_observations(
    agent, stop_reason, collect_observations, force_zero
):
    server, sandbox = agent
    server.config.execution_failure_reward_zero = force_zero
    server._capture_correlation_enabled = lambda: collect_observations
    download = sandbox.download.side_effect

    async def terminal_event(remote, local):
        await download(remote, local)
        if remote.endswith("events.jsonl"):
            # An earlier length stop followed by a clean final answer is recoverable.
            events = [json.loads(line) for line in local.read_text().splitlines()]
            events[-1][1]["messages"] = [
                {"role": "assistant", "stopReason": "length"},
                {"role": "assistant", "stopReason": stop_reason},
            ]
            local.write_text("\n".join(json.dumps(event) for event in events))

    sandbox.download.side_effect = terminal_event
    result = await server.run(SimpleNamespace(cookies={}), request_body())
    failed = stop_reason != "stop"
    assert result.pi_failed is failed
    assert result.finished_naturally is (not failed)
    assert result.reward == (0 if force_zero and failed else 1)
    assert result.model_dump()["library_reward"] == result.reward
    assert result.response.output  # Preserve the answer even when the verifier receives empty output.
    assert bool(server.server_client.post.await_args.kwargs["json"]["response"]["output"]) is not (
        force_zero and failed
    )
    assert (result.response.status == "incomplete") is (stop_reason == "length")
    if stop_reason == "length":
        assert result.response.incomplete_details.reason == "max_output_tokens"
    if collect_observations:
        invocation = next(r for r in result.ng_agent_observations.records if isinstance(r, AgentInvocation))
        assert (
            invocation.status
            == {"length": "incomplete", "aborted": "incomplete", "error": "failed", "stop": "completed"}[stop_reason]
        )
    receipt = json.loads((Path(result.pi_results_dir) / "generation.json").read_text())
    assert receipt["pi_failed"] is failed and receipt["response"]["output"]
    assert receipt["response"]["status"] == result.response.status

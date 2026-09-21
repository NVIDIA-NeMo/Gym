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

from nemo_gym.rollout_observability import ToolCallObservation
from nemo_gym.sandbox.agent_tools import restricted_network_policy
from nemo_gym.server_utils import ServerClient
from responses_api_agents.pi_agent.app import PiAgentRunRequest
from responses_api_agents.pi_sandboxed_agent import app
from responses_api_agents.pi_sandboxed_agent.app import _RUN, PiSandboxedAgent, PiSandboxedAgentConfig


def response(value):
    return SimpleNamespace(
        cookies={}, read=AsyncMock(return_value=json.dumps(value).encode()), raise_for_status=lambda: None
    )


@pytest.fixture
def agent(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "sandbox_server_url", lambda _: "http://model.example:8000")
    client = MagicMock(spec=ServerClient)

    async def post(**kwargs):
        return response(kwargs["json"] | {"reward": 1} if kwargs["url_path"] == "/verify" else {})

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


async def test_native_execution_preserves_settings_timing_and_verification(agent):
    server, sandbox = agent
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


@pytest.mark.parametrize("failure", ["exit", "timeout", "export", "cancel", "judge"])
async def test_failures_preserve_cleanup_and_zero_reward_boundary(agent, failure):
    server, sandbox = agent
    if failure == "exit":
        sandbox.exec.side_effect = [
            SimpleNamespace(return_code=0, error_type=None),
            SimpleNamespace(return_code=137, error_type=None),
        ]
    elif failure == "timeout":
        sandbox.exec.side_effect = [SimpleNamespace(return_code=0, error_type=None), TimeoutError()]
    elif failure == "export":
        sandbox.download.side_effect = OSError("export unavailable")
    elif failure == "cancel":
        sandbox.exec.side_effect = asyncio.CancelledError()
    else:
        server.server_client.post.side_effect = [response({}), RuntimeError("judge unavailable")]
    if failure in {"exit", "timeout"}:
        result = await server.run(SimpleNamespace(cookies={}), request_body())
        assert result.reward == 0 and result.pi_failed
        assert server.server_client.post.await_count == 1
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

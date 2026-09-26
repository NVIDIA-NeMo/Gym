# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent server: staging, the single exec, record collection, and termination mapping."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from nemo_gym.server_utils import ServerClient
from responses_api_agents.miniswe_in_sandbox_agent import app as app_module
from responses_api_agents.miniswe_in_sandbox_agent.app import (
    MiniSWEInSandboxAgent,
    MiniSWEInSandboxConfig,
    Termination,
    classify,
)
from responses_api_agents.miniswe_in_sandbox_agent.models import SeedSessionResponse


USAGE = {
    "input_tokens": 10,
    "output_tokens": 5,
    "total_tokens": 15,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 2},
}
ITEMS = [
    {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "hi", "annotations": []}],
    },
    {
        "id": "c1",
        "type": "function_call",
        "call_id": "c1",
        "name": "bash",
        "arguments": '{"command": "ls"}',
        "status": "completed",
    },
    {"type": "function_call_output", "call_id": "c1", "output": '{"returncode": 0, "output": ""}'},
]


def make_config(tmp_path, **overrides):
    values = dict(
        host="localhost",
        port=1,
        name="agent",
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "tb4"},
        model_server={"type": "responses_api_models", "name": "policy_model"},
        model_gateway_url="http://10.109.22.242:24402",
        artifacts_dir=tmp_path / "results",
        agent_max_timeout_sec=600,
        instruction_suffix="\n\n## Additional instructions\n\nBe careful.\n",
    )
    values.update(overrides)
    return MiniSWEInSandboxConfig(**values)


def seed(**overrides):
    values = dict(
        session_id="tb4-abc",
        task_id="terminal-bench/test",
        sandbox_descriptor={"sandbox_id": "box"},
        sandbox_provider={"opensandbox": {"connection": {"domain": "example.invalid"}}},
        instruction="Do the task.\n",
        user="cam",
        agent_timeout_sec=3600,
    )
    values.update(overrides)
    return SeedSessionResponse(**values)


class FakeSandbox:
    """Records exec/upload calls; download materializes the runner's records into the agent's directory."""

    def __init__(self, *, default_uid="0", exec_result=None, records=None, preflight_rc=0):
        self.calls = []
        self.uploads = []
        self.default_uid = default_uid
        self.exec_result = exec_result or SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
        self.records = records if records is not None else {}
        self.preflight_rc = preflight_rc

    async def exec(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command == "pwd":
            return SimpleNamespace(return_code=0, stdout="/home/cam/job\n", stderr="", error_type=None)
        if command == "id -u":
            return SimpleNamespace(return_code=0, stdout=self.default_uid + "\n", stderr="", error_type=None)
        if command.startswith("command -v setsid"):
            return SimpleNamespace(
                return_code=self.preflight_rc, stdout="3.12.3\n", stderr="no python", error_type=None
            )
        if command.startswith("setsid --wait"):
            return self.exec_result
        return SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)

    async def upload(self, local, remote):
        self.uploads.append(
            (Path(local).name, remote, Path(local).read_bytes() if Path(local).suffix != ".zip" else b"")
        )

    async def download(self, remote, local):
        name = remote.rsplit("/", 1)[-1]
        if name not in self.records:
            raise FileNotFoundError(remote)
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        value = self.records[name]
        Path(local).write_text(value if isinstance(value, str) else json.dumps(value))


def records(exit_status="Submitted", items=ITEMS, usages=(USAGE, USAGE), finished=True):
    return {
        "trajectory.json": {
            "trajectory_format": "mini-swe-agent-1.1",
            "info": {"exit_status": exit_status},
            "messages": [],
        },
        "output_items.json": items,
        "usages.json": list(usages),
        "result.json": {
            "exit_status": exit_status,
            "n_calls": 2,
            "steps": 1,
            "uid": 1000,
            "cwd": "/home/cam/job",
            "finished": finished,
        },
        "runner.log": "runner log\n",
    }


def make_agent(tmp_path, monkeypatch, sandbox, **config_overrides):
    agent = MiniSWEInSandboxAgent(
        config=make_config(tmp_path, **config_overrides), server_client=MagicMock(spec=ServerClient)
    )
    provider = SimpleNamespace(aclose=AsyncMock())
    monkeypatch.setattr(app_module, "create_provider", lambda cfg: provider)
    monkeypatch.setattr(app_module, "resolve_provider_config", lambda cfg: cfg)
    monkeypatch.setattr(app_module.AsyncSandbox, "connect", AsyncMock(return_value=sandbox))
    agent._token_id_capture_enabled = lambda: False
    return agent, provider


async def test_execute_stages_runs_as_the_task_user_and_builds_the_response(tmp_path, monkeypatch):
    sandbox = FakeSandbox(records=records())
    agent, provider = make_agent(tmp_path, monkeypatch, sandbox)
    params = app_module.NeMoGymResponseCreateParamsNonStreaming(input=[], max_output_tokens=65536)
    result = await agent.execute(seed(), params, rollout_id="r1", capture_model_calls=False)
    assert result.agent_started and result.termination.reason == "completed"
    # The verify body carries the prompt the runner used (server-harness parity), not the row's empty input.
    assert result.responses_create_params.input[0].role == "user"
    assert result.responses_create_params.input[0].content.startswith("Do the task.")
    assert result.harness_metadata["staging"]["bootstrap_uid"] == "0"
    assert result.harness_metadata["staging"]["staged_owner"] == "cam"
    assert result.harness_metadata["consistency"]["outputs_match_calls"] is True
    # Staging: readable dir, three uploads, chown to the task user because the default identity is root.
    commands = [c for c, _ in sandbox.calls]
    assert any(c.startswith("mkdir -p /tmp/ng-miniswe-tb4-abc") for c in commands)
    assert any(c == "chown -R cam /tmp/ng-miniswe-tb4-abc" for c in commands)
    assert [u[0] for u in sandbox.uploads] == ["miniswe_runner.py", "vendor.zip", "config.json"]
    config = json.loads(sandbox.uploads[2][2])
    assert config["model_url"] == "http://10.109.22.242:24402/v1" and config["headers"] == {"x-session-id": "tb4-abc"}
    assert config["task"] == "Do the task.\n\n\n## Additional instructions\n\nBe careful.\n"
    assert config["workdir"] == "/home/cam/job" and config["pids_file"] == "/tmp/tb4-abc.pids"
    # Responses defaults ride along, exactly as the server harness's params.model_dump(exclude_none=True).
    assert config["request_params"] == {"max_output_tokens": 65536, "parallel_tool_calls": True, "tool_choice": "auto"}
    assert config["budget_sec"] == 600 - 60
    assert config["templates"]["instance_template"].startswith("Please solve this issue: {{task}}")
    assert config["env"]["PAGER"] == "cat" and config["step_timeout_sec"] == 30
    # The single exec: as the task user, in the workdir, with the budget, registering its pgid for quiesce.
    run_call = next((c, k) for c, k in sandbox.calls if c.startswith("setsid --wait"))
    assert run_call[1] == {"user": "cam", "cwd": "/home/cam/job", "timeout_s": 600}
    assert "echo $$ >> /tmp/tb4-abc.pids" in run_call[0] and "miniswe_runner.py --config" in run_call[0]
    # Response and usage from the runner's records.
    assert [item.type for item in result.response.output] == ["message", "function_call", "function_call_output"]
    assert (
        result.response.usage.input_tokens == 20 and result.response.usage.output_tokens_details.reasoning_tokens == 4
    )
    assert result.harness_metadata["exit_status"] == "Submitted" and result.harness_metadata["n_calls"] == 2
    assert result.harness_metadata["harness_version"].startswith("miniswe-in-sandbox-runner")
    assert result.termination.artifacts == [str(tmp_path / "results/tb4-abc/trajectory.json")]
    assert set(result.agent_timings) == {"agent_setup", "agent_execution"}
    provider.aclose.assert_awaited_once()


async def test_capture_prefix_and_no_chown_for_root_agent(tmp_path, monkeypatch):
    sandbox = FakeSandbox(records=records())
    agent, _ = make_agent(tmp_path, monkeypatch, sandbox)
    agent._token_id_capture_enabled = lambda: True
    params = app_module.NeMoGymResponseCreateParamsNonStreaming(input=[])
    await agent.execute(seed(user=None), params, rollout_id="roll.1", capture_model_calls=True)
    config = json.loads(sandbox.uploads[2][2])
    assert config["model_url"] == "http://10.109.22.242:24402/ng-rollout/roll.1/training-token-capture/v1"
    assert not any(c.startswith("chown") for c, _ in sandbox.calls)
    run_call = next(k for c, k in sandbox.calls if c.startswith("setsid --wait"))
    assert run_call["user"] is None


OK = SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
KILLED = SimpleNamespace(return_code=137, stdout="", stderr="", error_type=None)


@pytest.mark.parametrize(
    "exec_result,exit_status,finished,reason,detail_part,stopped",
    [
        (OK, "LimitsExceeded", True, "nonzero_exit", "LimitsExceeded", False),
        (OK, "ContextWindowExceeded", True, "nonzero_exit", "ContextWindowExceeded", False),
        (OK, "TimeExceeded", True, "timeout", "TimeExceeded", False),
        (
            SimpleNamespace(return_code=1, stdout="", stderr="", error_type=None),
            "ModelServerError",
            True,
            "infrastructure_error",
            "ModelServerError",
            False,
        ),
        # Budget kill: OpenSandbox gives no timeout marker, only a non-terminal record and a long elapsed time.
        (KILLED, "", False, "timeout", "budget", True),
    ],
)
async def test_termination_mapping(
    tmp_path, monkeypatch, exec_result, exit_status, finished, reason, detail_part, stopped
):
    sandbox = FakeSandbox(exec_result=exec_result, records=records(exit_status=exit_status, finished=finished))
    agent, _ = make_agent(tmp_path, monkeypatch, sandbox, agent_max_timeout_sec=0.01)
    params = app_module.NeMoGymResponseCreateParamsNonStreaming(input=[])
    result = await agent.execute(seed(), params, rollout_id="r", capture_model_calls=False)
    assert result.agent_started and result.termination.reason == reason
    assert detail_part in (result.termination.detail or "")
    # Whatever the termination, the partial trajectory still yields the response items collected so far.
    assert len(result.response.output) == 3
    stops = [(c, k) for c, k in sandbox.calls if "kill -TERM" in c]
    assert bool(stops) is stopped and all(k["user"] == "cam" for c, k in stops)


async def test_runner_killed_inside_the_sandbox_is_graded_not_masked(tmp_path, monkeypatch):
    sandbox = FakeSandbox(exec_result=KILLED, records=records(exit_status="", finished=False))
    agent, _ = make_agent(tmp_path, monkeypatch, sandbox, agent_max_timeout_sec=3600)
    params = app_module.NeMoGymResponseCreateParamsNonStreaming(input=[])
    result = await agent.execute(seed(), params, rollout_id="r", capture_model_calls=False)
    assert result.termination.reason == "nonzero_exit" and result.termination.detail.startswith("RunnerKilled")


async def test_provider_overrides_touch_only_this_agents_transport(tmp_path, monkeypatch):
    sandbox = FakeSandbox(records=records())
    agent, _ = make_agent(tmp_path, monkeypatch, sandbox)
    seen = {}
    monkeypatch.setattr(
        app_module, "create_provider", lambda cfg: seen.setdefault("cfg", cfg) and SimpleNamespace(aclose=AsyncMock())
    )
    params = app_module.NeMoGymResponseCreateParamsNonStreaming(input=[])
    seeded = seed(sandbox_provider={"opensandbox": {"connection": {"domain": "x"}, "operations": {"retries": 3}}})
    await agent.execute(seeded, params, rollout_id="r", capture_model_calls=False)
    assert seen["cfg"]["opensandbox"]["operations"] == {
        "retries": 3,
        "background_poll_initial_s": 5.0,
        "background_poll_interval_s": 30.0,
        "status_poll_timeout_s": 30.0,
    }
    assert seeded.sandbox_provider["opensandbox"]["operations"] == {"retries": 3}  # the seed is not mutated


async def test_exec_wall_timeout_is_a_timeout_with_partial_records(tmp_path, monkeypatch):
    sandbox = FakeSandbox(records=records(exit_status="", finished=False))

    async def hang(command, **kwargs):
        if command.startswith("setsid --wait"):
            await asyncio.sleep(10)
        return await FakeSandbox.exec(sandbox, command, **kwargs)

    sandbox.exec = hang
    agent, _ = make_agent(tmp_path, monkeypatch, sandbox, agent_max_timeout_sec=1)
    monkeypatch.setattr(app_module, "runner_config", lambda **kw: {"stub": True})
    params = app_module.NeMoGymResponseCreateParamsNonStreaming(input=[])
    # budget 1 s + 120 s guard would be slow: shrink the guard through the budget passed to the timeout.
    agent.config.runner_exit_margin_sec = 0
    original_timeout = app_module.asyncio.timeout

    def short_timeout(seconds):
        return original_timeout(min(seconds, 1.5))

    monkeypatch.setattr(app_module.asyncio, "timeout", short_timeout)
    result = await agent.execute(seed(), params, rollout_id="r", capture_model_calls=False)
    assert result.agent_started and result.termination.reason == "timeout"
    assert len(result.response.output) == 3


async def test_preflight_and_mcp_failures_are_unstarted_infrastructure_errors(tmp_path, monkeypatch):
    sandbox = FakeSandbox(preflight_rc=1)
    agent, _ = make_agent(tmp_path, monkeypatch, sandbox)
    params = app_module.NeMoGymResponseCreateParamsNonStreaming(input=[])
    result = await agent.execute(seed(), params, rollout_id="r", capture_model_calls=False)
    assert not result.agent_started and result.termination.reason == "infrastructure_error"
    assert "Python >= 3.9" in result.termination.detail and not sandbox.uploads
    sandbox = FakeSandbox()
    agent, _ = make_agent(tmp_path, monkeypatch, sandbox)
    result = await agent.execute(seed(mcp_servers=[{"name": "x"}]), params, rollout_id="r", capture_model_calls=False)
    assert not result.agent_started and "MCP" in result.termination.detail


def test_classify_and_config_validation(tmp_path):
    ok = SimpleNamespace(return_code=0, stdout="", stderr="", error_type=None)
    assert classify(None, None, "").reason == "timeout"
    assert classify(ok, None, "", elapsed=10, budget=3600).reason == "infrastructure_error"
    assert classify(ok, {"exit_status": "Submitted", "finished": True}, "").reason == "completed"
    assert classify(ok, {"exit_status": "TimeExceeded", "finished": True}, "").reason == "timeout"
    assert classify(ok, {"exit_status": "RepeatedFormatError", "finished": True}, "") == Termination(
        reason="nonzero_exit", exit_code=0, detail="RepeatedFormatError"
    )
    assert classify(ok, {"exit_status": "KeyError", "finished": True}, "boom").reason == "infrastructure_error"
    killed = SimpleNamespace(return_code=137, stdout="", stderr="", error_type=None)
    assert classify(killed, {"exit_status": "", "finished": False}, "", elapsed=3600, budget=3600).reason == "timeout"
    assert (
        classify(killed, {"exit_status": "", "finished": False}, "", elapsed=100, budget=3600).reason == "nonzero_exit"
    )
    with pytest.raises(ValidationError, match="origin"):
        make_config(tmp_path, model_gateway_url="http://gw:1/v1")
    with pytest.raises(ValidationError):
        make_config(tmp_path, model_gateway_url="gw:1")


def test_seed_termination_short_circuits(tmp_path, monkeypatch):
    sandbox = FakeSandbox()
    agent, _ = make_agent(tmp_path, monkeypatch, sandbox)
    params = app_module.NeMoGymResponseCreateParamsNonStreaming(input=[])
    failed = seed(termination=Termination(reason="infrastructure_error", detail="pull failed"))
    result = asyncio.run(agent.execute(failed, params, rollout_id="r", capture_model_calls=False))
    assert not result.agent_started and result.termination.detail == "pull failed" and not sandbox.calls

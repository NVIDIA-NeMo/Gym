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
import logging
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.agents import restricted_network_policy
from nemo_gym.client import ServerClient
from nemo_gym.trajectory import AgentInvocation, SandboxObservation, ToolCallObservation
from nemo_gym.utils import IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME
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


@pytest.mark.parametrize("worker", [False, True])
def test_only_the_main_process_logs_the_deprecation_once(agent, caplog, monkeypatch, worker):
    """Workers of a multi-worker server rebuild it after the main process has logged the warning."""
    server, _ = agent
    app._warn_deprecated_once.cache_clear()
    monkeypatch.setenv(IS_NEMO_GYM_FASTAPI_WORKER_KEY_NAME, "1" if worker else "")
    with caplog.at_level(logging.WARNING, logger=app.__name__):
        for _ in range(2):
            PiSandboxedAgent(config=server.config, server_client=MagicMock(spec=ServerClient))

    warnings = [r.getMessage() for r in caplog.records if r.name == app.__name__]
    assert warnings == ([] if worker else [app.DEPRECATION_WARNING])
    for phrase in ("pi_sandboxed_agent", "legacy /run", "Use pi_agent", "tool_servers", "pi_agent/README.md"):
        assert phrase in app.DEPRECATION_WARNING


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


@pytest.mark.parametrize("budget,expected", [(None, 600), ("12.5", 12.5), ("3600", 600)])
async def test_task_timeout_bounds_pi_execution_but_not_setup(agent, budget, expected):
    server, sandbox = agent
    body = request_body()
    body.responses_create_params.metadata = None if budget is None else {"agent_timeout_sec": budget}
    result = await server.run(SimpleNamespace(cookies={}), body)
    setup, *_, execution = sandbox.exec.await_args_list
    assert setup.kwargs["timeout_s"] == 600
    assert execution.kwargs["timeout_s"] == expected
    assert result.response.metadata["agent_timeout_sec"] == str(float(expected))


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
            SimpleNamespace(return_code=124, error_type="timeout")
            if failure == "timeout"
            else SimpleNamespace(return_code=137, error_type=None),
        ]
    if recover:
        result = await server.run(SimpleNamespace(cookies={}), request_body())
        assert result.reward == 0 and result.pi_failed
        assert result.pi_exit_code == (137 if failure == "exit" else 124)
        assert result.pi_error_type == ("timeout" if failure == "timeout" else None)
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


@pytest.mark.parametrize("failure", ["exit", "timeout_result", "timeout_exit", "export", "cancel", "judge"])
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
    if failure in {"exit", "timeout_result", "timeout_exit"}:
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


def _provider_event(kind="provider_unavailable", **kwargs):
    from nemo_gym.sandbox import sandbox_failure

    return sandbox_failure(operation="exec", failure_kind=kind, sandbox_id="box", provider="opensandbox", **kwargs)


def _sandbox_event_of(raised):
    """The agent route failure's provider cause, checked through the serialized report as a reader sees it."""
    from nemo_gym.core import AGENT_ROUTE_FAILED, FailureHeader, NgFailureEvent, failure_report_for

    assert raised.failure_kind == AGENT_ROUTE_FAILED and (raised.boundary, raised.stage) == ("agent", "agent")
    cause = raised.__cause__
    assert isinstance(cause, NgFailureEvent) and cause.boundary == "sandbox"
    report = failure_report_for(raised)
    assert report.failure_id == raised.event_id
    assert report.events[raised.event_id].cause_id == cause.event_id
    assert set(report.events) == {raised.event_id, cause.event_id}
    wire = FailureHeader.encode(report)
    assert FailureHeader.NAME in wire
    return cause, report.events[cause.event_id]


@pytest.mark.parametrize(
    "outcome, kind, reason",
    [
        (RuntimeError("provider unavailable"), "provider_request_failed", None),
        (TimeoutError("command status request timed out"), "provider_io_timeout", None),
        (SimpleNamespace(return_code=-1, error_type=None), "provider_command_outcome_unknown", None),
        (
            SimpleNamespace(return_code=1, error_type="sandbox_error"),
            "provider_command_outcome_unknown",
            "sandbox_error",
        ),
    ],
)
async def test_untyped_provider_failure_is_a_typed_provider_event_and_never_verified(agent, outcome, kind, reason):
    from nemo_gym.core import NgFailureEvent

    server, sandbox = agent
    sandbox.exec.side_effect = [SimpleNamespace(return_code=0, error_type=None), outcome]
    with pytest.raises(NgFailureEvent) as raised:
        await server.run(SimpleNamespace(cookies={}), request_body())
    cause, event = _sandbox_event_of(raised.value)
    assert cause.failure_kind == event.failure_kind == kind
    # Provider identity and the provider's own reason are typed details, not part of either failure_reason.
    assert (event.details.provider, event.details.sandbox_id) == ("opensandbox", "box")
    assert event.details.provider_status_reason == reason
    # A -1 sentinel or an error type is no observed process exit.
    assert event.details.exit_code is None
    assert "opensandbox" not in raised.value.failure_reason and "opensandbox" not in event.failure_reason
    assert raised.value.failure_reason == "Agent operation agent.sandbox_exec failed: NgFailureEvent"
    assert server.server_client.post.await_count == 1  # seed only: the verifier is not called
    sandbox.stop.assert_awaited_once()
    assert _RUN.get() is None


async def test_a_raised_provider_timeout_keeps_its_exception_as_the_cause_chain(agent):
    from nemo_gym.core import NgFailureEvent

    server, sandbox = agent
    timeout = TimeoutError("command status request timed out")
    sandbox.exec.side_effect = [SimpleNamespace(return_code=0, error_type=None), timeout]
    with pytest.raises(NgFailureEvent) as raised:
        await server.run(SimpleNamespace(cookies={}), request_body())
    assert raised.value.__cause__.__cause__ is timeout


async def test_a_typed_provider_failure_is_chained_once_and_not_renormalized(agent):
    from nemo_gym.core import NgFailureEvent

    server, sandbox = agent
    typed = _provider_event(http_status=503)
    sandbox.exec.side_effect = [SimpleNamespace(return_code=0, error_type=None), typed]
    with pytest.raises(NgFailureEvent) as raised:
        await server.run(SimpleNamespace(cookies={}), request_body())
    cause, event = _sandbox_event_of(raised.value)
    assert cause is typed and event.failure_kind == "provider_unavailable" and event.http_status == 503
    assert server.server_client.post.await_count == 1
    sandbox.stop.assert_awaited_once()


async def test_a_returned_command_deadline_is_still_graded(agent):
    server, sandbox = agent
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        SimpleNamespace(return_code=124, error_type="timeout"),
    ]
    result = await server.run(SimpleNamespace(cookies={}), request_body())
    assert result.pi_failed and result.pi_error_type == "timeout" and not result.mask_sample
    assert server.server_client.post.await_count == 2


async def test_a_configured_rule_counts_a_provider_failure_as_a_zero_and_the_result_says_so(agent):
    from nemo_gym.core import RequestFailureLog, StageOutcomePolicyConfig
    from nemo_gym.core.outcome_policy import is_policy_counted_zero

    server, sandbox = agent
    server.config = server.config.model_copy(
        update={
            "outcome_policy": StageOutcomePolicyConfig(
                by_stage={"agent": {"by_failure_kind": {"provider_io_timeout": "count_as_zero"}}}
            )
        }
    )
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        TimeoutError("status request timed out"),
    ]
    with RequestFailureLog.capture() as log:
        result = await server.run(SimpleNamespace(cookies={}), request_body())
    row = result.model_dump()
    assert (row["reward"], row["mask_sample"], row["failure_kind"]) == (0.0, False, "provider_io_timeout")
    assert row["failure_outcome_action"] == "count_as_zero" and row["failure_outcome_attribution"] == "evaluation"
    assert is_policy_counted_zero(row)
    # The counted failure is recorded with its provider cause, and the verifier saw an empty answer.
    kinds = {event.failure_kind for event in log.report.events.values()}
    assert kinds == {"agent_route_failed", "provider_io_timeout"}
    # The selected zero is applied locally: the verifier is not called again on an empty answer.
    assert server.server_client.post.await_count == 1
    assert row["pi_failed"] and row["pi_exit_code"] is None
    # A provider timeout is not Pi's own deadline.
    observation = next(r for r in result.ng_agent_observations.records if isinstance(r, SandboxObservation))
    assert observation.outcome == "failed" and observation.error_type is None
    sandbox.stop.assert_awaited_once()


async def test_the_default_policy_counts_an_agent_stage_oom_as_a_zero(agent):
    server, sandbox = agent
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        _provider_event("provider_oom_killed"),
    ]
    result = await server.run(SimpleNamespace(cookies={}), request_body())
    assert (result.reward, result.mask_sample, result.failure_kind) == (0.0, False, "provider_oom_killed")
    assert result.failure_outcome_action == "count_as_zero"
    assert result.failure_outcome_attribution is not None


@pytest.mark.parametrize(
    "verifier_error",
    [None, RuntimeError("verifier unavailable")],
    ids=["masked_verifier", "verifier_exception"],
)
async def test_a_selected_zero_does_not_depend_on_a_second_grading_call(agent, verifier_error):
    from nemo_gym.core.outcome_policy import is_policy_counted_zero

    server, sandbox = agent

    async def post(**kwargs):
        if kwargs["url_path"] != "/verify":
            return response({})
        if verifier_error is not None:
            raise verifier_error
        return response(kwargs["json"] | {"reward": 0.0, "mask_sample": True, "failure_kind": "verifier_error"})

    server.server_client.post = AsyncMock(side_effect=post)
    sandbox.exec.side_effect = [
        SimpleNamespace(return_code=0, error_type=None),
        _provider_event("provider_oom_killed"),
    ]
    sandbox.stop.side_effect = RuntimeError("cleanup unavailable")
    result = await server.run(SimpleNamespace(cookies={}), request_body())
    row = result.model_dump()
    assert is_policy_counted_zero(row)
    assert (row["failure_kind"], row["failure_outcome_action"]) == ("provider_oom_killed", "count_as_zero")
    assert row["failure_outcome_attribution"] is not None
    assert row["_ng_failure_events"] and row["_ng_failure_id"]
    assert not any(c.kwargs["url_path"] == "/verify" for c in server.server_client.post.await_args_list)


async def test_an_unexpected_exec_exception_text_never_reaches_the_failure_report(agent):
    from nemo_gym.core import NgFailureEvent, failure_report_for

    marker = "SYNTHETIC_PRIVATE_BODY_MARKER"
    server, sandbox = agent
    sandbox.exec.side_effect = [SimpleNamespace(return_code=0, error_type=None), RuntimeError(marker)]
    with pytest.raises(NgFailureEvent) as raised:
        await server.run(SimpleNamespace(cookies={}), request_body())
    cause, event = _sandbox_event_of(raised.value)
    assert event.foreign_error.exception_type == "RuntimeError" and event.foreign_error.message == ""
    assert isinstance(cause.__cause__, RuntimeError)
    report = failure_report_for(raised.value)
    assert marker not in json.dumps(report.model_dump(mode="json")) + json.dumps(cause.to_record())
    assert marker not in json.dumps(raised.value.to_record())


def test_the_agent_stage_policy_supports_only_fail_and_count_as_zero():
    from pydantic import ValidationError

    from nemo_gym.core import StageOutcomePolicyConfig

    with pytest.raises(ValidationError, match="fail and count_as_zero"):
        PiSandboxedAgentConfig(
            name="pi",
            host="127.0.0.1",
            port=9000,
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "grader"},
            model_server={"type": "responses_api_models", "name": "model"},
            sandbox_provider="sandbox",
            sandbox_config={},
            outcome_policy=StageOutcomePolicyConfig(by_stage={"agent": {"default_action": "retry"}}),
        )

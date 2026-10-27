# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCP grants, required-server readiness, and sandbox ownership contract tests."""

import copy
import json
import re
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException

from nemo_gym.base_responses_api_agent import AgentSessionSetupError
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.tool_access import DirectHTTPToolAccess, MCPStreamableHTTPConnection, MCPToolAccess
from responses_api_agents.kilocode_agent.sandbox import KiloArtifacts, KiloSandboxSession
from responses_api_agents.kilocode_agent.tests.test_app import _make_model_server_agent
from responses_api_agents.kilocode_agent.tests.test_sandbox import APP, TEXT, TRANSPORT, receipt, seed, state


def mcp(name: str = "task.tools", *, required: bool = True, token: str = "Bearer task-token") -> MCPToolAccess:
    return MCPToolAccess(
        name=name,
        required=required,
        connection=MCPStreamableHTTPConnection(
            url="http://resources:9001/mcp", headers={"Authorization": token, "X-Task": "task-1"}
        ),
    )


def test_mcp_config_preserves_static_servers_and_uses_session_credentials() -> None:
    static = {"catalog": {"type": "remote", "url": "http://catalog/mcp", "enabled": False}}
    agent = _make_model_server_agent(kilo_config={"mcp": static}, tool_accesses=[mcp(token="Bearer old-token")])
    session = state()
    session.request.tool_accesses = [mcp(token="Bearer episode-token"), mcp("optional", required=False)]
    original = copy.deepcopy(agent.config.kilo_config)

    config = agent._sandbox_config(session, rollout_id="rollout-1-a2")

    assert config["mcp"]["catalog"] == static["catalog"]
    remote = config["mcp"]["task.tools"]
    assert remote["type"] == "remote"
    assert remote["url"] == "http://resources:9001/mcp"
    assert remote["headers"] == {"Authorization": "Bearer episode-token", "X-Task": "task-1"}
    assert remote["oauth"] is False and remote["enabled"] is True
    # Kilo's MCP timeout applies to tool execution too; setup must not shorten that budget.
    assert "timeout" not in remote
    assert config["mcp"]["optional"]["enabled"] is True
    assert config["model"] == f"nemo/{agent.config.model}"
    assert "/ng-rollout/rollout-1-a2/v1" in config["provider"]["nemo"]["options"]["baseURL"]
    assert agent.config.kilo_config == original
    assert agent.config.tool_accesses[0].connection.headers["Authorization"] == "Bearer old-token"


@pytest.mark.parametrize("static_name,grant_name", [("tools", "tools"), ("task.tools", "task_tools")])
async def test_static_mcp_namespace_collisions_fail_before_allocating_sandbox(
    static_name: str, grant_name: str
) -> None:
    agent = _make_model_server_agent(
        kilo_version="7.4.15", kilo_config={"mcp": {static_name: {"type": "remote", "url": "http://static/mcp"}}}
    )
    body = seed()
    body.tool_accesses = [mcp(grant_name)]
    with patch(f"{APP}.create_provider") as create, pytest.raises(HTTPException) as error:
        await agent._seed_agent_session_state(body)
    assert error.value.status_code == 422
    assert "collide" in error.value.detail
    create.assert_not_called()


def test_distinct_grants_cannot_share_sanitized_tool_namespace() -> None:
    body = seed()
    body.tool_accesses = [mcp("task.tools"), mcp("task_tools")]
    with pytest.raises(HTTPException) as error:
        _make_model_server_agent()._mcp_grants(body)
    assert error.value.status_code == 422 and "collide" in error.value.detail


@pytest.mark.parametrize("required", [True, False])
def test_direct_http_is_still_unsupported_but_optional_access_does_not_hide_mcp(required: bool) -> None:
    agent = _make_model_server_agent()
    session = state()
    grant = mcp()
    session.request.tool_accesses = [
        DirectHTTPToolAccess(name="http-tools", required=required, base_url="http://resources:9001"),
        grant,
    ]
    if required:
        with pytest.raises(HTTPException) as error:
            agent._mcp_grants(session.request)
        assert error.value.status_code == 422 and "http-tools" in error.value.detail
    else:
        assert agent._mcp_grants(session.request) == [grant]
        assert set(agent._sandbox_config(session, rollout_id="rollout-1-a2")["mcp"]) == {grant.name}


@pytest.mark.parametrize("borrowed", [True, False])
async def test_seed_preflights_required_mcp_in_borrowed_or_owned_sandbox(borrowed: bool) -> None:
    agent = _make_model_server_agent(
        kilo_version="7.4.15",
        sandbox_provider="sandbox",
        sandbox_config={"image": "test", "metadata": {"team": "benchmark", "gym-agent": "wrong"}},
    )
    body = seed(borrowed=borrowed)
    body.tool_accesses = [mcp(), mcp("optional", required=False)]
    sandbox, provider = AsyncMock(), AsyncMock()
    sandbox.sandbox_id = "sbx-1"
    sandbox.exec.return_value = SandboxExecResult(return_code=0, stdout="", stderr="")
    events: list[str] = []

    async def installed(**kwargs: object) -> None:
        events.append("installed")

    async def preflight(**kwargs: object) -> None:
        events.append("preflight")

    with (
        patch(f"{APP}.resolve_provider_config", return_value={}),
        patch(f"{APP}.resolve_provider_metadata", return_value={"owner": "provider", "team": "default"}),
        patch(f"{APP}.get_global_config_dict", return_value={}),
        patch(f"{APP}.create_provider", return_value=provider),
        patch(f"{APP}.AsyncSandbox", return_value=sandbox) as api,
        patch.object(KiloSandboxSession, "install_runtime", side_effect=installed),
        patch.object(KiloSandboxSession, "prepare_mcp", side_effect=preflight) as prepare,
    ):
        api.connect = AsyncMock(return_value=sandbox)
        session = await agent._seed_agent_session_state(body)

    assert events == ["installed", "preflight"]
    assert session.required_mcp_servers == ("task.tools",)
    assert "Bearer task-token" in session.mcp_header_values
    assert session.session.owns_sandbox is not borrowed
    config = json.loads(prepare.await_args.kwargs["config"])
    assert set(config["mcp"]) == {"task.tools", "optional"}
    assert config["mcp"]["task.tools"]["headers"]["Authorization"] == "Bearer task-token"
    if borrowed:
        api.connect.assert_awaited_once_with({"id": "task"}, provider=provider)
        sandbox.start.assert_not_awaited()
    else:
        api.connect.assert_not_awaited()
        spec = sandbox.start.await_args.args[0]
        assert spec.workdir == "/app"
        assert spec.ttl_s == 2 * agent.config.setup_timeout + agent.config.timeout + 600
        assert spec.metadata == {
            "owner": "provider",
            "team": "benchmark",
            "gym-agent": "kilocode",
            "gym-session": "session-1",
            "gym-rollout": "rollout-1-a2",
        }
        assert agent.config.sandbox_config["metadata"]["gym-agent"] == "wrong"


@pytest.mark.parametrize("borrowed", [True, False])
@pytest.mark.parametrize("cleanup_fails", [True, False])
async def test_failed_required_mcp_seed_cleans_up_and_retains_state_when_cleanup_fails(
    borrowed: bool, cleanup_fails: bool
) -> None:
    agent = _make_model_server_agent(kilo_version="7.4.15", sandbox_provider="sandbox")
    body = seed(borrowed=borrowed)
    body.tool_accesses = [mcp()]
    sandbox = AsyncMock()
    sandbox.exec.return_value = SandboxExecResult(return_code=0, stdout="", stderr="")
    preflight_error = RuntimeError("required MCP server unavailable")
    with (
        patch(f"{APP}.resolve_provider_config", return_value={}),
        patch(f"{APP}.resolve_provider_metadata", return_value={}),
        patch(f"{APP}.get_global_config_dict", return_value={}),
        patch(f"{APP}.create_provider", return_value=AsyncMock()),
        patch(f"{APP}.AsyncSandbox", return_value=sandbox) as api,
        patch.object(KiloSandboxSession, "install_runtime", new_callable=AsyncMock),
        patch.object(KiloSandboxSession, "prepare_mcp", side_effect=preflight_error),
        patch.object(
            KiloSandboxSession,
            "close",
            side_effect=RuntimeError("cleanup unavailable") if cleanup_fails else None,
        ) as close,
        pytest.raises(AgentSessionSetupError if cleanup_fails else RuntimeError) as error,
    ):
        api.connect = AsyncMock(return_value=sandbox)
        await agent._seed_agent_session_state(body)
    close.assert_awaited_once_with(timeout=agent.config.session_close_timeout_seconds)
    if cleanup_fails:
        assert error.value.error is preflight_error
        assert error.value.state.session.owns_sandbox is not borrowed
        assert error.value.state.required_mcp_servers == ("task.tools",)
    else:
        assert error.value is preflight_error


@pytest.mark.parametrize("ready", [False, True])
async def test_completed_model_text_requires_mcp_receipt(ready: bool) -> None:
    agent = _make_model_server_agent()
    session = state()
    session.required_mcp_servers = ("task.tools",)
    session.request.tool_accesses = [mcp()]
    session.session.execute = AsyncMock(
        return_value=KiloArtifacts(
            stdout=TEXT, stderr="", exit_code=0, task_environment_ready=True, required_mcp_ready=ready
        )
    )
    body = NeMoGymResponseCreateParamsNonStreaming(input="hi")
    if ready:
        response = await agent._sandbox_response(session, body, rollout_id="rollout-1-a2")
        assert response.status == "completed"
        assert response.output[0].content[0].text == "partial"
    else:
        with pytest.raises(RuntimeError, match="required MCP"):
            await agent._sandbox_response(session, body, rollout_id="rollout-1-a2")


@pytest.mark.parametrize(
    "ready_file,expected",
    [
        ('["task.tools"]', True),
        ('["other"]', False),
        ('["task.tools", "unexpected"]', False),
        ('{"task.tools": "connected"}', False),
        ("invalid JSON", False),
        (FileNotFoundError("missing receipt"), False),
    ],
)
async def test_mcp_receipt_must_name_exact_required_servers(ready_file: str | Exception, expected: bool) -> None:
    session = state()
    session.required_mcp_servers = ("task.tools",)
    with patch(f"{TRANSPORT}.read_text", side_effect=[ready_file]):
        assert await session._mcp_ready() is expected


@pytest.mark.parametrize(
    "cleanup",
    [
        None,
        receipt(error="cleanup failed"),
        receipt(timed_out=True),
        receipt(return_code=1),
        {**receipt(), "cleanup_confirmed": False},
    ],
)
async def test_preflight_requires_successful_cleanup_before_activation(cleanup: dict | None) -> None:
    session = state()
    session.required_mcp_servers = ("task.tools",)
    session.session.cleanup = cleanup
    session.session.execute = AsyncMock(
        return_value=KiloArtifacts(stdout="", stderr="", exit_code=0, required_mcp_ready=True)
    )
    with patch.object(session.session, "begin_activation", new_callable=AsyncMock) as begin:
        with pytest.raises(RuntimeError, match="required MCP"):
            await session.prepare_mcp(config="{}", timeout=10, close_timeout=5)
    begin.assert_not_awaited()


@pytest.mark.parametrize("exit_code,ready", [(1, True), (0, False)])
async def test_preflight_rejects_failed_cli_or_missing_required_receipt(exit_code: int, ready: bool) -> None:
    session = state()
    session.required_mcp_servers = ("task.tools",)
    session.session.execute = AsyncMock(
        return_value=KiloArtifacts(stdout="", stderr="", exit_code=exit_code, required_mcp_ready=ready)
    )
    with patch.object(session.session, "begin_activation", new_callable=AsyncMock) as begin:
        with pytest.raises(RuntimeError, match="required MCP"):
            await session.prepare_mcp(config="{}", timeout=10, close_timeout=5)
    begin.assert_not_awaited()


async def test_preflight_runs_without_model_and_rearms_activation_only_after_success() -> None:
    session = state()
    session.required_mcp_servers = ("task.tools",)
    session.session.execute = AsyncMock(
        return_value=KiloArtifacts(stdout="", stderr="", exit_code=0, required_mcp_ready=True)
    )
    with (
        patch.object(session, "stage_activation", new_callable=AsyncMock) as stage,
        patch.object(session.session, "begin_activation", new_callable=AsyncMock) as begin,
    ):
        await session.prepare_mcp(config='{"mcp": {}}', timeout=10, close_timeout=5)
        await session.session.execute.await_args.kwargs["stage_activation"]()
    assert stage.await_args.kwargs["command"] == [*session.kilo_argv, "debug", "agent", "build"]
    assert stage.await_args.kwargs["mcp_preflight"] is True
    assert stage.await_args.kwargs["config"] == '{"mcp": {}}'
    begin.assert_awaited_once_with(session_dir=f"{session.directory}/activation", timeout=5)


async def test_collected_diagnostics_redact_grant_headers_before_logging(caplog: pytest.LogCaptureFixture) -> None:
    session = state()
    session.required_mcp_servers = ("task.tools",)
    session.mcp_header_values = ("Bearer task-token", "task-token")
    session.session.cleanup = receipt(return_code=-15, timed_out=True)
    session.session.sandbox.exec.side_effect = [
        SandboxExecResult(return_code=0, stdout="MCP auth failed: Bearer task-token", stderr=""),
        SandboxExecResult(return_code=1, stdout="", stderr="no database"),
    ]
    with patch(
        f"{TRANSPORT}.read_text",
        side_effect=[
            TEXT,
            "MCP request Authorization=Bearer task-token; token=task-token",
            "1",
            '{"hostname": "sandbox", "pid": 42}',
            "ready\n",
            '["task.tools"]',
        ],
    ):
        artifacts = await session.collect_artifacts()
    assert "task-token" not in artifacts.stderr + artifacts.log_tail + caplog.text
    assert "[REDACTED]" in artifacts.stderr and "[REDACTED]" in artifacts.log_tail
    assert artifacts.required_mcp_ready is True


def test_diagnostics_redact_json_escaped_credentials() -> None:
    session = state()
    token = 'Bearer quoted-"credential-\\-é'
    session.mcp_header_values = (token,)
    for diagnostic in (token, json.dumps(token), json.dumps(token, ensure_ascii=False)):
        redacted = session.redact_diagnostics(diagnostic)
        assert "credential" not in redacted
        assert "[REDACTED]" in redacted


@pytest.mark.parametrize("suffix", ["x" * 4075, "界" * 1366, "x" * 4050 + " another-sensitive-header"])
async def test_log_tail_redacts_credential_crossing_tail_boundary(suffix: str) -> None:
    session = state()
    token = "Bearer abcdefghijklmnopqrstuvwxyz1234567890"
    session.mcp_header_values = (token, "another-sensitive-header")
    log = ("older log\n" * 500 + token + "\n" + suffix).encode()

    async def tail(command: str, **kwargs: object) -> SandboxExecResult:
        count = int(re.search(r"tail -c (\d+)", command).group(1))
        return SandboxExecResult(return_code=0, stdout=log[-count:].decode(errors="replace"), stderr="")

    session.session.sandbox.exec.side_effect = tail
    diagnostic = await session._read_log_tail()
    assert token[-20:] not in diagnostic
    assert "another-sensitive-header" not in diagnostic
    assert len(diagnostic.encode()) <= 4098  # A split UTF-8 sequence may become a replacement character.
    assert diagnostic.endswith(session.redact_diagnostics(suffix)[-100:])
